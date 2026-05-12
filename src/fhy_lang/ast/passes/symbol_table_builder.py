"""Construct a symbol table from a FhY AST module."""

__all__ = [
    "FhYSymbolTableBuilderError",
    "build_symbol_table",
]

import logging

from fhy_core import (
    AnalysisVisitablePass,
    CoreDataType,
    FunctionKeyword,
    FunctionSymbolTableFrame,
    Identifier,
    ImportSymbolTableFrame,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    Stack,
    SymbolTable,
    SymbolTableFrame,
    Type,
    TypeQualifier,
    UnknownProvenance,
    VariableSymbolTableFrame,
    get_logger,
    register_error,
    register_pass,
)
from fhy_core import (
    collect_identifiers as collect_core_identifiers,
)

from fhy_lang.ast.node import (
    Argument,
    DeclarationStatement,
    ForAllStatement,
    Import,
    Module,
    Node,
    Operation,
    Procedure,
)
from fhy_lang.ast.shape import narrow_shape
from fhy_lang.builtins import BUILTIN_LANG_IDENTIFIERS, BUILTINS_NAMESPACE_NAME

_logger: logging.Logger = get_logger(__name__)


def _format_provenance_location(provenance: Provenance | None) -> str | None:
    if provenance is None or isinstance(provenance, UnknownProvenance):
        return None
    return str(provenance)


@register_error
class FhYSymbolTableBuilderError(RuntimeError):
    """Raised when a symbol table builder error is detected."""

    provenance: "Provenance | None"

    def __init__(
        self, error_message: str, provenance: "Provenance | None" = None
    ) -> None:
        self.provenance = provenance
        location = _format_provenance_location(provenance)
        full_message = (
            f"An error occurred while building the symbol table: {error_message}"
        )
        if location is not None:
            full_message = f"{location}: {full_message}"
        super().__init__(full_message)


def _build_function_signature(
    function: Procedure | Operation,
) -> tuple[tuple[TypeQualifier, Type], ...]:
    return tuple(
        (argument.qualified_type.type_qualifier, argument.qualified_type.base_type)
        for argument in function.args
    )


@register_pass(
    "fhy_ast_symbol_table_builder",
    "Builds a symbol table for the given AST module node.",
)
class _SymbolTableBuilder(AnalysisVisitablePass[Node]):
    """Build a symbol table for the given AST module node."""

    _symbol_table: SymbolTable
    _namespace_stack: Stack[Identifier]

    def __init__(self) -> None:
        super().__init__()

        self._symbol_table = SymbolTable()
        self._symbol_table.add_namespace(BUILTINS_NAMESPACE_NAME, None)
        for identifier in BUILTIN_LANG_IDENTIFIERS.values():
            self._symbol_table.add_symbol(
                BUILTINS_NAMESPACE_NAME,
                identifier,
                ImportSymbolTableFrame(name=identifier),
            )

        self._namespace_stack = Stack[Identifier]()
        self._namespace_stack.push(BUILTINS_NAMESPACE_NAME)

    @property
    def symbol_table(self) -> SymbolTable:
        return self._symbol_table

    def get_noop_output(self, ir: Node) -> None:
        raise RuntimeError("This pass does not support a noop output.")

    def _push_namespace(self, namespace_name: Identifier) -> None:
        if len(self._namespace_stack) == 0:
            parent_namespace_name = None
        else:
            parent_namespace_name = self._namespace_stack.peek()

        self._symbol_table.add_namespace(namespace_name, parent_namespace_name)
        self._namespace_stack.push(namespace_name)

    def _pop_namespace(self) -> None:
        self._namespace_stack.pop()

    def _is_symbol_defined(self, symbol: Identifier) -> bool:
        return self._symbol_table.is_symbol_defined_in_namespace(
            self._namespace_stack.peek(), symbol
        )

    def _add_symbol(self, symbol: Identifier, frame: SymbolTableFrame) -> None:
        if len(self._namespace_stack) == 0:
            raise RuntimeError(
                "Expected current namespace to be set before adding a symbol to it."
            )
        namespace = self._namespace_stack.peek()
        self._symbol_table.add_symbol(namespace, symbol, frame)
        _logger.debug(
            "Registered symbol %s in namespace %s (frame=%s).",
            symbol,
            namespace,
            type(frame).__name__,
        )

    def _check_symbol_not_defined(
        self, symbol: Identifier, provenance: Provenance | None
    ) -> None:
        if self._is_symbol_defined(symbol):
            _logger.debug(
                "Symbol redeclaration: %s in namespace %s.",
                symbol,
                self._namespace_stack.peek(),
            )
            raise FhYSymbolTableBuilderError(
                f"Symbol {symbol.name_hint} is already defined.",
                provenance,
            )

    def before_visit_module(self, node: Module) -> None:
        self._push_namespace(node.name)

    def after_visit_module(self, node: Module) -> None:
        self._pop_namespace()
        if len(self._namespace_stack) != 1:
            raise RuntimeError(
                "Expected the namespace stack to only contain "
                f"{BUILTINS_NAMESPACE_NAME} after visiting module node."
            )

    def visit_import(self, node: Import) -> None:
        self._check_symbol_not_defined(node.name, node.provenance)
        self._add_symbol(node.name, ImportSymbolTableFrame(name=node.name))

    def before_visit_procedure(self, node: Procedure) -> None:
        self._check_symbol_not_defined(node.name, node.provenance)
        procedure_frame = FunctionSymbolTableFrame(
            name=node.name,
            keyword=FunctionKeyword.PROCEDURE,
            signature=_build_function_signature(node),
        )
        self._add_symbol(node.name, procedure_frame)
        self._push_namespace(node.name)

    def after_visit_procedure(self, node: Procedure) -> None:
        self._pop_namespace()

    def before_visit_operation(self, node: Operation) -> None:
        self._check_symbol_not_defined(node.name, node.provenance)
        operation_frame = FunctionSymbolTableFrame(
            name=node.name,
            keyword=FunctionKeyword.OPERATION,
            signature=_build_function_signature(node),
        )
        self._add_symbol(node.name, operation_frame)
        self._push_namespace(node.name)

    def after_visit_operation(self, node: Operation) -> None:
        self._pop_namespace()

    def visit_argument(self, node: Argument) -> None:
        argument_frame = VariableSymbolTableFrame(
            name=node.name,
            type=node.qualified_type.base_type,
            type_qualifier=node.qualified_type.type_qualifier,
        )
        self._add_symbol(node.name, argument_frame)

        if isinstance(node.qualified_type.base_type, NumericalType):
            self._add_implicit_shape_parameters(node.qualified_type.base_type)

    def _add_implicit_shape_parameters(self, numerical_type: NumericalType) -> None:
        shape_dimension_identifiers: set[Identifier] = set()
        for shape in narrow_shape(numerical_type.shape):
            shape_dimension_identifiers.update(collect_core_identifiers(shape))

        for dimension in shape_dimension_identifiers:
            if self._is_symbol_defined(dimension):
                continue
            _logger.debug(
                "Implicitly registered shape dim %s as PARAM uint32.", dimension
            )
            dimension_frame = VariableSymbolTableFrame(
                name=dimension,
                type=NumericalType(PrimitiveDataType(CoreDataType.UINT32)),
                type_qualifier=TypeQualifier.PARAM,
            )
            self._add_symbol(dimension, dimension_frame)

    def visit_declaration_statement(self, node: DeclarationStatement) -> None:
        self._check_symbol_not_defined(node.variable_name, node.provenance)
        variable_frame = VariableSymbolTableFrame(
            name=node.variable_name,
            type=node.variable_type.base_type,
            type_qualifier=node.variable_type.type_qualifier,
        )
        self._add_symbol(node.variable_name, variable_frame)

    def before_visit_for_all_statement(self, node: ForAllStatement) -> None:
        self._push_namespace(node.name)

    def after_visit_for_all_statement(self, node: ForAllStatement) -> None:
        self._pop_namespace()


def build_symbol_table(node: Module) -> SymbolTable:
    """Build a symbol table from a module AST node.

    Args:
        node: FhY module AST node.

    Returns:
        Symbol table cataloging all variables from the provided module, by
        appropriate frame.

    Raises:
        FhYSymbolTableBuilderError: An error occurred while building the
            symbol table.

    """
    builder = _SymbolTableBuilder()
    builder(node)
    return builder.symbol_table
