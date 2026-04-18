"""Construct a symbol table from a FhY AST module."""

__all__ = [
    "build_symbol_table",
    "FhYSymbolTableBuilderError",
]

from fhy_core import (
    AnalysisVisitablePass,
    CoreDataType,
    FunctionKeyword,
    FunctionSymbolTableFrame,
    Identifier,
    ImportSymbolTableFrame,
    NumericalType,
    PrimitiveDataType,
    Stack,
    SymbolTable,
    SymbolTableFrame,
    TypeQualifier,
    VariableSymbolTableFrame,
    register_error,
    register_pass,
)
from fhy_core import (
    collect_identifiers as collect_core_identifiers,
)

from fhy.lang.ast.node import (
    Argument,
    DeclarationStatement,
    ForAllStatement,
    Import,
    Module,
    Node,
    Operation,
    Procedure,
)
from fhy.lang.builtins import BUILTIN_LANG_IDENTIFIERS, BUILTINS_NAMESPACE_NAME


@register_error
class FhYSymbolTableBuilderError(RuntimeError):
    """Raised when a symbol table builder error is detected."""

    def __init__(self, error_message: str) -> None:
        super().__init__(
            f"An error occurred while building the symbol table: {error_message}"
        )


@register_pass(
    "fhy_ast_symbol_table_builder",
    "Builds a symbol table for the given AST module node.",
)
class _SymbolTableBuilder(AnalysisVisitablePass[Node]):
    """Builds a symbol table for the given AST module node."""

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
    def symbol_table(
        self,
    ) -> SymbolTable:
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
        self._symbol_table.add_symbol(self._namespace_stack.peek(), symbol, frame)

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
        if self._is_symbol_defined(node.name):
            raise FhYSymbolTableBuilderError(
                f"Symbol {node.name.name_hint} is already defined."
            )
        import_frame = ImportSymbolTableFrame(name=node.name)
        self._add_symbol(node.name, import_frame)

    def before_visit_procedure(self, node: Procedure) -> None:
        if self._is_symbol_defined(node.name):
            raise FhYSymbolTableBuilderError(
                f"Symbol {node.name.name_hint} is already defined."
            )
        proc_frame = FunctionSymbolTableFrame(
            name=node.name,
            keyword=FunctionKeyword.PROCEDURE,
            signature=tuple(
                (arg.qualified_type.type_qualifier, arg.qualified_type.base_type)
                for arg in node.args
            ),
        )
        self._add_symbol(node.name, proc_frame)
        self._push_namespace(node.name)

    def after_visit_procedure(self, node: Procedure) -> None:
        self._pop_namespace()

    def before_visit_operation(self, node: Operation) -> None:
        if self._is_symbol_defined(node.name):
            raise FhYSymbolTableBuilderError(
                f"Symbol {node.name.name_hint} is already defined."
            )
        op_frame = FunctionSymbolTableFrame(
            name=node.name,
            keyword=FunctionKeyword.OPERATION,
            signature=tuple(
                (arg.qualified_type.type_qualifier, arg.qualified_type.base_type)
                for arg in node.args
            ),
        )
        self._add_symbol(node.name, op_frame)
        self._push_namespace(node.name)

    def after_visit_operation(self, node: Operation) -> None:
        self._pop_namespace()

    def visit_argument(self, node: Argument) -> None:
        arg_frame = VariableSymbolTableFrame(
            name=node.name,
            type=node.qualified_type.base_type,
            type_qualifier=node.qualified_type.type_qualifier,
        )
        self._add_symbol(node.name, arg_frame)

        if isinstance(node.qualified_type.base_type, NumericalType):
            shape_dimension_identifiers: set[Identifier] = set()
            for shape in node.qualified_type.base_type.shape:
                shape_dimension_identifiers.update(collect_core_identifiers(shape))

            for dimension in shape_dimension_identifiers:
                if not self._is_symbol_defined(dimension):
                    var_frame = VariableSymbolTableFrame(
                        name=dimension,
                        type=NumericalType(PrimitiveDataType(CoreDataType.UINT64)),
                        type_qualifier=TypeQualifier.PARAM,
                    )
                    self._add_symbol(dimension, var_frame)

    def visit_declaration_statement(self, node: DeclarationStatement) -> None:
        if self._is_symbol_defined(node.variable_name):
            raise FhYSymbolTableBuilderError(
                f"Symbol {node.variable_name.name_hint} is already defined."
            )
        var_frame = VariableSymbolTableFrame(
            name=node.variable_name,
            type=node.variable_type.base_type,
            type_qualifier=node.variable_type.type_qualifier,
        )
        self._add_symbol(node.variable_name, var_frame)

    def before_visit_for_all_statement(self, node: ForAllStatement) -> None:
        self._push_namespace(node.name)

    def after_visit_for_all_statement(self, node: ForAllStatement) -> None:
        self._pop_namespace()


def build_symbol_table(node: Module) -> SymbolTable:
    """Build a symbol table from a module AST node.

    Argument:
        node: FhY module AST node

    Returns:
         Symbol table cataloging all variables from the provided module,
            by appropriate frame.

    Raises:
        FhYSymbolTableBuilderError: An error occurred while building the symbol table.

    """
    builder = _SymbolTableBuilder()
    builder(node)

    return builder.symbol_table
