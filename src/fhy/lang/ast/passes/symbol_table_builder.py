"""Construct a symbol table from a FhY AST module."""

__all__ = [
    "build_symbol_table",
]


from fhy_core import (
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
    VisitablePass,
    register_pass,
)
from fhy_core import IdentifierExpression as CoreIdentifierExpression
from fhy_core import (
    collect_identifiers as collect_core_identifiers,
)

from fhy.error import FhYSemanticsError
from fhy.lang.ast.node import (
    Argument,
    DeclarationStatement,
    Import,
    Node,
    Operation,
    Procedure,
    core,
    expression,
)
from fhy.lang.builtins import BUILTIN_LANG_IDENTIFIERS, BUILTINS_NAMESPACE_NAME


@register_pass(
    "fhy_ast_symbol_table_builder",
    "Builds a symbol table for the given AST module node.",
)
class _SymbolTableBuilder(VisitablePass[Node, None]):
    """Builds a symbol table for the given AST module node.

    The class will throw an exception if a variable is used before being declared or if
    a variable is declared more than once within the same namespace.

    Note:
        This builder pass for the symbol table only supports namespaces created by
        a new module or new operation/procedure. Nested scopes created by ForAll
        loop bodies and If/Else bodies are not supported and will be treated as
        the same namespace as the parent operation/procedure.

    Raises:
        FhYSemanticsError: A variable is used before being declared (undefined), or
            the variable is defined again (redefined), within the current namespace.
        RuntimeError: Unexpected behavior, indicating improper use.
        TypeError: Received wrong argument (node) type.

    """

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

    def _assert_symbol_not_defined(self, symbol: Identifier) -> None:
        if self._is_symbol_defined(symbol):
            msg = "Symbol Identifier previously declared (redefined) in current"
            raise FhYSemanticsError(f"{msg} namespace: {symbol.name_hint}")

    def _assert_symbol_defined(self, symbol: Identifier) -> None:
        if not self._is_symbol_defined(symbol):
            msg = "Undeclared Symbol Identifier used in current namespace"
            raise FhYSemanticsError(f"{msg}: {symbol.name_hint}")

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

    def visit_module(self, node: core.Module) -> None:
        self._push_namespace(node.name)
        for statement in node.statements:
            self.visit(statement)
        self._pop_namespace()

        if len(self._namespace_stack) != 1:
            error_message: str = "Expected the namespace stack to only contain "
            error_message += f"{BUILTINS_NAMESPACE_NAME} after visiting "
            error_message += "module node."
            raise RuntimeError(error_message)

    def visit_import(self, node: Import) -> None:
        self._assert_symbol_not_defined(node.name)
        import_frame = ImportSymbolTableFrame(name=node.name)
        self._add_symbol(node.name, import_frame)

    def visit_procedure(self, node: Procedure) -> None:
        self._assert_symbol_not_defined(node.name)
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
        for arg in node.args:
            self.visit(arg)
        for statement in node.body:
            self.visit(statement)
        self._pop_namespace()

    def visit_operation(self, node: Operation) -> None:
        self._assert_symbol_not_defined(node.name)
        op_frame = FunctionSymbolTableFrame(
            name=node.name,
            keyword=FunctionKeyword.OPERATION,
            signature=tuple(
                [
                    (arg.qualified_type.type_qualifier, arg.qualified_type.base_type)
                    for arg in node.args
                ]
                + [(node.return_type.type_qualifier, node.return_type.base_type)]
            ),
        )
        self._add_symbol(node.name, op_frame)
        self._push_namespace(node.name)
        for arg in node.args:
            self.visit(arg)
        for statement in node.body:
            self.visit(statement)
        self.visit(node.return_type)
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
        self._assert_symbol_not_defined(node.variable_name)
        var_frame = VariableSymbolTableFrame(
            name=node.variable_name,
            type=node.variable_type.base_type,
            type_qualifier=node.variable_type.type_qualifier,
        )
        self._add_symbol(node.variable_name, var_frame)

    def visit_identifier_expression(
        self, node: expression.IdentifierExpression
    ) -> None:
        self._assert_symbol_defined(node.identifier)

    def visit_core_identifier_expression(self, node: CoreIdentifierExpression) -> None:
        self._assert_symbol_defined(node.identifier)


def build_symbol_table(node: core.Module) -> SymbolTable:
    """Build a symbol table from a module AST node.

    Argument:
        node: FhY module AST node

    Returns:
         Symbol table cataloging all variables from the provided module,
            by appropriate frame.

    Raises:
        FhYSemanticsError: A variable is used before being declared (undefined), or
            the variable is defined again (redefined), within the current namespace.
        RuntimeError: Unexpected behavior, indicating improper use.
        TypeError: Received wrong argument (node) type.

    """
    builder = _SymbolTableBuilder()
    builder(node)

    return builder.symbol_table
