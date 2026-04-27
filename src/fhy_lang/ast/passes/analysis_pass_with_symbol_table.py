"""Analysis pass with symbol table."""

__all__ = [
    "AnalysisPassWithSymbolTable",
]

from abc import ABC

from fhy_core import (
    AnalysisVisitablePass,
    Identifier,
    Stack,
    SymbolTable,
    SymbolTableFrame,
)

from fhy_lang.ast.node import ForAllStatement, Module, Node, Operation, Procedure
from fhy_lang.builtins import BUILTINS_NAMESPACE_NAME


class AnalysisPassWithSymbolTable(AnalysisVisitablePass[Node], ABC):
    """Analysis pass with an attached symbol table and namespace stack."""

    _symbol_table: SymbolTable
    _namespace_stack: Stack[Identifier]

    def __init__(self, symbol_table: SymbolTable) -> None:
        super().__init__()
        self._symbol_table = symbol_table
        self._namespace_stack = Stack[Identifier]()
        self._namespace_stack.push(BUILTINS_NAMESPACE_NAME)

    def get_noop_output(self, ir: Node) -> None:
        raise RuntimeError("This pass does not support a noop output.")

    @property
    def current_namespace(self) -> Identifier:
        return self._namespace_stack.peek()

    def get_frame_from_namespace(
        self, namespace: Identifier, name: Identifier
    ) -> SymbolTableFrame:
        """Get a symbol table frame from a specific namespace.

        Args:
            namespace: The namespace to get the frame from.
            name: The name of the symbol to get the frame for.

        Returns:
            The frame for the given symbol in the given namespace.

        """
        return self._symbol_table.get_frame_from_namespace(namespace, name)

    def _push_namespace(self, namespace: Identifier) -> None:
        self._namespace_stack.push(namespace)

    def _pop_namespace(self) -> None:
        self._namespace_stack.pop()

    def before_visit_module(self, node: Module) -> None:
        self._push_namespace(node.name)

    def after_visit_module(self, node: Module) -> None:
        self._pop_namespace()
        if len(self._namespace_stack) != 1:
            raise RuntimeError(
                "Expected the namespace stack to only contain "
                f"{BUILTINS_NAMESPACE_NAME} after visiting module node."
            )

    def before_visit_procedure(self, node: Procedure) -> None:
        self._push_namespace(node.name)

    def after_visit_procedure(self, node: Procedure) -> None:
        self._pop_namespace()

    def before_visit_operation(self, node: Operation) -> None:
        self._push_namespace(node.name)

    def after_visit_operation(self, node: Operation) -> None:
        self._pop_namespace()

    def before_visit_for_all_statement(self, node: ForAllStatement) -> None:
        self._push_namespace(node.name)

    def after_visit_for_all_statement(self, node: ForAllStatement) -> None:
        self._pop_namespace()
