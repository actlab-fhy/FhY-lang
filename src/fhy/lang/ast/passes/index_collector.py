"""Index collection passes."""

from collections.abc import Callable

from fhy_core import AnalysisVisitablePass, Identifier, register_pass

from fhy.lang.ast.node import core
from fhy.lang.ast.node.expression import FunctionExpression, IdentifierExpression


@register_pass(
    "fhy_ast_index_collector", "Collects all the indices used in an AST expression."
)
class IndexCollector(AnalysisVisitablePass[core.Expression]):
    """Collect all the indices used in an AST expression."""

    _is_identifier_index: Callable[[Identifier], bool]
    _indices: set[Identifier]

    def __init__(self, is_identifier_index_func: Callable[[Identifier], bool]) -> None:
        super().__init__()
        self._indices = set()
        self._is_identifier_index = is_identifier_index_func

    @property
    def indices(self) -> frozenset[Identifier]:
        return frozenset(self._indices)

    def visit_identifier_expression(self, node: IdentifierExpression) -> None:
        if self._is_identifier_index(node.identifier):
            self._indices.add(node.identifier)


def collect_indices(
    node: core.Expression,
    is_identifier_index: Callable[[Identifier], bool],
) -> frozenset[Identifier]:
    """Collect all the indices used in an AST expression.

    Args:
        node: The AST expression node to collect indices from.
        is_identifier_index: A function that determines if an identifier is an index.

    Returns:
        The set of indices used in the AST expression.

    """
    index_collector = IndexCollector(is_identifier_index)
    index_collector(node)
    return index_collector.indices


# NOTE: if FhY supports indices in reduction's parameters that are not just the
#       identifier itself, this pass must be modified
@register_pass(
    "fhy_ast_reduced_index_collector",
    "Collects all the indices used in an AST expression that are reduced.",
)
class ReducedIndexCollector(AnalysisVisitablePass[core.Expression]):
    """Collect all the indices used in an AST expression that are reduced."""

    _is_identifier_index: Callable[[Identifier], bool]
    _reduced_indices: set[Identifier]

    def __init__(self, is_identifier_index_func: Callable[[Identifier], bool]) -> None:
        super().__init__()
        self._reduced_indices = set()
        self._is_identifier_index = is_identifier_index_func

    @property
    def reduced_indices(self) -> frozenset[Identifier]:
        return frozenset(self._reduced_indices)

    def visit_function_expression(self, node: FunctionExpression) -> None:
        for index in node.indices:
            if not isinstance(index, IdentifierExpression):
                raise RuntimeError(f"Index {index} is not an identifier expression.")
            if self._is_identifier_index(index.identifier):
                self._reduced_indices.add(index.identifier)


def collect_reduced_indices(
    node: core.Expression,
    is_identifier_index: Callable[[Identifier], bool],
) -> frozenset[Identifier]:
    """Collect all the indices used in an AST expression that are reduced.

    Args:
        node: The AST expression node to collect indices from.
        is_identifier_index: A function that
            determines if an identifier is an index.

    Returns:
        The set of indices used in the AST expression that are reduced.

    """
    reduced_index_collector = ReducedIndexCollector(is_identifier_index)
    reduced_index_collector(node)
    return reduced_index_collector.reduced_indices
