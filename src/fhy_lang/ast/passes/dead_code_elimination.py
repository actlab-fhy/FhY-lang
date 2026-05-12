"""Dead code elimination over the FhY AST."""

__all__ = [
    "DeadCodeEliminationPass",
]

import logging
from collections import Counter

from fhy_core import (
    Identifier,
    Stack,
    SymbolTable,
    TypeQualifier,
    VariableSymbolTableFrame,
    get_logger,
    register_pass,
)

from fhy_lang.ast.node import (
    ArrayAccessExpression,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    ForAllStatement,
    IdentifierExpression,
    Module,
    Node,
    Operation,
    Procedure,
    QualifiedType,
    SelectionStatement,
    Statement,
)

from .expression_side_effect_analysis import ExpressionSideEffectAnalysis
from .identifier_collector import collect_identifiers
from .liveness_analysis import LivenessAnalysis, LivenessResult
from .transformer import Statements, Transformer

_logger: logging.Logger = get_logger(__name__)


def _count_identifier_occurrences_in_body(
    body: tuple[Statement, ...], counts: Counter[Identifier]
) -> None:
    for statement in body:
        if isinstance(statement, ForAllStatement):
            for identifier in collect_identifiers(statement.index):
                counts[identifier] += 1
            _count_identifier_occurrences_in_body(statement.body, counts)
        elif isinstance(statement, SelectionStatement):
            for identifier in collect_identifiers(statement.condition):
                counts[identifier] += 1
            _count_identifier_occurrences_in_body(statement.true_body, counts)
            _count_identifier_occurrences_in_body(statement.false_body, counts)
        else:
            for identifier in collect_identifiers(statement):
                counts[identifier] += 1


def _count_identifier_occurrences(module: Module) -> Counter[Identifier]:
    """Count identifier occurrences across atomic statements of every function.

    An identifier's count reflects how many atomic statements mention it:
    :class:`ExpressionStatement`, :class:`DeclarationStatement`, and
    :class:`ReturnStatement` contribute one count per identifier they
    mention. :class:`ForAllStatement` and :class:`SelectionStatement`
    contribute via their ``index`` / ``condition``, with their nested
    bodies walked recursively. An uninitialized ``TEMP`` declaration is
    dead when its variable's count is exactly 1 (it only appears in its
    own declaration).

    """
    counts: Counter[Identifier] = Counter()
    for top_level in module.statements:
        if isinstance(top_level, Procedure | Operation):
            _count_identifier_occurrences_in_body(top_level.body, counts)
    return counts


def _get_assignment_target_identifier(
    left_hand_side: Expression,
) -> Identifier | None:
    """Return the identifier written by a simple assignment LHS, if any."""
    if isinstance(left_hand_side, IdentifierExpression):
        return left_hand_side.identifier
    if isinstance(left_hand_side, ArrayAccessExpression) and isinstance(
        left_hand_side.array_expression, IdentifierExpression
    ):
        return left_hand_side.array_expression.identifier
    return None


@register_pass(
    "fhy_ast_dead_code_elimination",
    "Removes assignments and declarations whose value is not live.",
)
class DeadCodeEliminationPass(Transformer):
    """Remove dead writes to ``TEMP`` variables using liveness analysis.

    Depends on :class:`LivenessAnalysis` via an injected
    :class:`AnalysisManager` so that the pass can participate in a fixpoint
    group where liveness is re-computed after each iteration that changes
    the IR.

    """

    _symbol_table: SymbolTable
    _namespace_stack: Stack[Identifier]
    _live_out: dict[int, frozenset[Identifier]]
    _identifier_use_counts: Counter[Identifier]
    _removed_count: int

    def __init__(
        self,
        symbol_table: SymbolTable,
    ) -> None:
        super().__init__()
        self._symbol_table = symbol_table
        self._live_out = {}
        self._identifier_use_counts = Counter()
        self._removed_count = 0

    def run_pass(self, ir: Node) -> Node:
        if not isinstance(ir, Module):
            raise TypeError(
                f"{type(self).__name__} requires a Module input; "
                f"got {type(ir).__name__}."
            )
        liveness: LivenessResult = self.get_analysis(LivenessAnalysis, ir)
        self._live_out = dict(liveness.live_out)
        self._identifier_use_counts = _count_identifier_occurrences(ir)
        self._removed_count = 0
        _logger.debug(
            "DCE inputs: %d live-out entries, %d distinct identifier(s) counted.",
            len(self._live_out),
            len(self._identifier_use_counts),
        )
        return super().run_pass(ir)

    def did_change(self, input_ir: Node, output: Node) -> bool:
        _ = (input_ir, output)
        return self._removed_count > 0

    def visit_expression_statement(self, node: ExpressionStatement) -> Statements:
        if self._is_dead_assignment(node):
            self._removed_count += 1
            _logger.debug("Removed dead assignment to TEMP variable.")
            return []
        return super().visit_expression_statement(node)

    def visit_declaration_statement(self, node: DeclarationStatement) -> Statements:
        if self._is_dead_declaration(node):
            self._removed_count += 1
            _logger.debug(
                "Removed dead declaration of TEMP variable %s.",
                node.variable_name,
            )
            return []
        return super().visit_declaration_statement(node)

    def visit_qualified_type(self, node: QualifiedType) -> QualifiedType:
        return node

    def _is_dead_assignment(self, node: ExpressionStatement) -> bool:
        if node.left is None:
            return False
        target = _get_assignment_target_identifier(node.left)
        if target is None:
            return False
        if not self._is_temp_variable(target):
            return False
        if self.get_analysis(ExpressionSideEffectAnalysis, node.right):
            return False
        return target not in self._live_out.get(id(node), frozenset())

    def _is_dead_declaration(self, node: DeclarationStatement) -> bool:
        if node.variable_type.type_qualifier != TypeQualifier.TEMP:
            return False
        if node.expression is None:
            return self._identifier_use_counts.get(node.variable_name, 0) <= 1
        if self.get_analysis(ExpressionSideEffectAnalysis, node.expression):
            return False
        return node.variable_name not in self._live_out.get(id(node), frozenset())

    def _is_temp_variable(self, identifier: Identifier) -> bool:
        frame = self._symbol_table.get_frame_from_namespace(
            self.get_current_namespace(), identifier
        )
        return (
            isinstance(frame, VariableSymbolTableFrame)
            and frame.type_qualifier == TypeQualifier.TEMP
        )
