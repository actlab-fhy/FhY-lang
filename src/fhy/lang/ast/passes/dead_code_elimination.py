"""Dead code elimination over the FhY AST."""

__all__ = [
    "DeadCodeEliminationPass",
]

from collections import Counter

from fhy_core import (
    AnalysisVisitablePass,
    Identifier,
    Stack,
    SymbolTable,
    TypeQualifier,
    VariableSymbolTableFrame,
    register_pass,
)

from fhy.lang.ast.node import (
    ArrayAccessExpression,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    ForAllStatement,
    FunctionExpression,
    IdentifierExpression,
    Module,
    Node,
    Operation,
    Procedure,
    QualifiedType,
    SelectionStatement,
    Statement,
)
from fhy.lang.builtins import (
    BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS,
)

from .identifier_collector import collect_identifiers
from .liveness_analysis import LivenessAnalysis, LivenessResult
from .transformer import Statements, Transformer


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
    `ExpressionStatement`, `DeclarationStatement`, and `ReturnStatement`
    contribute one count per identifier they mention; `ForAllStatement` and
    `SelectionStatement` contribute via their `index` / `condition`, with their
    nested bodies walked recursively. An uninitialized TEMP declaration is
    dead when its variable's count is exactly 1 (it only appears in its own
    declaration).

    """
    counts: Counter[Identifier] = Counter()
    for top in module.statements:
        if isinstance(top, Procedure | Operation):
            _count_identifier_occurrences_in_body(top.body, counts)
    return counts


class _FunctionCallFinder(AnalysisVisitablePass[Node]):
    _found: bool

    def __init__(self) -> None:
        super().__init__()
        self._found = False

    @property
    def found(self) -> bool:
        return self._found

    def visit_function_expression(self, node: FunctionExpression) -> None:
        if (
            isinstance(node.function, IdentifierExpression)
            and node.function.identifier
            in BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.values()
        ):
            return
        else:
            self._found = True


def _is_expression_may_have_side_effects(expression: Expression) -> bool:
    finder = _FunctionCallFinder()
    finder(expression)
    return finder.found


@register_pass(
    "fhy_ast_dead_code_elimination",
    "Removes assignments and declarations whose value is not live.",
)
class DeadCodeEliminationPass(Transformer):
    """Remove dead writes to TEMP variables using liveness analysis.

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
        return super().run_pass(ir)

    def did_change(self, input_ir: Node, output: Node) -> bool:
        _ = (input_ir, output)
        return self._removed_count > 0

    def visit_expression_statement(self, node: ExpressionStatement) -> Statements:
        if self._is_dead_assignment(node):
            self._removed_count += 1
            return []
        return super().visit_expression_statement(node)

    def visit_declaration_statement(self, node: DeclarationStatement) -> Statements:
        if self._is_dead_declaration(node):
            self._removed_count += 1
            return []
        return super().visit_declaration_statement(node)

    def visit_qualified_type(self, node: QualifiedType) -> QualifiedType:
        return node

    def _is_dead_assignment(self, node: ExpressionStatement) -> bool:
        if isinstance(node.left, IdentifierExpression):
            target = node.left.identifier
        elif isinstance(node.left, ArrayAccessExpression) and isinstance(
            node.left.array_expression, IdentifierExpression
        ):
            target = node.left.array_expression.identifier
        else:
            return False
        if not self._is_temp_variable(target):
            return False
        elif _is_expression_may_have_side_effects(node.right):
            return False
        else:
            return target not in self._live_out.get(id(node), frozenset())

    def _is_dead_declaration(self, node: DeclarationStatement) -> bool:
        if node.variable_type.type_qualifier != TypeQualifier.TEMP:
            return False
        if node.expression is None:
            return self._identifier_use_counts.get(node.variable_name, 0) <= 1
        if _is_expression_may_have_side_effects(node.expression):
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
