"""Dead code elimination over the FhY AST."""

__all__ = [
    "DeadCodeEliminationPass",
]

from fhy_core import (
    AnalysisManager,
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
    FunctionExpression,
    IdentifierExpression,
    Module,
    Node,
    QualifiedType,
)
from fhy.lang.builtins import (
    BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS,
)

from .liveness_analysis import LivenessAnalysis, LivenessResult
from .transformer import Statements, Transformer


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

    _analysis_manager: AnalysisManager[Module]
    _symbol_table: SymbolTable
    _namespace_stack: Stack[Identifier]
    _live_out: dict[int, frozenset[Identifier]]
    _removed_count: int

    def __init__(
        self,
        analysis_manager: AnalysisManager[Module],
        symbol_table: SymbolTable,
    ) -> None:
        super().__init__()
        self._analysis_manager = analysis_manager
        self._symbol_table = symbol_table
        self._live_out = {}
        self._removed_count = 0

    def run_pass(self, ir: Module) -> Module:
        liveness: LivenessResult = self._analysis_manager.get(LivenessAnalysis, ir)
        self._live_out = dict(liveness.live_out)
        self._removed_count = 0
        return super().run_pass(ir)

    def did_change(self, input_ir: Module, output: Module) -> bool:
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
        if node.expression is None:
            return False
        elif node.variable_type.type_qualifier != TypeQualifier.TEMP:
            return False
        elif _is_expression_may_have_side_effects(node.expression):
            return False
        else:
            return node.variable_name not in self._live_out.get(id(node), frozenset())

    def _is_temp_variable(self, identifier: Identifier) -> bool:
        frame = self._symbol_table.get_frame_from_namespace(
            self.get_current_namespace(), identifier
        )
        return (
            isinstance(frame, VariableSymbolTableFrame)
            and frame.type_qualifier == TypeQualifier.TEMP
        )
