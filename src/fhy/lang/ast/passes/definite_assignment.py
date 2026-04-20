"""Definite-assignment analysis and validation over the FhY AST."""

__all__ = [
    "validate_definite_assignment",
]

from dataclasses import dataclass, field

from fhy_core import (
    CompilerPass,
    FunctionSymbolTableFrame,
    Identifier,
    SymbolTable,
    TypeQualifier,
    register_pass,
)

from fhy.lang.ast.cfg import (
    CFGNode,
    CFGNodeKind,
    ControlFlowGraph,
    build_cfg,
)
from fhy.lang.ast.error import FhYSemanticsError
from fhy.lang.ast.node import (
    ArrayAccessExpression,
    DeclarationStatement,
    ExpressionStatement,
    ForAllStatement,
    Function,
    FunctionExpression,
    IdentifierExpression,
    Module,
    Operation,
    Procedure,
    ReturnStatement,
    SelectionStatement,
)

from .identifier_collector import collect_identifiers
from .liveness_analysis import LivenessAnalysis, LivenessResult


@dataclass(frozen=True)
class _FunctionDefiniteAssignment:
    """Per-CFG dataflow result: definite-assignment sets at entry/exit of each node."""

    cfg: ControlFlowGraph
    live_in: dict[int, frozenset[Identifier]] = field(default_factory=dict)
    live_out: dict[int, frozenset[Identifier]] = field(default_factory=dict)


def _get_function_universe(
    function: Function, cfg: ControlFlowGraph
) -> frozenset[Identifier]:
    universe: set[Identifier] = {arg.name for arg in function.args}
    for node in cfg.nodes:
        if node.statement is None:
            continue
        if isinstance(node.statement, DeclarationStatement):
            universe.add(node.statement.variable_name)
    return frozenset(universe)


def _get_identifiers_definitely_assigned_on_func_entry(
    function: Function,
) -> frozenset[Identifier]:
    return frozenset(
        arg.name
        for arg in function.args
        if arg.qualified_type.type_qualifier != TypeQualifier.OUTPUT
    )


def _get_procedure_call_output_writes(
    call: FunctionExpression, symbol_table: SymbolTable, namespace: Identifier
) -> frozenset[Identifier]:
    """Identifiers assigned by a procedure call via its OUTPUT arguments.

    Array-access arguments are skipped (they would partially write into an
    existing aggregate, which we do not model here).

    """
    if not isinstance(call.function, IdentifierExpression):
        return frozenset()
    try:
        frame = symbol_table.get_frame_from_namespace(
            namespace, call.function.identifier
        )
    except Exception:
        return frozenset()
    if not isinstance(frame, FunctionSymbolTableFrame):
        return frozenset()
    defs: set[Identifier] = set()
    for arg_expression, (qualifier, _) in zip(call.args, frame.signature):
        if qualifier != TypeQualifier.OUTPUT:
            continue
        if isinstance(arg_expression, IdentifierExpression):
            defs.add(arg_expression.identifier)
        elif isinstance(arg_expression, ArrayAccessExpression) and isinstance(
            arg_expression.array_expression, IdentifierExpression
        ):
            defs.add(arg_expression.array_expression.identifier)
    return frozenset(defs)


def _statement_gen(
    node: CFGNode, symbol_table: SymbolTable, namespace: Identifier
) -> frozenset[Identifier]:
    if node.statement is None:
        return frozenset()
    statement = node.statement
    if isinstance(statement, DeclarationStatement):
        if statement.expression is None:
            return frozenset()
        return frozenset({statement.variable_name})
    elif isinstance(statement, ExpressionStatement):
        if statement.left is not None:
            if isinstance(statement.left, IdentifierExpression):
                return frozenset({statement.left.identifier})
            if isinstance(statement.left, ArrayAccessExpression) and isinstance(
                statement.left.array_expression, IdentifierExpression
            ):
                # An array-access write partially defines the array. We treat
                # it as a full definition so idiomatic per-element
                # initialization inside a ForAll is accepted.
                return frozenset({statement.left.array_expression.identifier})
            return frozenset()
        # Bare `procedure(...);` statements: OUTPUT arguments bound to
        # identifier actuals are written through to the caller's scope.
        if isinstance(statement.right, FunctionExpression):
            return _get_procedure_call_output_writes(
                statement.right, symbol_table, namespace
            )
    else:
        return frozenset()


def _compute_definite_assignment(
    function: Function,
    cfg: ControlFlowGraph,
    symbol_table: SymbolTable,
) -> _FunctionDefiniteAssignment:
    """Forward MUST dataflow: definitely-assigned identifiers at every node."""
    universe = _get_function_universe(function, cfg)
    entry_assigned = _get_identifiers_definitely_assigned_on_func_entry(function)

    gen: dict[int, frozenset[Identifier]] = {
        node.id: _statement_gen(node, symbol_table, function.name) for node in cfg.nodes
    }

    in_sets: dict[int, frozenset[Identifier]] = {}
    out_sets: dict[int, frozenset[Identifier]] = {}
    for node in cfg.nodes:
        if node.kind == CFGNodeKind.ENTRY:
            in_sets[node.id] = frozenset()
            out_sets[node.id] = entry_assigned
        else:
            in_sets[node.id] = universe
            out_sets[node.id] = universe

    changed = True
    while changed:
        changed = False
        for node in cfg.nodes:
            if node.kind == CFGNodeKind.ENTRY:
                continue
            predecessors = cfg.get_predecessors(node)
            if not predecessors:
                new_in: frozenset[Identifier] = frozenset()
            else:
                iterator = iter(predecessors)
                new_in = out_sets[next(iterator).id]
                for predecessor in iterator:
                    new_in = new_in & out_sets[predecessor.id]
            new_out = new_in | gen[node.id]
            if new_in != in_sets[node.id] or new_out != out_sets[node.id]:
                in_sets[node.id] = new_in
                out_sets[node.id] = new_out
                changed = True

    return _FunctionDefiniteAssignment(cfg=cfg, live_in=in_sets, live_out=out_sets)


def _collect_read_identifiers(  # noqa: C901, PLR0912
    node: CFGNode, symbol_table: SymbolTable, namespace: Identifier
) -> frozenset[Identifier]:
    """Identifiers read by the statement at `node` (excluding pure writes).

    For an assignment `x = rhs`, the RHS is read and `x` is not. For an
    array-indexed write `b[i] = rhs`, the index identifiers (`i`) and the
    RHS are read; `b` itself is not read. For a bare procedure call, OUTPUT
    arguments bound to an identifier actual are writes (handled by
    `_statement_gen`) and so are excluded from reads.

    """
    statement = node.statement
    if statement is None:
        return frozenset()
    elif isinstance(statement, DeclarationStatement):
        if statement.expression is None:
            return frozenset()
        else:
            return collect_identifiers(statement.expression)
    elif isinstance(statement, ExpressionStatement):
        if statement.left is None and isinstance(statement.right, FunctionExpression):
            return _collect_procedure_call_reads(
                statement.right, symbol_table, namespace
            )
        reads: set[Identifier] = set(collect_identifiers(statement.right))
        if statement.left is None:
            return frozenset(reads)
        if isinstance(statement.left, ArrayAccessExpression):
            for index in statement.left.indices:
                reads.update(collect_identifiers(index))
        return frozenset(reads)
    elif isinstance(statement, ReturnStatement):
        return collect_identifiers(statement.expression)
    elif isinstance(statement, ForAllStatement):
        return collect_identifiers(statement.index)
    elif isinstance(statement, SelectionStatement):
        return collect_identifiers(statement.condition)
    else:
        return frozenset()


def _collect_procedure_call_reads(
    call: FunctionExpression, symbol_table: SymbolTable, namespace: Identifier
) -> frozenset[Identifier]:
    """Identifier reads inside a bare procedure-call statement.

    OUTPUT identifier arguments are writes, not reads. Non-OUTPUT arguments
    contribute their full identifier set as reads (indices, shape, etc).

    """
    if not isinstance(call.function, IdentifierExpression):
        return collect_identifiers(call)
    try:
        frame = symbol_table.get_frame_from_namespace(
            namespace, call.function.identifier
        )
    except Exception:
        return collect_identifiers(call)
    reads: set[Identifier] = set()
    if not isinstance(frame, FunctionSymbolTableFrame):
        return collect_identifiers(call)
    for arg_expression, (qualifier, _) in zip(call.args, frame.signature):
        if qualifier == TypeQualifier.OUTPUT and isinstance(
            arg_expression, IdentifierExpression
        ):
            continue
        reads.update(collect_identifiers(arg_expression))
    return frozenset(reads)


def _is_qualifier_requiring_definite_assignment(
    identifier: Identifier, symbol_table: SymbolTable, namespace: Identifier
) -> bool:
    from fhy_core import IndexType, VariableSymbolTableFrame

    try:
        frame = symbol_table.get_frame_from_namespace(namespace, identifier)
    except Exception:
        return False
    if not isinstance(frame, VariableSymbolTableFrame):
        return False
    if frame.type_qualifier not in {TypeQualifier.TEMP, TypeQualifier.OUTPUT}:
        return False
    # Index variables are implicitly bound by their enclosing `forall` or
    # reduction and are never assigned by an explicit statement.
    if isinstance(frame.type, IndexType):
        return False
    return True


@register_pass(
    "fhy_ast_definite_assignment_validator",
    "Validates definite assignment of OUTPUT arguments and TEMP reads.",
)
class _DefiniteAssignmentValidator(CompilerPass[Module, None]):
    """Definite-assignment validator as a standalone compiler pass.

    Uses :func:`build_cfg` per function together with the existing
    :class:`LivenessAnalysis` to restrict the use-before-def check to reads
    that are also live (unused reads cannot cause observable violations).

    """

    _symbol_table: SymbolTable

    def __init__(self, symbol_table: SymbolTable) -> None:
        super().__init__()
        self._symbol_table = symbol_table

    def get_noop_output(self, ir: Module) -> None:
        _ = ir

    def run_pass(self, ir: Module) -> None:
        liveness = LivenessAnalysis().run(ir)
        for statement in ir.statements:
            if isinstance(statement, Procedure | Operation):
                self._validate_function(statement, liveness)

    def _validate_function(  # noqa: C901
        self, function: Function, liveness: LivenessResult
    ) -> None:
        cfg = build_cfg(function)
        result = _compute_definite_assignment(function, cfg, self._symbol_table)

        # 1. Every OUTPUT argument must be definitely assigned at the exit.
        assigned_at_exit = result.live_in.get(cfg.exit.id, frozenset())
        for argument in function.args:
            if argument.qualified_type.type_qualifier != TypeQualifier.OUTPUT:
                continue
            if argument.name not in assigned_at_exit:
                raise FhYSemanticsError(
                    f"OUTPUT argument {argument.name.name_hint!r} of "
                    f"{function.name.name_hint!r} is not assigned on every "
                    "control flow path.",
                    argument.provenance,
                )

        # 2. Every read of a TEMP (or OUTPUT-arg-being-read) must be preceded
        # by a definite assignment. Restrict the check to identifiers that
        # are also live at the point of the read — unused reads cannot cause
        # observable use-before-def.
        for node in cfg.nodes:
            if node.kind != CFGNodeKind.STATEMENT or node.statement is None:
                continue
            read_identifiers = _collect_read_identifiers(
                node, self._symbol_table, function.name
            )
            if not read_identifiers:
                continue
            assigned_in = result.live_in.get(node.id, frozenset())
            live_in = liveness.live_in.get(id(node.statement), frozenset())
            for identifier in read_identifiers:
                if identifier in assigned_in:
                    continue
                if identifier not in live_in:
                    continue
                if not _is_qualifier_requiring_definite_assignment(
                    identifier, self._symbol_table, function.name
                ):
                    continue
                raise FhYSemanticsError(
                    f"Variable {identifier.name_hint!r} may be read before it "
                    "is assigned.",
                    node.statement.provenance,
                )


def validate_definite_assignment(ast: Module, symbol_table: SymbolTable) -> None:
    """Check for definite assignment violations in the AST.

    Args:
        ast: The AST to validate.
        symbol_table: The symbol table to use.

    Raises:
        FhYSemanticsError: When an OUTPUT argument is not definitely
            assigned at function exit or a read may precede its definition
            on some path. Wrapped in `PassExecutionError` when raised from
            inside the compiler-pass framework.

    """
    validator = _DefiniteAssignmentValidator(symbol_table)
    validator(ast)
