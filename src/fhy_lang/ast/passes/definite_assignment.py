"""Definite-assignment analysis and validation over the FhY AST.

The validator orchestrates two independent analyses that share a
validator entry point but are otherwise decoupled:

1. Scalar identifier definite assignment — a forward MUST dataflow over
   each function's CFG that tracks which variables have been definitely
   assigned at every control-flow point. It enforces that scalar
   OUTPUT arguments are written on every path and that every read of a
   TEMP (or an OUTPUT being read) is preceded by a definite assignment.

2. Shape-symbolic coverage of OUTPUT arrays — a recursive walk over
   each function's body that, for every OUTPUT array argument, builds
   a Z3-friendly boolean predicate over fresh point variables
   asserting "this index position is written on every path", and uses
   satisfiability to prove that the predicate covers the array's
   declared shape.

Each analysis is encapsulated in its own helper class, so a new
definite-assignment-style check can be added by introducing a new
analysis class plus one ``_check_...`` method on the validator.
"""

__all__ = [
    "DefiniteAssignmentValidator",
]

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from fhy_core import (
    CompilerPass,
    DiagnosticLevel,
    FunctionSymbolTableFrame,
    Identifier,
    IndexType,
    NumericalType,
    SymbolTable,
    SymbolTableError,
    SymbolTableFrame,
    SymbolType,
    TypeQualifier,
    VariableSymbolTableFrame,
    get_logger,
    is_satisfiable,
    register_pass,
)
from fhy_core import (
    Expression as CoreExpression,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)
from fhy_core import (
    LiteralExpression as CoreLiteralExpression,
)
from fhy_core import (
    collect_identifiers as core_collect_identifiers,
)
from frozendict import frozendict

from fhy_lang.ast.cfg import (
    CFGNode,
    CFGNodeKind,
    ControlFlowGraph,
    build_cfg,
)
from fhy_lang.ast.node import (
    Argument,
    ArrayAccessExpression,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    ForAllStatement,
    FunctionExpression,
    IdentifierExpression,
    Module,
    Operation,
    Procedure,
    ReturnStatement,
    SelectionStatement,
    Statement,
)
from fhy_lang.ast.shape import narrow_shape

from .ast_to_core_expression_converter import (
    convert_ast_expression_to_core_expression,
)
from .identifier_collector import collect_identifiers
from .liveness_analysis import LivenessAnalysis, LivenessResult
from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)

_FunctionDefinition = Procedure | Operation


def _is_array_output_argument(argument: Argument) -> bool:
    """True when ``argument`` is an OUTPUT with non-scalar numerical shape."""
    if argument.qualified_type.type_qualifier != TypeQualifier.OUTPUT:
        return False
    else:
        base_type = argument.qualified_type.base_type
        return isinstance(base_type, NumericalType) and not base_type.is_scalar()


def _try_get_callee_frame(
    call: FunctionExpression,
    symbol_table: SymbolTable,
    namespace: Identifier,
) -> FunctionSymbolTableFrame | None:
    """Resolve a call site to its callee's signature frame, if possible."""
    if not isinstance(call.function, IdentifierExpression):
        return None
    try:
        frame = symbol_table.get_frame_from_namespace(
            namespace, call.function.identifier
        )
    except SymbolTableError:
        return None
    if isinstance(frame, FunctionSymbolTableFrame):
        return frame
    else:
        return None


def _qualifier_requires_definite_assignment(
    identifier: Identifier,
    symbol_table: SymbolTable,
    namespace: Identifier,
) -> bool:
    """True when reading ``identifier`` before assignment is a violation.

    Only TEMPs and OUTPUTs are subject to the check; INPUTs and PARAMs
    are assigned on entry. Index-typed variables are implicitly bound
    by their enclosing ``forall`` or reduction and are never assigned
    by an explicit statement.

    """
    try:
        frame = symbol_table.get_frame_from_namespace(namespace, identifier)
    except SymbolTableError:
        return False
    if not isinstance(frame, VariableSymbolTableFrame):
        return False
    elif frame.type_qualifier not in {TypeQualifier.TEMP, TypeQualifier.OUTPUT}:
        return False
    elif isinstance(frame.type, IndexType):
        return False
    else:
        return True


def _conjoin_all(clauses: Iterable[CoreExpression]) -> CoreExpression:
    """Left-fold ``LOGICAL_AND`` over ``clauses``; empty ⇒ ``TRUE``."""
    iterator = iter(clauses)
    try:
        result = next(iterator)
    except StopIteration:
        return CoreLiteralExpression(True)
    for clause in iterator:
        result = CoreExpression.logical_and(result, clause)
    return result


@dataclass(frozen=True)
class _ScalarDefiniteAssignmentResult:
    """Per-CFG-node sets of identifiers that are definitely assigned.

    Sets are keyed by CFG-node id. ``definitely_assigned_in[n]`` is the
    must-set on entry to node ``n``; ``definitely_assigned_out[n]`` is
    the must-set on exit.

    """

    cfg: ControlFlowGraph
    definitely_assigned_in: frozendict[int, frozenset[Identifier]]
    definitely_assigned_out: frozendict[int, frozenset[Identifier]]


class _ScalarDefiniteAssignmentAnalysis:
    """Forward MUST dataflow for scalar-identifier definite assignment.

    Treats an array-indexed LHS as a full definition of its base
    identifier — the canonical FhY idiom for per-element initialization
    inside a ``forall``. The stronger shape-sensitive coverage check
    for OUTPUT arrays is performed separately by
    :class:`_ArrayCoverageAnalysis`.

    """

    _function: _FunctionDefinition
    _symbol_table: SymbolTable
    _cfg: ControlFlowGraph

    def __init__(
        self,
        function: _FunctionDefinition,
        symbol_table: SymbolTable,
    ) -> None:
        self._function = function
        self._symbol_table = symbol_table
        self._cfg = build_cfg(function)

    @property
    def cfg(self) -> ControlFlowGraph:
        return self._cfg

    def run(self) -> _ScalarDefiniteAssignmentResult:
        universe = self._get_universe()
        entry_assigned = self._get_entry_assigned_identifiers()
        gen_sets = {node.id: self._compute_gen_set(node) for node in self._cfg.nodes}

        in_sets: dict[int, frozenset[Identifier]] = {}
        out_sets: dict[int, frozenset[Identifier]] = {}
        for node in self._cfg.nodes:
            if node.kind == CFGNodeKind.ENTRY:
                in_sets[node.id] = frozenset()
                out_sets[node.id] = entry_assigned
            else:
                in_sets[node.id] = universe
                out_sets[node.id] = universe

        changed = True
        while changed:
            changed = False
            for node in self._cfg.nodes:
                if node.kind == CFGNodeKind.ENTRY:
                    continue
                predecessors = self._cfg.get_predecessors(node)
                if not predecessors:
                    new_in: frozenset[Identifier] = frozenset()
                else:
                    predecessor_iter = iter(predecessors)
                    new_in = out_sets[next(predecessor_iter).id]
                    for predecessor in predecessor_iter:
                        new_in = new_in & out_sets[predecessor.id]
                new_out = new_in | gen_sets[node.id]
                if new_in != in_sets[node.id] or new_out != out_sets[node.id]:
                    in_sets[node.id] = new_in
                    out_sets[node.id] = new_out
                    changed = True

        return _ScalarDefiniteAssignmentResult(
            cfg=self._cfg,
            definitely_assigned_in=frozendict(in_sets),
            definitely_assigned_out=frozendict(out_sets),
        )

    def get_read_identifiers(self, node: CFGNode) -> frozenset[Identifier]:
        """Return the identifiers read by the statement wrapped by ``node``.

        Public for the validator's use-before-def check, which shares
        this analysis's view of reads so both sides of the comparison
        agree on which operands count as reads.

        For an assignment ``x = rhs`` the RHS is read and ``x`` is not;
        for an array-indexed write ``b[i] = rhs`` the index identifiers
        and the RHS are read but ``b`` itself is not. In a bare
        procedure-call statement, OUTPUT arguments bound to an
        identifier actual are writes (tracked by :meth:`_compute_gen_set`)
        and are excluded from reads.

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
            return self._compute_expression_statement_reads(statement)
        elif isinstance(statement, ReturnStatement):
            return collect_identifiers(statement.expression)
        elif isinstance(statement, ForAllStatement):
            return collect_identifiers(statement.index)
        elif isinstance(statement, SelectionStatement):
            return collect_identifiers(statement.condition)
        else:
            return frozenset()

    def _get_universe(self) -> frozenset[Identifier]:
        universe: set[Identifier] = {argument.name for argument in self._function.args}
        for node in self._cfg.nodes:
            if isinstance(node.statement, DeclarationStatement):
                universe.add(node.statement.variable_name)
        return frozenset(universe)

    def _get_entry_assigned_identifiers(self) -> frozenset[Identifier]:
        return frozenset(
            argument.name
            for argument in self._function.args
            if argument.qualified_type.type_qualifier != TypeQualifier.OUTPUT
        )

    def _compute_gen_set(self, node: CFGNode) -> frozenset[Identifier]:
        statement = node.statement
        if statement is None:
            return frozenset()
        elif isinstance(statement, DeclarationStatement):
            if statement.expression is None:
                return frozenset()
            else:
                return frozenset({statement.variable_name})
        elif isinstance(statement, ExpressionStatement):
            return self._compute_expression_statement_gen_set(statement)
        else:
            return frozenset()

    def _compute_expression_statement_gen_set(
        self, statement: ExpressionStatement
    ) -> frozenset[Identifier]:
        if statement.left is not None:
            if isinstance(statement.left, IdentifierExpression):
                return frozenset({statement.left.identifier})
            elif isinstance(statement.left, ArrayAccessExpression) and isinstance(
                statement.left.array_expression, IdentifierExpression
            ):
                # An array-access write partially defines the array.
                # Accept it as a full definition here so per-element
                # initialization inside a forall is not spuriously
                # flagged; shape-sensitive coverage is handled by
                # _ArrayCoverageAnalysis.
                return frozenset({statement.left.array_expression.identifier})
            else:
                return frozenset()
        elif isinstance(statement.right, FunctionExpression):
            return self._collect_call_output_writes(statement.right)
        else:
            return frozenset()

    def _collect_call_output_writes(
        self, call: FunctionExpression
    ) -> frozenset[Identifier]:
        frame = _try_get_callee_frame(call, self._symbol_table, self._function.name)
        if frame is None:
            return frozenset()
        definitions: set[Identifier] = set()
        for argument_expression, (qualifier, _) in zip(call.args, frame.signature):
            if qualifier != TypeQualifier.OUTPUT:
                continue
            if isinstance(argument_expression, IdentifierExpression):
                definitions.add(argument_expression.identifier)
            elif isinstance(argument_expression, ArrayAccessExpression) and isinstance(
                argument_expression.array_expression, IdentifierExpression
            ):
                definitions.add(argument_expression.array_expression.identifier)
        return frozenset(definitions)

    def _compute_expression_statement_reads(
        self, statement: ExpressionStatement
    ) -> frozenset[Identifier]:
        if statement.left is None:
            if isinstance(statement.right, FunctionExpression):
                return self._collect_call_reads(statement.right)
            else:
                return collect_identifiers(statement.right)
        else:
            reads: set[Identifier] = set(collect_identifiers(statement.right))
            if isinstance(statement.left, ArrayAccessExpression):
                for index in statement.left.indices:
                    reads.update(collect_identifiers(index))
            return frozenset(reads)

    def _collect_call_reads(self, call: FunctionExpression) -> frozenset[Identifier]:
        frame = _try_get_callee_frame(call, self._symbol_table, self._function.name)
        if frame is None:
            return collect_identifiers(call)
        reads: set[Identifier] = set()
        for argument_expression, (qualifier, _) in zip(call.args, frame.signature):
            if qualifier == TypeQualifier.OUTPUT and isinstance(
                argument_expression, IdentifierExpression
            ):
                continue
            reads.update(collect_identifiers(argument_expression))
        return frozenset(reads)


@dataclass(frozen=True)
class _WriteRegion:
    """A hypercube covered by a single write to an array.

    ``lower_bounds[j]`` and ``upper_bounds[j]`` are inclusive bounds on
    dimension ``j``. If the index at position ``j`` is an index-typed
    identifier, the bounds come from its declared range; for any other
    core-convertible expression, the interval collapses to the point
    ``[e, e]``.

    """

    lower_bounds: tuple[CoreExpression, ...]
    upper_bounds: tuple[CoreExpression, ...]

    def build_covers_predicate(
        self, point_expressions: tuple[CoreExpression, ...]
    ) -> CoreExpression:
        return _conjoin_all(
            CoreExpression.logical_and(lower <= point, point <= upper)
            for point, lower, upper in zip(
                point_expressions, self.lower_bounds, self.upper_bounds
            )
        )


@dataclass(frozen=True)
class _Coverage:
    """Must-coverage state threaded through the recursive body walk.

    Immutable by design — the walk produces a new :class:`_Coverage`
    at every step rather than mutating in place, so the merge at a
    :class:`SelectionStatement` is a straightforward intersection of
    two independently derived values.

    """

    predicate: CoreExpression
    fully_written: bool = False
    uncharacterisable_writes: tuple[Statement, ...] = ()

    @staticmethod
    def create_empty() -> "_Coverage":
        """Return the bottom of the lattice: nothing covered."""
        return _Coverage(predicate=CoreLiteralExpression(False))

    def add_region(
        self,
        region: _WriteRegion,
        point_expressions: tuple[CoreExpression, ...],
    ) -> "_Coverage":
        if self.fully_written:
            return self
        else:
            return replace(
                self,
                predicate=CoreExpression.logical_or(
                    self.predicate,
                    region.build_covers_predicate(point_expressions),
                ),
            )

    def mark_fully_written(self) -> "_Coverage":
        return replace(self, fully_written=True)

    def add_uncharacterisable(self, statement: Statement) -> "_Coverage":
        return replace(
            self,
            uncharacterisable_writes=(*self.uncharacterisable_writes, statement),
        )

    def intersect(self, other: "_Coverage") -> "_Coverage":
        """Merge two branches: a point is covered only if both cover it."""
        return _Coverage(
            predicate=CoreExpression.logical_and(self.predicate, other.predicate),
            fully_written=self.fully_written and other.fully_written,
            uncharacterisable_writes=(
                self.uncharacterisable_writes + other.uncharacterisable_writes
            ),
        )


@dataclass(frozen=True)
class _ArrayCoverageResult:
    """Outcome of shape-symbolic coverage analysis for one OUTPUT array.

    ``complete`` is ``True`` when every valid index position is proven
    written on every control-flow path, ``False`` when a counter-example
    exists, and ``None`` when the SMT solver returned unknown.
    ``uncharacterisable_writes`` lists writes whose region we could not
    represent symbolically; when non-empty, failures attribute their
    diagnostic to the first such statement.

    """

    complete: bool | None
    uncharacterisable_writes: tuple[Statement, ...] = ()


class _ArrayCoverageAnalysis:
    """Per-argument shape-symbolic coverage analyzer.

    One instance analyzes exactly one OUTPUT array argument of one
    function. The walk threads an immutable :class:`_Coverage` through
    the body, handling each statement kind as follows:

    - an :class:`ExpressionStatement` whose LHS writes the target
      array contributes a hypercube (or marks the array fully written
      when the LHS is the array identifier);
    - a bare procedure call with an OUTPUT argument bound to the
      target contributes similarly;
    - a :class:`ForAllStatement` is traversed straight-line — an
      index variable's declared range is already baked into the
      hypercube of any inner write, so the loop structure itself
      does not affect coverage;
    - a :class:`SelectionStatement` intersects the two branches
      (must-cover on both paths).

    """

    _function: _FunctionDefinition
    _symbol_table: SymbolTable
    _argument: Argument
    _target: Identifier
    _shape: Sequence[CoreExpression]
    _points: tuple[CoreExpression, ...]

    def __init__(
        self,
        function: _FunctionDefinition,
        symbol_table: SymbolTable,
        argument: Argument,
    ) -> None:
        base_type = argument.qualified_type.base_type
        assert isinstance(base_type, NumericalType)
        self._function = function
        self._symbol_table = symbol_table
        self._argument = argument
        self._target = argument.name
        self._shape = narrow_shape(base_type.shape)
        self._points = self._make_fresh_points(argument.name, len(base_type.shape))

    def run(self) -> _ArrayCoverageResult:
        coverage = self._analyze_block(self._function.body, _Coverage.create_empty())
        complete = self._is_coverage_complete(coverage)
        return _ArrayCoverageResult(
            complete=complete,
            uncharacterisable_writes=coverage.uncharacterisable_writes,
        )

    def _is_coverage_complete(self, coverage: _Coverage) -> bool | None:
        if coverage.fully_written:
            return True
        one = CoreLiteralExpression(1)
        in_domain = _conjoin_all(
            CoreExpression.logical_and(point >= one, point <= dim_size)
            for point, dim_size in zip(self._points, self._shape)
        )
        uncovered = CoreExpression.logical_and(
            in_domain, coverage.predicate.logical_not()
        )
        identifiers = set(core_collect_identifiers(uncovered))
        symbol_types = dict.fromkeys(identifiers, SymbolType.INT)
        sat = is_satisfiable(identifiers, uncovered, symbol_types)
        _logger.debug(
            "Array coverage SMT check: %d symbolic var(s), satisfiable=%s.",
            len(identifiers),
            sat,
        )
        if sat is None:
            return None
        else:
            return not sat

    @staticmethod
    def _make_fresh_points(
        array_name: Identifier, rank: int
    ) -> tuple[CoreExpression, ...]:
        return tuple(
            CoreIdentifierExpression(
                Identifier(f"__coverage_point__{array_name.name_hint}__{j}")
            )
            for j in range(rank)
        )

    def _analyze_block(
        self, body: Sequence[Statement], coverage: _Coverage
    ) -> _Coverage:
        for statement in body:
            if coverage.fully_written:
                break
            coverage = self._analyze_statement(statement, coverage)
        return coverage

    def _analyze_statement(
        self, statement: Statement, coverage: _Coverage
    ) -> _Coverage:
        if isinstance(statement, ExpressionStatement):
            return self._analyze_expression_statement(statement, coverage)
        elif isinstance(statement, ForAllStatement):
            return self._analyze_block(statement.body, coverage)
        elif isinstance(statement, SelectionStatement):
            true_coverage = self._analyze_block(statement.true_body, coverage)
            false_coverage = self._analyze_block(statement.false_body, coverage)
            return true_coverage.intersect(false_coverage)
        else:
            # DeclarationStatement / ReturnStatement contribute no writes.
            return coverage

    def _analyze_expression_statement(
        self, statement: ExpressionStatement, coverage: _Coverage
    ) -> _Coverage:
        if statement.left is not None:
            return self._analyze_direct_write(statement, coverage)
        elif isinstance(statement.right, FunctionExpression):
            return self._analyze_call_output_writes(
                statement, statement.right, coverage
            )
        else:
            return coverage

    def _analyze_direct_write(
        self, statement: ExpressionStatement, coverage: _Coverage
    ) -> _Coverage:
        left = statement.left
        if isinstance(left, IdentifierExpression) and left.identifier == self._target:
            return coverage.mark_fully_written()
        elif (
            isinstance(left, ArrayAccessExpression)
            and isinstance(left.array_expression, IdentifierExpression)
            and left.array_expression.identifier == self._target
        ):
            region = self._build_region_for_access(left)
            if region is None:
                return coverage.add_uncharacterisable(statement)
            else:
                return coverage.add_region(region, self._points)
        else:
            return coverage

    def _analyze_call_output_writes(
        self,
        statement: ExpressionStatement,
        call: FunctionExpression,
        coverage: _Coverage,
    ) -> _Coverage:
        frame = _try_get_callee_frame(call, self._symbol_table, self._function.name)
        if frame is None:
            return coverage
        for argument_expression, (qualifier, _) in zip(call.args, frame.signature):
            if qualifier != TypeQualifier.OUTPUT:
                continue
            if (
                isinstance(argument_expression, IdentifierExpression)
                and argument_expression.identifier == self._target
            ):
                return coverage.mark_fully_written()
            elif (
                isinstance(argument_expression, ArrayAccessExpression)
                and isinstance(
                    argument_expression.array_expression, IdentifierExpression
                )
                and argument_expression.array_expression.identifier == self._target
            ):
                region = self._build_region_for_access(argument_expression)
                if region is None:
                    coverage = coverage.add_uncharacterisable(statement)
                else:
                    coverage = coverage.add_region(region, self._points)
        return coverage

    def _build_region_for_access(
        self, array_access: ArrayAccessExpression
    ) -> _WriteRegion | None:
        lower_bounds: list[CoreExpression] = []
        upper_bounds: list[CoreExpression] = []
        for index_expression in array_access.indices:
            bounds = self._get_index_range(index_expression)
            if bounds is None:
                return None
            lower, upper = bounds
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        return _WriteRegion(
            lower_bounds=tuple(lower_bounds),
            upper_bounds=tuple(upper_bounds),
        )

    def _get_index_range(
        self, index_expression: Expression
    ) -> tuple[CoreExpression, CoreExpression] | None:
        if isinstance(index_expression, IdentifierExpression):
            frame = self._try_get_frame(index_expression.identifier)
            if isinstance(frame, VariableSymbolTableFrame) and isinstance(
                frame.type, IndexType
            ):
                return frame.type.lower_bound, frame.type.upper_bound
        try:
            core_expression = convert_ast_expression_to_core_expression(
                index_expression
            )
        except NotImplementedError:
            return None
        else:
            return core_expression, core_expression

    def _try_get_frame(self, identifier: Identifier) -> SymbolTableFrame | None:
        try:
            return self._symbol_table.get_frame_from_namespace(
                self._function.name, identifier
            )
        except SymbolTableError:
            return None


@register_pass(
    "fhy_ast_definite_assignment_validator",
    "Validates definite assignment of OUTPUT arguments and TEMP reads.",
)
class DefiniteAssignmentValidator(CompilerPass[Module, None]):
    """Definite-assignment validator.

    Runs both the scalar identifier analysis (see
    :class:`_ScalarDefiniteAssignmentAnalysis`) and the shape-symbolic
    coverage analysis (see :class:`_ArrayCoverageAnalysis`) on each
    function, emitting one diagnostic per violation so a single run
    surfaces every problem.

    """

    _symbol_table: SymbolTable

    def __init__(self, symbol_table: SymbolTable) -> None:
        super().__init__()
        self._symbol_table = symbol_table

    def get_noop_output(self, ir: Module) -> None:
        _ = ir

    def run_pass(self, ir: Module) -> None:
        liveness = LivenessAnalysis().run(ir)
        function_count = 0
        for statement in ir.statements:
            if isinstance(statement, Procedure | Operation):
                _logger.debug(
                    "Definite assignment: validating function %s.", statement.name
                )
                self._validate_function(statement, liveness)
                function_count += 1
        _logger.info(
            "Definite assignment validation complete: %d function(s) checked.",
            function_count,
        )

    def _validate_function(
        self, function: _FunctionDefinition, liveness: LivenessResult
    ) -> None:
        scalar_analysis = _ScalarDefiniteAssignmentAnalysis(
            function, self._symbol_table
        )
        scalar_result = scalar_analysis.run()
        self._check_scalar_outputs_assigned_at_exit(function, scalar_result)
        self._check_reads_are_preceded_by_definite_assignment(
            function, scalar_analysis, scalar_result, liveness
        )
        self._check_output_arrays_fully_written(function)

    def _check_scalar_outputs_assigned_at_exit(
        self,
        function: _FunctionDefinition,
        scalar_result: _ScalarDefiniteAssignmentResult,
    ) -> None:
        """Every scalar OUTPUT argument must be definitely assigned at the exit.

        Array OUTPUT arguments are validated by
        :meth:`_check_output_arrays_fully_written` instead; its
        shape-sensitive coverage check subsumes the scalar-identifier
        check for them.

        """
        assigned_at_exit = scalar_result.definitely_assigned_in.get(
            scalar_result.cfg.exit.id, frozenset()
        )
        for argument in function.args:
            if argument.qualified_type.type_qualifier != TypeQualifier.OUTPUT:
                continue
            if _is_array_output_argument(argument):
                continue
            if argument.name in assigned_at_exit:
                continue
            self._report_scalar_output_not_assigned(function, argument)

    def _check_reads_are_preceded_by_definite_assignment(
        self,
        function: _FunctionDefinition,
        scalar_analysis: _ScalarDefiniteAssignmentAnalysis,
        scalar_result: _ScalarDefiniteAssignmentResult,
        liveness: LivenessResult,
    ) -> None:
        """Every read of a TEMP (or an OUTPUT being read) must be preceded by a write.

        Restricted to identifiers that are also live at the point of
        the read: unused reads cannot cause observable use-before-def.

        """
        for node in scalar_result.cfg.nodes:
            if node.kind != CFGNodeKind.STATEMENT or node.statement is None:
                continue
            read_identifiers = scalar_analysis.get_read_identifiers(node)
            if not read_identifiers:
                continue
            assigned_in = scalar_result.definitely_assigned_in.get(node.id, frozenset())
            live_in = liveness.live_in.get(id(node.statement), frozenset())
            for identifier in read_identifiers:
                if identifier in assigned_in:
                    continue
                if identifier not in live_in:
                    continue
                if not _qualifier_requires_definite_assignment(
                    identifier, self._symbol_table, function.name
                ):
                    continue
                self._report_use_before_def(node.statement, identifier)

    def _check_output_arrays_fully_written(self, function: _FunctionDefinition) -> None:
        """Every OUTPUT array argument must have every element written.

        Builds a shape-symbolic must-coverage predicate over fresh point
        variables and checks that the predicate covers the array's
        declared shape on every control-flow path.

        """
        for argument in function.args:
            if not _is_array_output_argument(argument):
                continue
            result = _ArrayCoverageAnalysis(
                function, self._symbol_table, argument
            ).run()
            if result.complete is True:
                continue
            self._report_array_coverage_failure(function, argument, result)

    def _report_scalar_output_not_assigned(
        self, function: _FunctionDefinition, argument: Argument
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "semantic error",
                f"OUTPUT argument {argument.name.name_hint!r} of "
                f"{function.name.name_hint!r} is not assigned on every "
                "control flow path.",
                argument.provenance,
            ),
        )

    def _report_use_before_def(
        self, statement: Statement, identifier: Identifier
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "semantic error",
                f"Variable {identifier.name_hint!r} may be read before it is assigned.",
                statement.provenance,
            ),
        )

    def _report_array_coverage_failure(
        self,
        function: _FunctionDefinition,
        argument: Argument,
        result: _ArrayCoverageResult,
    ) -> None:
        provenance = (
            result.uncharacterisable_writes[0].provenance
            if result.uncharacterisable_writes
            else argument.provenance
        )
        if result.uncharacterisable_writes:
            message = (
                f"OUTPUT array argument {argument.name.name_hint!r} of "
                f"{function.name.name_hint!r} contains a write whose covered "
                "region the analysis could not represent symbolically; "
                "every element may not be written."
            )
        elif result.complete is None:
            message = (
                f"OUTPUT array argument {argument.name.name_hint!r} of "
                f"{function.name.name_hint!r} could not be proven fully "
                "written: the SMT solver returned unknown for the "
                "coverage query."
            )
        else:
            message = (
                f"OUTPUT array argument {argument.name.name_hint!r} of "
                f"{function.name.name_hint!r} is not fully written on every "
                "control-flow path; some element may remain undefined."
            )
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message("semantic error", message, provenance),
        )
