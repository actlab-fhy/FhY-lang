"""Type-check expression statements, declarations, and returns in the AST.

Checks:
    - The right-hand side of expression statements are compatible (with
      promotion) with the left-hand side.
    - The right-hand side of declarations are compatible (with promotion)
      with the declared symbol.
    - The right-hand side of returns are compatible (with promotion) with
      the return type of the current operation.
    - Arguments passed to functions are compatible (with promotion) with
      the arguments of the function.

"""

__all__ = [
    "TypeChecker",
]

from collections.abc import Sequence
from dataclasses import dataclass

from fhy_core import (
    CoreDataType,
    DiagnosticLevel,
    FhYCoreTypeError,
    FunctionKeyword,
    FunctionSymbolTableFrame,
    Identifier,
    ImportSymbolTableFrame,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    SymbolTable,
    SymbolTableError,
    Type,
    VariableSymbolTableFrame,
    promote_primitive_data_types,
    register_pass,
    resolve_literal_core_data_type,
)
from fhy_core import (
    Expression as CoreExpression,
)
from fhy_core import (
    LiteralExpression as CoreLiteralExpression,
)

from fhy.lang.ast.node import (
    ArrayAccessExpression,
    BinaryExpression,
    ComplexLiteral,
    DeclarationStatement,
    Expression,
    ExpressionStatement,
    FloatLiteral,
    ForAllStatement,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    ReturnStatement,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy.lang.builtins import BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .index_collector import collect_indices, collect_reduced_indices
from .utils import format_diagnostic_message


class _TypeCheckError(Exception):
    """Internal flow-control exception for :class:`TypeChecker` helpers.

    Deeply-nested ``_infer_*`` helpers raise this to abandon inference of a
    single top-level statement. The visitor catches it at the statement
    boundary, emits a diagnostic via ``self.report(...)``, and moves on --
    so checking continues across statements even when one fails. This
    exception is *not* exported; it must never escape the module.

    """

    provenance: Provenance | None

    def __init__(self, message: str, provenance: Provenance | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.provenance = provenance


@dataclass(frozen=True)
class _InferredType:
    type: Type
    free_indices: frozenset[Identifier]


_INTEGER_CORE_DATA_TYPES: frozenset[CoreDataType] = frozenset(
    {
        CoreDataType.UINT,
        CoreDataType.INT,
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
        CoreDataType.INT8,
        CoreDataType.INT16,
        CoreDataType.INT32,
        CoreDataType.INT64,
    }
)

_UINT32_MAX_EXCLUSIVE = 1 << 32


def _are_shapes_equivalent(
    shape_a: Sequence[CoreExpression], shape_b: Sequence[CoreExpression]
) -> bool:
    if len(shape_a) != len(shape_b):
        return False
    return all(a.is_structurally_equivalent(b) for a, b in zip(shape_a, shape_b))


def _are_shapes_call_compatible(
    param_shape: Sequence[CoreExpression], actual_shape: Sequence[CoreExpression]
) -> bool:
    """Apply the shape-compatibility rule for function call argument passing.

    FhY does not support broadcasting. Arities must match; per-dimension
    compatibility is:

    - Both sides literal: value must match exactly.
    - Otherwise (at least one symbolic dim): accept. Symbolic dims on the
      callee side name parameters in the callee's namespace, and matching
      them precisely against caller-side symbolic dims (possibly named the
      same thing by coincidence) would require cross-namespace unification
      that FhY does not currently perform.

    """
    if len(param_shape) != len(actual_shape):
        return False
    for param_dim, actual_dim in zip(param_shape, actual_shape):
        if isinstance(param_dim, CoreLiteralExpression) and isinstance(
            actual_dim, CoreLiteralExpression
        ):
            if not param_dim.is_structurally_equivalent(actual_dim):
                return False
    return True


def _is_assignable(target: Type, source: Type) -> bool:
    if target.is_structurally_equivalent(source):
        return True
    elif not isinstance(target, NumericalType) or not isinstance(source, NumericalType):
        return False
    elif not _are_shapes_equivalent(target.shape, source.shape):
        return False
    elif not isinstance(target.data_type, PrimitiveDataType) or not isinstance(
        source.data_type, PrimitiveDataType
    ):
        return False
    try:
        promoted = promote_primitive_data_types(target.data_type, source.data_type)
    except FhYCoreTypeError:
        return False
    return promoted.core_data_type == target.data_type.core_data_type


def _are_element_types_assignable(target: Type, source: Type) -> bool:
    if not isinstance(target, NumericalType) or not isinstance(source, NumericalType):
        return False
    return _is_assignable(
        NumericalType(target.data_type), NumericalType(source.data_type)
    )


def _is_callable_with(param_type: Type, actual_type: Type) -> bool:
    """Return True when ``actual_type`` can be passed where ``param_type`` is declared.

    Same promotion rule as :func:`_is_assignable`, but with
    :func:`_are_shapes_call_compatible` instead of strict
    :func:`_are_shapes_equivalent` -- so symbolic dims in the callee's
    signature accept any matching-arity caller shape.

    """
    if param_type.is_structurally_equivalent(actual_type):
        return True
    if not isinstance(param_type, NumericalType) or not isinstance(
        actual_type, NumericalType
    ):
        return False
    if not _are_shapes_call_compatible(param_type.shape, actual_type.shape):
        return False
    if not isinstance(param_type.data_type, PrimitiveDataType) or not isinstance(
        actual_type.data_type, PrimitiveDataType
    ):
        return False
    try:
        promoted = promote_primitive_data_types(
            param_type.data_type, actual_type.data_type
        )
    except FhYCoreTypeError:
        return False
    return promoted.core_data_type == param_type.data_type.core_data_type


def _is_integer_numerical_type(node_type: Type) -> bool:
    """Return True when the type is a numerical type with an integer core type."""
    return (
        isinstance(node_type, NumericalType)
        and isinstance(node_type.data_type, PrimitiveDataType)
        and node_type.data_type.core_data_type in _INTEGER_CORE_DATA_TYPES
    )


def _pick_weak_literal_target(value: int) -> CoreDataType:
    if 0 <= value < _UINT32_MAX_EXCLUSIVE:
        return CoreDataType.UINT
    return CoreDataType.INT


def _is_reduction_frame(frame: object) -> bool:
    return (
        isinstance(frame, ImportSymbolTableFrame)
        and frame.name in BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.values()
    )


@register_pass(
    "fhy_ast_type_checker",
    "Type-checks assignments, declarations, and returns in the AST.",
)
class TypeChecker(AnalysisPassWithSymbolTable):
    """Type-check assignments, declarations, returns and call expressions.

    Internally the :class:`_InferredType` helpers still raise
    :class:`_TypeCheckError` on local violations; each top-level visitor
    catches it and routes it through :meth:`_report_type_error` so that
    per-statement errors become ERROR diagnostics without terminating the
    walk. Checking then continues on the next statement.

    """

    _bound_forall_indices: set[Identifier]
    _operation_return_types: dict[Identifier, Type]
    _current_return_type: Type | None

    def __init__(self, symbol_table: SymbolTable) -> None:
        super().__init__(symbol_table)
        self._bound_forall_indices = set()
        self._operation_return_types = {}
        self._current_return_type = None

    def before_visit_module(self, node: Module) -> None:
        super().before_visit_module(node)
        for statement in node.statements:
            if isinstance(statement, Operation):
                self._operation_return_types[statement.name] = (
                    statement.return_type.base_type
                )

    def before_visit_operation(self, node: Operation) -> None:
        super().before_visit_operation(node)
        self._current_return_type = node.return_type.base_type

    def after_visit_operation(self, node: Operation) -> None:
        self._current_return_type = None
        super().after_visit_operation(node)

    def before_visit_for_all_statement(self, node: ForAllStatement) -> None:
        super().before_visit_for_all_statement(node)
        if isinstance(node.index, IdentifierExpression):
            self._bound_forall_indices.add(node.index.identifier)

    def after_visit_for_all_statement(self, node: ForAllStatement) -> None:
        if isinstance(node.index, IdentifierExpression):
            self._bound_forall_indices.discard(node.index.identifier)
        super().after_visit_for_all_statement(node)

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if node.left is None:
            # Procedure / operation call used as a bare statement. Its
            # return type (if any) is discarded, but the argument types
            # still need to match the callee's signature.
            if isinstance(node.right, FunctionExpression):
                try:
                    self._check_call_argument_types(node.right)
                except _TypeCheckError as error:
                    self._report_type_error(error)
            return
        try:
            lhs = self._infer_type(node.left)
            rhs = self._infer_type(node.right)
            self._check_compatible(
                lhs,
                rhs,
                context="expression statement",
                provenance=node.provenance,
            )
        except _TypeCheckError as error:
            self._report_type_error(error)

    def visit_declaration_statement(self, node: DeclarationStatement) -> None:
        if node.expression is None:
            return
        try:
            lhs = _InferredType(
                type=node.variable_type.base_type, free_indices=frozenset()
            )
            rhs = self._infer_type(node.expression)
            self._check_compatible(
                lhs,
                rhs,
                context=f"declaration of {node.variable_name.name_hint!r}",
                provenance=node.provenance,
            )
        except _TypeCheckError as error:
            self._report_type_error(error)

    def visit_return_statement(self, node: ReturnStatement) -> None:
        if self._current_return_type is None:
            return
        try:
            lhs = _InferredType(
                type=self._current_return_type, free_indices=frozenset()
            )
            rhs = self._infer_type(node.expression)
            self._check_compatible(
                lhs,
                rhs,
                context="return statement",
                provenance=node.provenance,
            )
        except _TypeCheckError as error:
            self._report_type_error(error)

    def _report_type_error(self, error: _TypeCheckError) -> None:
        """Convert a raised :class:`_TypeCheckError` into an ERROR diagnostic."""
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message("type error", error.message, error.provenance),
        )

    def _check_call_argument_types(self, expression: FunctionExpression) -> None:
        """Verify argument types at a call site against the callee's signature.

        Reductions and builtin-import calls are skipped (no declared
        signature). When the argument count does not match the callee's
        signature, the check is also skipped -- the
        :class:`CallSiteValidator` reports that specific mismatch.
        Otherwise, each actual argument's inferred type is compared against
        the declared parameter type via :func:`_is_callable_with`, which
        accepts data-type promotion and matches literal shape dims exactly
        while leaving symbolic dims unconstrained.

        Raises:
            _TypeCheckError: At the first argument whose type cannot be
                passed for the declared parameter type.

        """
        if not isinstance(expression.function, IdentifierExpression):
            return
        identifier = expression.function.identifier
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        except SymbolTableError:
            return
        if not isinstance(frame, FunctionSymbolTableFrame):
            return
        self._check_call_argument_types_against_signature(
            identifier, expression, frame.signature
        )

    def _check_call_argument_types_against_signature(
        self,
        identifier: Identifier,
        expression: FunctionExpression,
        signature: Sequence[tuple[object, Type]],
    ) -> frozenset[Identifier]:
        """Check actual arguments against a known callee signature.

        Also aggregates the free-index set across every argument (the caller
        of :meth:`_infer_function_expression` needs it). Returns an empty
        frozenset when the count mismatches so callers don't over-report --
        the missing free indices are harmless because a diagnostic was
        already emitted at the call site.

        """
        if len(expression.args) != len(signature):
            # Count mismatch is reported by `CallSiteValidator`; skipping
            # per-argument type checks avoids cascaded diagnostics.
            return frozenset()
        argument_free_indices: frozenset[Identifier] = frozenset()
        for position, (actual_expression, (_qualifier, param_type)) in enumerate(
            zip(expression.args, signature)
        ):
            actual = self._infer_type(actual_expression)
            argument_free_indices |= actual.free_indices
            if not _is_callable_with(param_type, actual.type):
                raise _TypeCheckError(
                    f"Argument {position} to {identifier.name_hint!r}: "
                    f"cannot pass {actual.type} where {param_type} is "
                    "expected.",
                    actual_expression.provenance,
                )
        return argument_free_indices

    def _check_compatible(
        self,
        expected: _InferredType,
        actual: _InferredType,
        *,
        context: str,
        provenance: Provenance | None,
    ) -> None:
        self._check_compatible_types(expected, actual, context, provenance)
        self._check_compatible_free_indices(expected, actual, context, provenance)

    def _check_compatible_types(
        self,
        expected: _InferredType,
        actual: _InferredType,
        context: str,
        provenance: Provenance | None,
    ) -> None:
        if _is_assignable(expected.type, actual.type):
            return
        element_types_assignable = _are_element_types_assignable(
            expected.type, actual.type
        )
        scalar_to_shaped_broadcast_ok = (
            element_types_assignable
            and isinstance(expected.type, NumericalType)
            and isinstance(actual.type, NumericalType)
            and not actual.type.shape
            and expected.type.shape
        )
        if scalar_to_shaped_broadcast_ok:
            return
        raise _TypeCheckError(
            f"Type mismatch in {context}: expected {expected.type}, got {actual.type}.",
            provenance,
        )

    def _check_compatible_free_indices(
        self,
        expected: _InferredType,
        actual: _InferredType,
        context: str,
        provenance: Provenance | None,
    ) -> None:
        # The statement runs inside an implicit loop over the indices used in
        # its expressions. Free indices on the right that are not already
        # bound on the left are absorbed by the left-hand side's shape
        # dimensions.
        extra_free = actual.free_indices - expected.free_indices
        missing_free = expected.free_indices - actual.free_indices
        lhs_shape = (
            expected.type.shape if isinstance(expected.type, NumericalType) else ()
        )
        rhs_shape = actual.type.shape if isinstance(actual.type, NumericalType) else ()
        absorbed = len(lhs_shape) - len(rhs_shape)
        if not missing_free and len(extra_free) == max(absorbed, 0):
            return
        expected_names = sorted(
            identifier.name_hint for identifier in expected.free_indices
        )
        actual_names = sorted(
            identifier.name_hint for identifier in actual.free_indices
        )
        raise _TypeCheckError(
            f"Free-index mismatch in {context}: expected indices "
            f"{{{', '.join(expected_names)}}}, got "
            f"{{{', '.join(actual_names)}}}.",
            provenance,
        )

    def _infer_type(self, expression: Expression) -> _InferredType:
        if isinstance(expression, IntLiteral):
            return self._infer_int_literal(expression)
        elif isinstance(expression, FloatLiteral):
            return _InferredType(
                type=NumericalType(PrimitiveDataType(CoreDataType.FLOAT)),
                free_indices=frozenset(),
            )
        elif isinstance(expression, ComplexLiteral):
            raise _TypeCheckError(
                "Complex literals are not yet supported by the type checker.",
                expression.provenance,
            )
        elif isinstance(expression, IdentifierExpression):
            return self._infer_identifier(expression.identifier, expression.provenance)
        elif isinstance(expression, UnaryExpression):
            return self._infer_unary(expression)
        elif isinstance(expression, BinaryExpression):
            return self._infer_binary(expression)
        elif isinstance(expression, TernaryExpression):
            return self._infer_ternary(expression)
        elif isinstance(expression, ArrayAccessExpression):
            return self._infer_array_access(expression)
        elif isinstance(expression, FunctionExpression):
            return self._infer_function_expression(expression)
        else:
            raise _TypeCheckError(
                "Unsupported expression in type inference: "
                f"{type(expression).__name__}.",
                expression.provenance,
            )

    def _infer_int_literal(self, expression: IntLiteral) -> _InferredType:
        # Dispatch to fhy_core's literal resolver with a "weak" target
        # family: non-negative values that fit uint32 pick from the uint
        # chain, otherwise fall through to the signed chain so large
        # positives still land on int64. Negatives always go to the signed
        # chain.
        value = expression.value
        weak_target = _pick_weak_literal_target(value)
        try:
            core_type = resolve_literal_core_data_type(value, weak_target)
        except FhYCoreTypeError as exc:
            raise _TypeCheckError(
                f"Integer literal {value} does not fit in any supported integer type.",
                expression.provenance,
            ) from exc
        return _InferredType(
            type=NumericalType(PrimitiveDataType(core_type)),
            free_indices=frozenset(),
        )

    def _infer_identifier(
        self, identifier: Identifier, provenance: Provenance | None
    ) -> _InferredType:
        frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        if not isinstance(frame, VariableSymbolTableFrame):
            raise _TypeCheckError(
                f"Identifier {identifier.name_hint!r} does not refer to a variable.",
                provenance,
            )
        return _InferredType(type=frame.type, free_indices=frozenset())

    def _infer_unary(self, expression: UnaryExpression) -> _InferredType:
        operand = self._infer_type(expression.expression)
        if not isinstance(operand.type, NumericalType):
            raise _TypeCheckError(
                f"Unary {expression.operation.value!r} requires a numerical "
                f"operand; got {operand.type}.",
                expression.provenance,
            )
        if not isinstance(operand.type.data_type, PrimitiveDataType):
            raise _TypeCheckError(
                f"Unary {expression.operation.value!r} requires a primitive "
                f"operand data type; got {operand.type.data_type}.",
                expression.provenance,
            )
        self._check_unary_operation_constraints(expression, operand.type)
        return operand

    def _check_unary_operation_constraints(
        self, expression: UnaryExpression, operand_type: NumericalType
    ) -> None:
        if expression.operation is UnaryOperation.BITWISE_NOT:
            if not _is_integer_numerical_type(operand_type):
                raise _TypeCheckError(
                    "Unary '~' requires an integer operand; got "
                    f"{operand_type.data_type}.",
                    expression.provenance,
                )
        elif expression.operation is UnaryOperation.LOGICAL_NOT:
            if operand_type.shape:
                raise _TypeCheckError(
                    "Unary '!' requires a scalar operand; got a shaped "
                    f"operand {operand_type}.",
                    expression.provenance,
                )
        # UnaryOperation.NEGATION is permitted on any numerical primitive
        # type (int / float / complex).

    def _infer_binary(self, expression: BinaryExpression) -> _InferredType:
        left = self._infer_type(expression.left)
        right = self._infer_type(expression.right)
        if not isinstance(left.type, NumericalType) or not isinstance(
            right.type, NumericalType
        ):
            raise _TypeCheckError(
                "Binary expression operands must be numerical; got "
                f"{left.type} and {right.type}.",
                expression.provenance,
            )
        if not _are_shapes_equivalent(left.type.shape, right.type.shape):
            raise _TypeCheckError(
                "Binary expression operand shapes are not structurally "
                f"equivalent: {left.type.shape} vs {right.type.shape}.",
                expression.provenance,
            )
        if not isinstance(left.type.data_type, PrimitiveDataType) or not isinstance(
            right.type.data_type, PrimitiveDataType
        ):
            raise _TypeCheckError(
                "Binary expression operands must have primitive data types; "
                f"got {left.type.data_type} and {right.type.data_type}.",
                expression.provenance,
            )
        try:
            promoted = promote_primitive_data_types(
                left.type.data_type, right.type.data_type
            )
        except FhYCoreTypeError as exc:
            raise _TypeCheckError(
                f"Cannot promote binary operand data types "
                f"{left.type.data_type} and {right.type.data_type}: {exc}",
                expression.provenance,
            ) from exc
        return _InferredType(
            type=NumericalType(promoted, shape=left.type.shape),
            free_indices=left.free_indices | right.free_indices,
        )

    def _infer_ternary(self, expression: TernaryExpression) -> _InferredType:
        condition = self._infer_type(expression.condition)
        self._check_ternary_condition(condition, expression)
        true_branch = self._infer_type(expression.true)
        false_branch = self._infer_type(expression.false)
        if not isinstance(true_branch.type, NumericalType) or not isinstance(
            false_branch.type, NumericalType
        ):
            raise _TypeCheckError(
                "Ternary expression branches must be numerical; got "
                f"{true_branch.type} and {false_branch.type}.",
                expression.provenance,
            )
        if not _are_shapes_equivalent(true_branch.type.shape, false_branch.type.shape):
            raise _TypeCheckError(
                "Ternary expression branch shapes are not structurally "
                f"equivalent: {true_branch.type.shape} vs "
                f"{false_branch.type.shape}.",
                expression.provenance,
            )
        if not isinstance(
            true_branch.type.data_type, PrimitiveDataType
        ) or not isinstance(false_branch.type.data_type, PrimitiveDataType):
            raise _TypeCheckError(
                "Ternary expression branches must have primitive data types; "
                f"got {true_branch.type.data_type} and "
                f"{false_branch.type.data_type}.",
                expression.provenance,
            )
        try:
            promoted = promote_primitive_data_types(
                true_branch.type.data_type, false_branch.type.data_type
            )
        except FhYCoreTypeError as exc:
            raise _TypeCheckError(
                f"Cannot promote ternary branch data types "
                f"{true_branch.type.data_type} and "
                f"{false_branch.type.data_type}: {exc}",
                expression.provenance,
            ) from exc
        return _InferredType(
            type=NumericalType(promoted, shape=true_branch.type.shape),
            free_indices=(
                condition.free_indices
                | true_branch.free_indices
                | false_branch.free_indices
            ),
        )

    def _check_ternary_condition(
        self, condition: _InferredType, expression: TernaryExpression
    ) -> None:
        if not isinstance(condition.type, NumericalType):
            raise _TypeCheckError(
                f"Ternary condition must be a numerical scalar; got {condition.type}.",
                expression.provenance,
            )
        if condition.type.shape:
            raise _TypeCheckError(
                "Ternary condition must be a scalar; got shaped type "
                f"{condition.type}.",
                expression.provenance,
            )
        if not _is_integer_numerical_type(condition.type):
            raise _TypeCheckError(
                "Ternary condition must be an integer-typed scalar; got "
                f"{condition.type.data_type}.",
                expression.provenance,
            )

    def _infer_array_access(self, expression: ArrayAccessExpression) -> _InferredType:
        base = self._infer_type(expression.array_expression)
        if not isinstance(base.type, NumericalType):
            raise _TypeCheckError(
                f"Array access requires a numerical type; got {base.type}.",
                expression.provenance,
            )
        shape = base.type.shape
        if len(expression.indices) != len(shape):
            raise _TypeCheckError(
                f"Array access has {len(expression.indices)} indices but the "
                f"array has {len(shape)} dimensions.",
                expression.provenance,
            )
        free_indices = set(base.free_indices)
        for index_expression in expression.indices:
            free_indices.update(
                collect_indices(index_expression, self._is_identifier_index)
            )
        return _InferredType(
            type=NumericalType(base.type.data_type),
            free_indices=frozenset(free_indices - self._bound_forall_indices),
        )

    def _infer_function_expression(
        self, expression: FunctionExpression
    ) -> _InferredType:
        if not isinstance(expression.function, IdentifierExpression):
            raise _TypeCheckError(
                "Function expression must be called on an identifier; got "
                f"{type(expression.function).__name__}.",
                expression.function.provenance,
            )
        identifier = expression.function.identifier
        frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        if _is_reduction_frame(frame):
            return self._infer_reduction_expression(expression, identifier)
        elif isinstance(frame, FunctionSymbolTableFrame):
            return self._infer_user_function_call(expression, identifier, frame)
        elif isinstance(frame, ImportSymbolTableFrame):
            return self._infer_builtin_call(expression, identifier)
        else:
            raise _TypeCheckError(
                f"Function {identifier.name_hint!r} is not callable in a type context.",
                expression.provenance,
            )

    def _infer_reduction_expression(
        self, expression: FunctionExpression, identifier: Identifier
    ) -> _InferredType:
        if len(expression.args) != 1:
            raise _TypeCheckError(
                f"Reduction {identifier.name_hint!r} must be passed exactly "
                f"one argument; got {len(expression.args)}.",
                expression.provenance,
            )
        argument = self._infer_type(expression.args[0])
        reduced = collect_reduced_indices(expression, self._is_identifier_index)
        return _InferredType(
            type=argument.type, free_indices=argument.free_indices - reduced
        )

    def _infer_user_function_call(
        self,
        expression: FunctionExpression,
        identifier: Identifier,
        frame: FunctionSymbolTableFrame,
    ) -> _InferredType:
        if frame.keyword == FunctionKeyword.PROCEDURE:
            raise _TypeCheckError(
                f"Procedure {identifier.name_hint!r} cannot be used as an "
                "expression because it does not return a value.",
                expression.provenance,
            )
        return_type = self._operation_return_types.get(frame.name)
        if return_type is None:
            raise _TypeCheckError(
                f"Cannot determine return type of operation {identifier.name_hint!r}.",
                expression.provenance,
            )
        argument_free_indices = self._check_call_argument_types_against_signature(
            identifier, expression, frame.signature
        )
        return _InferredType(type=return_type, free_indices=argument_free_indices)

    def _infer_builtin_call(
        self, expression: FunctionExpression, identifier: Identifier
    ) -> _InferredType:
        if not expression.args:
            raise _TypeCheckError(
                f"Builtin {identifier.name_hint!r} requires at least one argument.",
                expression.provenance,
            )
        return self._infer_type(expression.args[0])

    def _is_identifier_index(self, identifier: Identifier) -> bool:
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        except SymbolTableError:
            return True
        return isinstance(frame, VariableSymbolTableFrame) and isinstance(
            frame.type, IndexType
        )
