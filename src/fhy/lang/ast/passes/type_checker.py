"""Type-check expression statements, declarations, and returns in the AST.

Checks:
    - The right-hand side of expression statements are compatible (with promotion)
        with the left-hand side.
    - The right-hand side of declarations are compatible (with promotion) with the
        declared symbol.
    - The right-hand side of returns are compatible (with promotion) with the return
        type of the current operation.
    - Arguments passed to functions are compatible (with promotion) with the arguments
        of the function.

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
)
from fhy_core import (
    Expression as CoreExpression,
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
    boundary, emits a diagnostic via ``self.report(...)``, and moves on —
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


def _shapes_equivalent(
    shape_a: Sequence[CoreExpression], shape_b: Sequence[CoreExpression]
) -> bool:
    if len(shape_a) != len(shape_b):
        return False
    return all(a.is_structurally_equivalent(b) for a, b in zip(shape_a, shape_b))


def _is_assignable(target: Type, source: Type) -> bool:
    if target.is_structurally_equivalent(source):
        return True
    elif not isinstance(target, NumericalType) or not isinstance(source, NumericalType):
        return False
    elif not _shapes_equivalent(target.shape, source.shape):
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


def _is_element_types_assignable(target: Type, source: Type) -> bool:
    if not isinstance(target, NumericalType) or not isinstance(source, NumericalType):
        return False
    else:
        return _is_assignable(
            NumericalType(target.data_type), NumericalType(source.data_type)
        )


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


def _is_integer_numerical(ty: Type) -> bool:
    """Return True when `ty` is a numerical type with an integer core type."""
    return (
        isinstance(ty, NumericalType)
        and isinstance(ty.data_type, PrimitiveDataType)
        and ty.data_type.core_data_type in _INTEGER_CORE_DATA_TYPES
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

    def _report_type_error(self, error: _TypeCheckError) -> None:
        """Convert a raised :class:`_TypeCheckError` into an ERROR diagnostic."""
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message("type error", error.message, error.provenance),
        )

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if node.left is None:
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

    def _check_compatible(
        self,
        expected: _InferredType,
        actual: _InferredType,
        *,
        context: str,
        provenance: Provenance | None,
    ) -> None:
        element_types_assignable = _is_element_types_assignable(
            expected.type, actual.type
        )
        if not _is_assignable(expected.type, actual.type) and not (
            element_types_assignable
            and isinstance(expected.type, NumericalType)
            and isinstance(actual.type, NumericalType)
            and not actual.type.shape
            and expected.type.shape
        ):
            raise _TypeCheckError(
                f"Type mismatch in {context}: expected {expected.type}, got "
                f"{actual.type}.",
                provenance,
            )

        # The statement runs inside an implicit loop over the indices used in
        # its expressions. Free indices on the right that are not already bound
        # on the left are absorbed by the left-hand side's shape dimensions.
        extra_free = actual.free_indices - expected.free_indices
        missing_free = expected.free_indices - actual.free_indices
        lhs_shape = (
            expected.type.shape if isinstance(expected.type, NumericalType) else ()
        )
        rhs_shape = actual.type.shape if isinstance(actual.type, NumericalType) else ()
        absorbed = len(lhs_shape) - len(rhs_shape)
        if missing_free or len(extra_free) != max(absorbed, 0):
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
            primitive = PrimitiveDataType(
                self._get_int_literal_core_type(expression.value)
            )
            return _InferredType(
                type=NumericalType(primitive), free_indices=frozenset()
            )
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
                f"Unsupported expression in type inference: "
                f"{type(expression).__name__}.",
                expression.provenance,
            )

    @staticmethod
    def _get_int_literal_core_type(value: int) -> CoreDataType:
        return CoreDataType.UINT if value >= 0 else CoreDataType.INT

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

        if expression.operation is UnaryOperation.BITWISE_NOT:
            if not _is_integer_numerical(operand.type):
                raise _TypeCheckError(
                    "Unary '~' requires an integer operand; got "
                    f"{operand.type.data_type}.",
                    expression.provenance,
                )
        elif expression.operation is UnaryOperation.LOGICAL_NOT:
            if operand.type.shape:
                raise _TypeCheckError(
                    "Unary '!' requires a scalar operand; got a shaped "
                    f"operand {operand.type}.",
                    expression.provenance,
                )
        # UnaryOperation.NEGATION is permitted on any numerical primitive
        # type (int / float / complex).

        return operand

    def _infer_binary(self, expression: BinaryExpression) -> _InferredType:
        left = self._infer_type(expression.left)
        right = self._infer_type(expression.right)
        if not isinstance(left.type, NumericalType) or not isinstance(
            right.type, NumericalType
        ):
            raise _TypeCheckError(
                f"Binary expression operands must be numerical; got "
                f"{left.type} and {right.type}.",
                expression.provenance,
            )
        if not _shapes_equivalent(left.type.shape, right.type.shape):
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
        if not isinstance(condition.type, NumericalType):
            raise _TypeCheckError(
                "Ternary condition must be a numerical scalar; got "
                f"{condition.type}.",
                expression.provenance,
            )
        if condition.type.shape:
            raise _TypeCheckError(
                "Ternary condition must be a scalar; got shaped type "
                f"{condition.type}.",
                expression.provenance,
            )
        if not _is_integer_numerical(condition.type):
            raise _TypeCheckError(
                "Ternary condition must be an integer-typed scalar; got "
                f"{condition.type.data_type}.",
                expression.provenance,
            )
        true_branch = self._infer_type(expression.true)
        false_branch = self._infer_type(expression.false)
        if not isinstance(true_branch.type, NumericalType) or not isinstance(
            false_branch.type, NumericalType
        ):
            raise _TypeCheckError(
                f"Ternary expression branches must be numerical; got "
                f"{true_branch.type} and {false_branch.type}.",
                expression.provenance,
            )
        if not _shapes_equivalent(true_branch.type.shape, false_branch.type.shape):
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
        new_free = set(base.free_indices)
        for index_expr in expression.indices:
            new_free.update(collect_indices(index_expr, self._is_identifier_index))
        return _InferredType(
            type=NumericalType(base.type.data_type),
            free_indices=frozenset(new_free - self._bound_forall_indices),
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
        is_reduction = (
            isinstance(frame, ImportSymbolTableFrame)
            and frame.name in BUILTIN_REDUCTION_FUNCTION_IDENTIFIERS.values()
        )
        if is_reduction:
            if len(expression.args) != 1:
                raise _TypeCheckError(
                    f"Reduction {identifier.name_hint!r} must be passed "
                    f"exactly one argument; got {len(expression.args)}.",
                    expression.provenance,
                )
            arg = self._infer_type(expression.args[0])
            reduced = collect_reduced_indices(expression, self._is_identifier_index)
            return _InferredType(type=arg.type, free_indices=arg.free_indices - reduced)
        elif isinstance(frame, FunctionSymbolTableFrame):
            if frame.keyword == FunctionKeyword.PROCEDURE:
                raise _TypeCheckError(
                    f"Procedure {identifier.name_hint!r} cannot be used as an "
                    "expression because it does not return a value.",
                    expression.provenance,
                )
            return_type = self._operation_return_types.get(frame.name)
            if return_type is None:
                raise _TypeCheckError(
                    f"Cannot determine return type of operation "
                    f"{identifier.name_hint!r}.",
                    expression.provenance,
                )
            arg_free_indices: frozenset[Identifier] = frozenset()
            for call_arg in expression.args:
                arg_free_indices |= self._infer_type(call_arg).free_indices
            return _InferredType(type=return_type, free_indices=arg_free_indices)
        elif isinstance(frame, ImportSymbolTableFrame):
            if not expression.args:
                raise _TypeCheckError(
                    f"Builtin {identifier.name_hint!r} requires at least one "
                    "argument.",
                    expression.provenance,
                )
            return self._infer_type(expression.args[0])
        else:
            raise _TypeCheckError(
                f"Function {identifier.name_hint!r} is not callable in a type context.",
                expression.provenance,
            )

    def _is_identifier_index(self, identifier: Identifier) -> bool:
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        except SymbolTableError:
            return True
        return isinstance(frame, VariableSymbolTableFrame) and isinstance(
            frame.type, IndexType
        )
