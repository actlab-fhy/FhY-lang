"""Validate that array-access indices fall within the array's dimension domains."""

__all__ = [
    "validate_index_domains",
]

from fhy_core import (
    BinaryExpression as CoreBinaryExpression,
)
from fhy_core import (
    BinaryOperation as CoreBinaryOperation,
)
from fhy_core import (
    CoreDataType,
    FhYCoreTypeError,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    SymbolTable,
    SymbolType,
    Type,
    TypeQualifier,
    VariableSymbolTableFrame,
    collect_identifiers,
    is_satisfiable,
    register_pass,
    synthesize_expression_type,
)
from fhy_core import (
    Expression as CoreExpression,
)
from fhy_core import (
    LiteralExpression as CoreLiteralExpression,
)

from fhy.lang.ast.error import FhYSemanticsError, FhYStructuralError, FhYTypeError
from fhy.lang.ast.node import (
    ArrayAccessExpression,
    IdentifierExpression,
    Module,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .ast_to_core_expression_converter import (
    convert_ast_expression_to_core_expression,
)

_UNSIGNED_INTEGER_CORE_DATA_TYPES = frozenset(
    {
        CoreDataType.UINT,
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
        CoreDataType.UINT64,
    }
)


@register_pass(
    "fhy_ast_index_domain_validator",
    "Validates array accesses and that the indices fall within the array's dimensions.",
)
class _IndexDomainValidator(AnalysisPassWithSymbolTable):
    def visit_array_access_expression(self, node: ArrayAccessExpression) -> None:
        if not isinstance(node.array_expression, IdentifierExpression):
            raise NotImplementedError(
                "Non-identifier array access expressions are not yet supported."
            )
        array_name = node.array_expression.identifier
        frame = self.get_frame_from_namespace(self.current_namespace, array_name)
        if not isinstance(frame, VariableSymbolTableFrame) or not isinstance(
            frame.type, NumericalType
        ):
            raise RuntimeError(
                f"Array access on {array_name.name_hint!r} is not a vector; got "
                f"type {frame.type}."
            )
        shape = frame.type.shape
        if len(node.indices) != len(shape):
            raise FhYStructuralError(
                f"Array access on {array_name.name_hint!r} has {len(node.indices)} "
                f"indices but {len(shape)} dimensions; got shape {shape}."
            )

        for ast_index, dim_size in zip(node.indices, shape):
            try:
                core_index = convert_ast_expression_to_core_expression(ast_index)
            except NotImplementedError as exc:
                raise FhYTypeError(
                    f"Array-access index {ast_index} is not a supported type; got "
                    f"type {type(ast_index).__name__}."
                ) from exc
            try:
                index_type, index_qualifier = synthesize_expression_type(
                    core_index, self._get_identifier_type
                )
            except FhYCoreTypeError as exc:
                raise FhYTypeError(
                    f"Failed to synthesize a type for array-access index "
                    f"{ast_index}: {exc}"
                ) from exc
            lower_bound, upper_bound = self._get_index_bounds(
                ast_index, core_index, index_type, index_qualifier
            )
            self._check_index_in_domain(array_name, lower_bound, upper_bound, dim_size)

    def _get_index_bounds(
        self,
        ast_index: object,
        core_index: CoreExpression,
        index_type: Type,
        index_qualifier: TypeQualifier,
    ) -> tuple[CoreExpression, CoreExpression]:
        if isinstance(index_type, IndexType):
            return index_type.lower_bound, index_type.upper_bound
        if (
            isinstance(index_type, NumericalType)
            and index_type.is_scalar()
            and isinstance(index_type.data_type, PrimitiveDataType)
            and index_type.data_type.core_data_type in _UNSIGNED_INTEGER_CORE_DATA_TYPES
            and index_qualifier == TypeQualifier.PARAM
        ):
            return core_index, core_index
        raise FhYTypeError(
            f"Array-access index {ast_index} must resolve to either an index "
            "type or a scalar unsigned-integer PARAM expression; got type "
            f"{index_type} with qualifier {index_qualifier.value!r}."
        )

    def _get_identifier_type(
        self, identifier: Identifier
    ) -> tuple[Type, TypeQualifier]:
        frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        if not isinstance(frame, VariableSymbolTableFrame):
            raise FhYCoreTypeError(
                f"Identifier {identifier.name_hint!r} does not refer to a " "variable."
            )
        return frame.type, frame.type_qualifier

    def _check_index_in_domain(
        self,
        array_name: Identifier,
        lower_bound: CoreExpression,
        upper_bound: CoreExpression,
        dim_size: CoreExpression,
    ) -> None:
        lower_bound_constraint = lower_bound >= CoreLiteralExpression(1)
        upper_bound_constraint = upper_bound <= dim_size
        constraint = CoreBinaryExpression(
            CoreBinaryOperation.LOGICAL_AND,
            lower_bound_constraint,
            upper_bound_constraint,
        )
        violation = constraint.logical_not()
        identifiers = collect_identifiers(violation)
        symbol_types = {identifier: SymbolType.INT for identifier in identifiers}
        if is_satisfiable(identifiers, violation, symbol_types):
            raise FhYSemanticsError(
                f"Array access on {array_name.name_hint!r} is out of bounds: "
                f"index range [{lower_bound}, {upper_bound}] is not contained "
                f"in [1, {dim_size}]."
            )


def validate_index_domains(module: Module, symbol_table: SymbolTable) -> None:
    """Validate array accesses and that the indices fall within the array's dimensions.

    Args:
        module: The module to validate.
        symbol_table: The symbol table to use.

    Raises:
        FhYStructuralError: If an array-access index is a tuple access
            expression (not yet supported).
        FhYTypeError: If an array-access index does not resolve to either an
            index type or a scalar unsigned-integer PARAM expression.
        FhYSemanticsError: If an array-access index is out of bounds for the
            accessed dimension.

    """
    validator = _IndexDomainValidator(symbol_table)
    validator(module)
