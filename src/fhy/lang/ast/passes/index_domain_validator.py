"""Validate that array-access indices fall within the array's dimension domains."""

__all__ = [
    "IndexDomainValidator",
]

from fhy_core import (
    BinaryExpression as CoreBinaryExpression,
)
from fhy_core import (
    BinaryOperation as CoreBinaryOperation,
)
from fhy_core import (
    CoreDataType,
    DiagnosticLevel,
    FhYCoreTypeError,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    SymbolTableError,
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

from fhy.lang.ast.node import (
    ArrayAccessExpression,
    Expression,
    IdentifierExpression,
)

from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .ast_to_core_expression_converter import (
    convert_ast_expression_to_core_expression,
)
from .utils import format_diagnostic_message

_UNSIGNED_INTEGER_CORE_DATA_TYPES = frozenset(
    {
        CoreDataType.UINT,
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
    }
)


@register_pass(
    "fhy_ast_index_domain_validator",
    "Validates array accesses and that the indices fall within the array's dimensions.",
)
class IndexDomainValidator(AnalysisPassWithSymbolTable):
    """Validate that every array-access index falls within the array's domain.

    For each :class:`ArrayAccessExpression`, the pass:

    - Checks that the accessed base is an identifier expression.
    - Checks that the identifier resolves to a numerical (array) variable.
    - Checks that the rank of the access matches the rank of the array.
    - Synthesizes the type of each index expression and derives its
      ``[lower_bound, upper_bound]`` either from an ``IndexType`` or from a
      scalar unsigned-integer ``PARAM`` expression.
    - Uses z3 satisfiability to verify
      ``lower_bound >= 1 AND upper_bound <= dim_size``.

    All failures are reported as diagnostics; the visitor continues so the
    user gets every diagnostic in one run.

    """

    def visit_array_access_expression(self, node: ArrayAccessExpression) -> None:
        if not isinstance(node.array_expression, IdentifierExpression):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    "Non-identifier array access expressions are not yet "
                    f"supported; got {type(node.array_expression).__name__}.",
                    node.provenance,
                ),
            )
            return
        array_name = node.array_expression.identifier
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, array_name)
        except SymbolTableError:
            return
        if not isinstance(frame, VariableSymbolTableFrame) or not isinstance(
            frame.type, NumericalType
        ):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "type error",
                    f"Array access on {array_name.name_hint!r} is not a "
                    f"vector; got {frame}.",
                    node.provenance,
                ),
            )
            return
        shape = frame.type.shape
        if len(node.indices) != len(shape):
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "structural error",
                    f"Array access on {array_name.name_hint!r} has "
                    f"{len(node.indices)} indices but {len(shape)} "
                    f"dimensions; got shape {shape}.",
                    node.provenance,
                ),
            )
            return

        for ast_index, dim_size in zip(node.indices, shape):
            try:
                core_index = convert_ast_expression_to_core_expression(ast_index)
            except NotImplementedError:
                self.report(
                    DiagnosticLevel.ERROR,
                    format_diagnostic_message(
                        "type error",
                        f"Array-access index {ast_index} is not a supported "
                        f"type; got type {type(ast_index).__name__}.",
                        ast_index.provenance,
                    ),
                )
                continue
            try:
                index_type, index_qualifier = synthesize_expression_type(
                    core_index, self._get_identifier_type
                )
            except FhYCoreTypeError as exc:
                self.report(
                    DiagnosticLevel.ERROR,
                    format_diagnostic_message(
                        "type error",
                        f"Failed to synthesize a type for array-access "
                        f"index {ast_index}: {exc}",
                        ast_index.provenance,
                    ),
                )
                continue
            bounds = self._get_index_bounds(
                ast_index, core_index, index_type, index_qualifier
            )
            if bounds is None:
                continue
            lower_bound, upper_bound = bounds
            self._check_index_in_domain(
                array_name, ast_index, lower_bound, upper_bound, dim_size
            )

    def _get_index_bounds(
        self,
        ast_index: Expression,
        core_index: CoreExpression,
        index_type: Type,
        index_qualifier: TypeQualifier,
    ) -> tuple[CoreExpression, CoreExpression] | None:
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
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "type error",
                f"Array-access index {ast_index} must resolve to either an "
                "index type or a scalar unsigned-integer PARAM expression; "
                f"got type {index_type} with qualifier "
                f"{index_qualifier.value!r}.",
                ast_index.provenance,
            ),
        )
        return None

    def _get_identifier_type(
        self, identifier: Identifier
    ) -> tuple[Type, TypeQualifier]:
        frame = self.get_frame_from_namespace(self.current_namespace, identifier)
        if not isinstance(frame, VariableSymbolTableFrame):
            raise FhYCoreTypeError(
                f"Identifier {identifier.name_hint!r} does not refer to a variable."
            )
        return frame.type, frame.type_qualifier

    def _check_index_in_domain(
        self,
        array_name: Identifier,
        ast_index: Expression,
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
            self.report(
                DiagnosticLevel.ERROR,
                format_diagnostic_message(
                    "semantic error",
                    f"Array access on {array_name.name_hint!r} is out of "
                    f"bounds: index range [{lower_bound}, {upper_bound}] is "
                    f"not contained in [1, {dim_size}].",
                    ast_index.provenance,
                ),
            )
