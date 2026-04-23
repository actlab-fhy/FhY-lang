"""Validate that array-access indices fall within the array's dimension domains."""

__all__ = [
    "IndexDomainValidator",
]

from collections.abc import Sequence
from dataclasses import dataclass

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

_UNSIGNED_INTEGER_CORE_DATA_TYPES: frozenset[CoreDataType] = frozenset(
    {
        CoreDataType.UINT,
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
    }
)


@dataclass(frozen=True)
class _DomainCheckInput:
    """Inputs needed to check that an index falls within an array dimension."""

    array_name: Identifier
    ast_index: Expression
    lower_bound: CoreExpression
    upper_bound: CoreExpression
    dimension_size: CoreExpression


def _is_unsigned_integer_param(
    index_type: Type, index_qualifier: TypeQualifier
) -> bool:
    return (
        isinstance(index_type, NumericalType)
        and index_type.is_scalar()
        and isinstance(index_type.data_type, PrimitiveDataType)
        and index_type.data_type.core_data_type in _UNSIGNED_INTEGER_CORE_DATA_TYPES
        and index_qualifier == TypeQualifier.PARAM
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
      ``[lower_bound, upper_bound]`` either from an :class:`IndexType` or
      from a scalar unsigned-integer ``PARAM`` expression.
    - Uses z3 satisfiability to verify
      ``lower_bound >= 1 AND upper_bound <= dim_size``.

    All failures are reported as diagnostics; the visitor continues so the
    user gets every diagnostic in one run.

    """

    def visit_array_access_expression(self, node: ArrayAccessExpression) -> None:
        if not isinstance(node.array_expression, IdentifierExpression):
            self._report_non_identifier_array_expression(node)
            return
        array_name = node.array_expression.identifier
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, array_name)
        except SymbolTableError:
            return
        if not isinstance(frame, VariableSymbolTableFrame) or not isinstance(
            frame.type, NumericalType
        ):
            self._report_non_vector_access(node, array_name, frame)
            return
        shape = frame.type.shape
        if len(node.indices) != len(shape):
            self._report_rank_mismatch(node, array_name, shape)
            return

        for ast_index, dimension_size in zip(node.indices, shape):
            self._check_single_index(array_name, ast_index, dimension_size)

    def _check_single_index(
        self,
        array_name: Identifier,
        ast_index: Expression,
        dimension_size: CoreExpression,
    ) -> None:
        try:
            core_index = convert_ast_expression_to_core_expression(ast_index)
        except NotImplementedError:
            self._report_unsupported_index_type(ast_index)
            return
        try:
            index_type, index_qualifier = synthesize_expression_type(
                core_index, self._get_identifier_type
            )
        except FhYCoreTypeError as exc:
            self._report_failed_type_synthesis(ast_index, exc)
            return
        bounds = self._get_index_bounds(
            ast_index, core_index, index_type, index_qualifier
        )
        if bounds is None:
            return
        lower_bound, upper_bound = bounds
        self._check_index_in_domain(
            _DomainCheckInput(
                array_name=array_name,
                ast_index=ast_index,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                dimension_size=dimension_size,
            )
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
        if _is_unsigned_integer_param(index_type, index_qualifier):
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

    def _check_index_in_domain(self, check_input: _DomainCheckInput) -> None:
        lower_bound_constraint = check_input.lower_bound >= CoreLiteralExpression(1)
        upper_bound_constraint = check_input.upper_bound <= check_input.dimension_size
        constraint = CoreBinaryExpression(
            CoreBinaryOperation.LOGICAL_AND,
            lower_bound_constraint,
            upper_bound_constraint,
        )
        violation = constraint.logical_not()
        identifiers = collect_identifiers(violation)
        symbol_types = dict.fromkeys(identifiers, SymbolType.INT)
        if not is_satisfiable(identifiers, violation, symbol_types):
            return
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "semantic error",
                f"Array access on {check_input.array_name.name_hint!r} is out "
                f"of bounds: index range [{check_input.lower_bound}, "
                f"{check_input.upper_bound}] is not contained in "
                f"[1, {check_input.dimension_size}].",
                check_input.ast_index.provenance,
            ),
        )

    def _report_non_identifier_array_expression(
        self, node: ArrayAccessExpression
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                "Non-identifier array access expressions are not yet "
                f"supported; got {type(node.array_expression).__name__}.",
                node.provenance,
            ),
        )

    def _report_non_vector_access(
        self,
        node: ArrayAccessExpression,
        array_name: Identifier,
        frame: object,
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "type error",
                f"Array access on {array_name.name_hint!r} is not a vector; "
                f"got {frame}.",
                node.provenance,
            ),
        )

    def _report_rank_mismatch(
        self,
        node: ArrayAccessExpression,
        array_name: Identifier,
        shape: Sequence[CoreExpression],
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "structural error",
                f"Array access on {array_name.name_hint!r} has "
                f"{len(node.indices)} indices but {len(shape)} dimensions; "
                f"got shape {shape}.",
                node.provenance,
            ),
        )

    def _report_unsupported_index_type(self, ast_index: Expression) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "type error",
                f"Array-access index {ast_index} is not a supported type; "
                f"got type {type(ast_index).__name__}.",
                ast_index.provenance,
            ),
        )

    def _report_failed_type_synthesis(
        self, ast_index: Expression, exc: FhYCoreTypeError
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "type error",
                f"Failed to synthesize a type for array-access index "
                f"{ast_index}: {exc}",
                ast_index.provenance,
            ),
        )
