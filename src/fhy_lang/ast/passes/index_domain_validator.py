"""Validate that array-access indices fall within the array's dimension domains."""

__all__ = [
    "IndexDomainValidator",
]

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

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
    does_expression_imply,
    get_logger,
    holds_for_all_free_assignments,
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

from fhy_lang.ast.node import (
    ArrayAccessExpression,
    Expression,
    IdentifierExpression,
)
from fhy_lang.ast.shape import narrow_shape

from ..pprint import pformat_ast
from .analysis_pass_with_symbol_table import AnalysisPassWithSymbolTable
from .ast_to_core_expression_converter import (
    convert_ast_expression_to_core_expression,
)
from .utils import format_diagnostic_message

_logger: logging.Logger = get_logger(__name__)

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
    core_index: CoreExpression
    preconditions: tuple[CoreExpression, ...]
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


def _iter_subexpressions(expression: CoreExpression) -> Iterator[CoreExpression]:
    yield expression
    for child in expression.get_visit_children():
        if isinstance(child, CoreExpression):
            yield from _iter_subexpressions(child)


def _contains_non_integer_literal(expression: CoreExpression) -> bool:
    for sub in _iter_subexpressions(expression):
        if isinstance(sub, CoreLiteralExpression) and (
            isinstance(sub.value, bool) or not isinstance(sub.value, int)
        ):
            return True
    return False


def _conjoin(expressions: Sequence[CoreExpression]) -> CoreExpression:
    result = expressions[0]
    for expr in expressions[1:]:
        result = result.logical_and(expr)
    return result


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
    - Classifies every free identifier in the index expression as either an
      :class:`IndexType` variable (whose ``[lower_bound, upper_bound]`` is
      added to the precondition for the z3 check) or a scalar
      unsigned-integer ``PARAM`` (left free for z3).
    - Uses z3 to verify that the precondition implies
      ``index >= 1 AND index <= dim_size``. This proves the symbolic
      minimum and maximum of the index expression are inside the array
      dimension.

    All failures are reported as diagnostics; the visitor continues so the
    user gets every diagnostic in one run.

    """

    def visit_array_access_expression(self, node: ArrayAccessExpression) -> None:
        if not isinstance(node.array_expression, IdentifierExpression):
            self._report_non_identifier_array_expression(node)
            return
        array_name = node.array_expression.identifier
        _logger.debug(
            "Index domain check: array=%s, index_count=%d.",
            array_name.name_hint,
            len(node.indices),
        )
        try:
            frame = self.get_frame_from_namespace(self.current_namespace, array_name)
        except SymbolTableError:
            return
        if not isinstance(frame, VariableSymbolTableFrame) or not isinstance(
            frame.type, NumericalType
        ):
            self._report_non_vector_access(node, array_name, frame)
            return
        shape = narrow_shape(frame.type.shape)
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
        if _contains_non_integer_literal(core_index):
            self._report_non_integer_index(ast_index)
            return
        preconditions = self._collect_index_preconditions(ast_index, core_index)
        if preconditions is None:
            return
        self._check_index_in_domain(
            _DomainCheckInput(
                array_name=array_name,
                ast_index=ast_index,
                core_index=core_index,
                preconditions=preconditions,
                dimension_size=dimension_size,
            )
        )

    def _collect_index_preconditions(
        self, ast_index: Expression, core_index: CoreExpression
    ) -> tuple[CoreExpression, ...] | None:
        preconditions: list[CoreExpression] = []
        for identifier in collect_identifiers(core_index):
            try:
                identifier_type, identifier_qualifier = self._get_identifier_type(
                    identifier
                )
            except (SymbolTableError, FhYCoreTypeError):
                return None
            if isinstance(identifier_type, IndexType):
                identifier_expression = CoreIdentifierExpression(identifier)
                preconditions.append(
                    identifier_expression >= identifier_type.lower_bound
                )
                preconditions.append(
                    identifier_expression <= identifier_type.upper_bound
                )
            elif _is_unsigned_integer_param(identifier_type, identifier_qualifier):
                continue
            else:
                self._report_unsupported_index_identifier(
                    ast_index, identifier, identifier_type, identifier_qualifier
                )
                return None
        return tuple(preconditions)

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
        lower_bound_constraint = check_input.core_index >= CoreLiteralExpression(1)
        upper_bound_constraint = check_input.core_index <= check_input.dimension_size
        constraint = lower_bound_constraint.logical_and(upper_bound_constraint)
        symbol_types = self._build_symbol_types(check_input, constraint)
        if check_input.preconditions:
            antecedent = _conjoin(check_input.preconditions)
            holds = does_expression_imply(antecedent, constraint, symbol_types)
        else:
            holds = holds_for_all_free_assignments(set(), constraint, symbol_types)
        if holds is True:
            return
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "semantic error",
                f"Array access on {check_input.array_name.name_hint!r} is out "
                f"of bounds: index expression {check_input.core_index} is not "
                f"provably within [1, {check_input.dimension_size}] given the "
                "declared bounds of its index variables.",
                check_input.ast_index.provenance,
            ),
        )

    @staticmethod
    def _build_symbol_types(
        check_input: _DomainCheckInput, constraint: CoreExpression
    ) -> dict[Identifier, SymbolType]:
        identifiers: set[Identifier] = set()
        identifiers |= collect_identifiers(constraint)
        for precondition in check_input.preconditions:
            identifiers |= collect_identifiers(precondition)
        return dict.fromkeys(identifiers, SymbolType.INT)

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
                f"Array-access index {pformat_ast(ast_index)} is not a "
                f"supported type; got type {type(ast_index).__name__}.",
                ast_index.provenance,
            ),
        )

    def _report_non_integer_index(self, ast_index: Expression) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "type error",
                f"Array-access index {pformat_ast(ast_index)} contains a "
                "non-integer literal; indices must be integer-valued.",
                ast_index.provenance,
            ),
        )

    def _report_unsupported_index_identifier(
        self,
        ast_index: Expression,
        identifier: Identifier,
        identifier_type: Type,
        identifier_qualifier: TypeQualifier,
    ) -> None:
        self.report(
            DiagnosticLevel.ERROR,
            format_diagnostic_message(
                "type error",
                f"Identifier {identifier.name_hint!r} used in array-access "
                f"index {pformat_ast(ast_index)} must be either an index "
                "variable or a scalar unsigned-integer PARAM; got type "
                f"{identifier_type} with qualifier "
                f"{identifier_qualifier.value!r}.",
                ast_index.provenance,
            ),
        )
