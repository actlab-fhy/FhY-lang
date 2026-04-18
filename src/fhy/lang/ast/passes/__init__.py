"""Passes for the FhY AST."""

__all__ = [
    "convert_ast_expression_to_core_expression",
    "collect_identifiers",
    "replace_identifiers",
    "collect_indices",
    "collect_reduced_indices",
    "build_symbol_table",
    "validate_type_qualifiers",
    "validate_expression_statement_lhs",
    "validate_for_all_statements",
    "validate_reductions",
    "validate_index_domains",
    "validate_call_sites",
    "validate_types",
    "FhYSymbolTableBuilderError",
    "FhYTypeQualifierValidatorError",
]

from .ast_to_core_expression_converter import convert_ast_expression_to_core_expression
from .call_site_validator import validate_call_sites
from .expression_statement_lhs_validator import (
    validate_expression_statement_lhs,
)
from .for_all_statement_validator import validate_for_all_statements
from .identifier_collector import collect_identifiers
from .identifier_replacer import replace_identifiers
from .index_collector import collect_indices, collect_reduced_indices
from .index_domain_validator import validate_index_domains
from .reduction_validator import validate_reductions
from .symbol_table_builder import FhYSymbolTableBuilderError, build_symbol_table
from .type_checker import validate_types
from .type_qualifier_validator import (
    FhYTypeQualifierValidatorError,
    validate_type_qualifiers,
)
