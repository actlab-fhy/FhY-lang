"""Passes for the FhY AST."""

from .ast_to_core_expression_converter import convert_ast_expression_to_core_expression
from .identifier_collector import collect_identifiers
from .identifier_replacer import replace_identifiers
from .index_collector import collect_indices, collect_reduced_indices
from .symbol_table_builder import build_symbol_table

__all__ = [
    "convert_ast_expression_to_core_expression",
    "collect_identifiers",
    "replace_identifiers",
    "collect_indices",
    "collect_reduced_indices",
    "build_symbol_table",
]
