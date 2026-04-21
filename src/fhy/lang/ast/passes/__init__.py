"""Passes for the FhY AST."""

__all__ = [
    "convert_ast_expression_to_core_expression",
    "collect_identifiers",
    "replace_identifiers",
    "collect_indices",
    "collect_reduced_indices",
    "build_symbol_table",
    "CallSiteValidator",
    "ConstantSafetyValidator",
    "DefiniteAssignmentValidator",
    "ExpressionStatementLHSValidator",
    "ForAllStatementValidator",
    "IndexDomainValidator",
    "OperationValidator",
    "ReductionValidator",
    "ReturnValidator",
    "TypeChecker",
    "TypeQualifierValidator",
    "DeadCodeEliminationPass",
    "LivenessAnalysis",
    "LivenessResult",
    "FhYSymbolTableBuilderError",
]

from .ast_to_core_expression_converter import convert_ast_expression_to_core_expression
from .call_site_validator import CallSiteValidator
from .constant_safety_validator import ConstantSafetyValidator
from .dead_code_elimination import DeadCodeEliminationPass
from .definite_assignment import DefiniteAssignmentValidator
from .expression_statement_lhs_validator import ExpressionStatementLHSValidator
from .for_all_statement_validator import ForAllStatementValidator
from .identifier_collector import collect_identifiers
from .identifier_replacer import replace_identifiers
from .index_collector import collect_indices, collect_reduced_indices
from .index_domain_validator import IndexDomainValidator
from .liveness_analysis import LivenessAnalysis, LivenessResult
from .operation_validator import OperationValidator
from .reduction_validator import ReductionValidator
from .return_validator import ReturnValidator
from .symbol_table_builder import FhYSymbolTableBuilderError, build_symbol_table
from .type_checker import TypeChecker
from .type_qualifier_validator import TypeQualifierValidator
