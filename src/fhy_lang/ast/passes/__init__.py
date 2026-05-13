"""Passes for the FhY AST."""

__all__ = [
    "AlgebraicSimplificationPass",
    "CallSiteValidator",
    "ConstantFoldingPass",
    "ConstantSafetyValidator",
    "CrossIterationStateValidator",
    "DeadCodeEliminationPass",
    "DefiniteAssignmentValidator",
    "ExpressionSideEffectAnalysis",
    "ExpressionStatementLHSValidator",
    "FhYSymbolTableBuilderError",
    "ForAllStatementValidator",
    "IndexDomainValidator",
    "LivenessAnalysis",
    "LivenessResult",
    "MainProcedureValidator",
    "ModuleScopeValidator",
    "OperationValidator",
    "RecursionValidator",
    "ReductionValidator",
    "ReservedNameValidator",
    "ReturnValidator",
    "Transformer",
    "TupleAccessValidator",
    "TypeChecker",
    "TypeQualifierValidator",
    "UnreachableCodeEliminationPass",
    "build_symbol_table",
    "collect_identifiers",
    "collect_indices",
    "collect_reduced_indices",
    "convert_ast_expression_to_core_expression",
    "replace_identifiers",
]

from .algebraic_simplification import AlgebraicSimplificationPass
from .ast_to_core_expression_converter import convert_ast_expression_to_core_expression
from .call_site_validator import CallSiteValidator
from .constant_folding import ConstantFoldingPass
from .constant_safety_validator import ConstantSafetyValidator
from .cross_iteration_state_validator import CrossIterationStateValidator
from .dead_code_elimination import DeadCodeEliminationPass
from .definite_assignment import DefiniteAssignmentValidator
from .expression_side_effect_analysis import ExpressionSideEffectAnalysis
from .expression_statement_lhs_validator import ExpressionStatementLHSValidator
from .for_all_statement_validator import ForAllStatementValidator
from .identifier_collector import collect_identifiers
from .identifier_replacer import replace_identifiers
from .index_collector import collect_indices, collect_reduced_indices
from .index_domain_validator import IndexDomainValidator
from .liveness_analysis import LivenessAnalysis, LivenessResult
from .main_procedure_validator import MainProcedureValidator
from .module_scope_validator import ModuleScopeValidator
from .operation_validator import OperationValidator
from .recursion_validator import RecursionValidator
from .reduction_validator import ReductionValidator
from .reserved_name_validator import ReservedNameValidator
from .return_validator import ReturnValidator
from .symbol_table_builder import FhYSymbolTableBuilderError, build_symbol_table
from .transformer import Transformer
from .tuple_access_validator import TupleAccessValidator
from .type_checker import TypeChecker
from .type_qualifier_validator import TypeQualifierValidator
from .unreachable_code_elimination import UnreachableCodeEliminationPass
