"""Validate the FhY AST."""

__all__ = [
    "ValidationFailedError",
    "ValidationReport",
    "build_semantic_validation_manager",
    "build_structural_validation_manager",
    "validate_ast",
]

from typing import Any, cast

from fhy_core import (
    CompilerPass,
    FixpointPassGroup,
    Identifier,
    PassManager,
    SymbolTable,
    ValidationFailedError,
    ValidationManager,
    ValidationReport,
)

from .node import Module
from .passes import (
    AlgebraicSimplificationPass,
    CallSiteValidator,
    ConstantFoldingPass,
    ConstantSafetyValidator,
    DeadCodeEliminationPass,
    DefiniteAssignmentValidator,
    ExpressionStatementLHSValidator,
    ForAllStatementValidator,
    IndexDomainValidator,
    OperationValidator,
    RecursionValidator,
    ReductionValidator,
    ReturnValidator,
    TupleAccessValidator,
    TypeChecker,
    TypeQualifierValidator,
    build_symbol_table,
)


def _as_module_validator(
    validator: CompilerPass[Any, Any],
) -> CompilerPass[Module, Any]:
    """Adapt a ``CompilerPass[Node, ...]`` for a ``ValidationManager[Module]``.

    Most FhY validator passes inherit from ``AnalysisVisitablePass[Node]``,
    which declares their input as ``Node`` even though at runtime they are
    always invoked with a :class:`Module`. The :class:`ValidationManager`
    pipeline is parameterised by the concrete IR type it runs over, so this
    helper performs the matching type-level narrowing via a single ``cast``
    — the runtime dispatch is unchanged because ``Module`` is a ``Node``.

    """
    return cast("CompilerPass[Module, Any]", validator)


def build_structural_validation_manager(
    symbol_table: SymbolTable,
) -> ValidationManager[Module]:
    """Build the structural-validation pipeline.

    Structural validators enforce the invariants every subsequent pass
    assumes: every assignment has a sensible LHS, every forall and
    reduction has well-formed index expressions, every call site resolves,
    and operations have scalar argument / return shapes. These passes run
    before any type- or semantic-level analysis — if they fail, later
    passes would likely crash or produce spurious diagnostics.

    Args:
        symbol_table: The symbol table built for the module.

    Returns:
        A :class:`ValidationManager` pre-populated with the structural
        validators in a stable execution order.

    """
    manager = ValidationManager[Module](Identifier("fhy_ast_structural_validation"))
    manager.add(_as_module_validator(ExpressionStatementLHSValidator()))
    manager.add(_as_module_validator(ForAllStatementValidator(symbol_table)))
    manager.add(_as_module_validator(ReductionValidator(symbol_table)))
    manager.add(_as_module_validator(CallSiteValidator(symbol_table)))
    manager.add(_as_module_validator(OperationValidator()))
    manager.add(_as_module_validator(ReturnValidator()))
    manager.add(_as_module_validator(RecursionValidator()))
    return manager


def build_semantic_validation_manager(
    symbol_table: SymbolTable,
) -> ValidationManager[Module]:
    """Build the type + semantic validation pipeline.

    These passes rely on the structural invariants above. Running them on
    a structurally ill-formed module would at best produce cascaded /
    confusing diagnostics and at worst raise from within the pass. We
    therefore only invoke this manager after the structural manager has
    completed without ERROR diagnostics.

    Args:
        symbol_table: The symbol table built for the module.

    Returns:
        A :class:`ValidationManager` pre-populated with the type checker,
        qualifier validator, index-domain validator, definite-assignment
        validator, and constant-safety validator, in a stable order.

    """
    manager = ValidationManager[Module](Identifier("fhy_ast_semantic_validation"))
    manager.add(_as_module_validator(TypeQualifierValidator(symbol_table)))
    manager.add(_as_module_validator(TypeChecker(symbol_table)))
    manager.add(_as_module_validator(TupleAccessValidator(symbol_table)))
    manager.add(_as_module_validator(IndexDomainValidator(symbol_table)))
    manager.add(_as_module_validator(DefiniteAssignmentValidator(symbol_table)))
    manager.add(_as_module_validator(ConstantSafetyValidator()))
    return manager


def validate_ast(
    ast: Module, perform_optimizations: bool = True
) -> tuple[Module, SymbolTable]:
    """Validate the FhY AST.

    High-level steps:
        1. Symbol table construction.
        2. Structural validation (via a :class:`ValidationManager`).
        3. Semantic validation (via a second :class:`ValidationManager`),
           run only if step 2 produced no ERROR diagnostics.
        4. Optional optimization (DCE fixpoint).

    Validators report diagnostics through ``CompilerPass.report(...)`` and
    never raise directly. After each :class:`ValidationManager` runs, we
    call :meth:`ValidationReport.raise_if_failed` to convert any
    accumulated ERROR diagnostics into a single :class:`ValidationFailedError`
    whose message is the aggregated report. Warnings and info diagnostics
    from passes that completed without errors are preserved but do not
    stop compilation.

    The structural manager intentionally runs on its own: its validators
    establish the invariants (well-formed LHS, well-formed calls, etc.)
    that the semantic validators rely on, so running the semantic manager
    on a structurally invalid AST would produce misleading cascaded
    diagnostics (or outright crashes inside a pass). If the structural
    manager reports errors, compilation stops there.

    Args:
        ast: The FhY AST to validate.
        perform_optimizations: Whether to perform optimizations on the AST.

    Returns:
        A tuple containing the validated AST and the symbol table.

    Raises:
        ValidationFailedError: If any ERROR diagnostic was emitted by
            either the structural or semantic validation pipeline.

    """
    symbol_table = build_symbol_table(ast)

    pre_constant_folding_pass_manager = PassManager[Module](
        Identifier("fhy_ast_pre_constant_folding_pass_manager")
    )
    pre_constant_folding_pass_manager.add_pass(
        cast(CompilerPass[Module, Module], ConstantFoldingPass())
    )
    ast = pre_constant_folding_pass_manager.run(ast).output

    structural_report = build_structural_validation_manager(symbol_table).validate(ast)
    structural_report.raise_if_failed()

    semantic_report = build_semantic_validation_manager(symbol_table).validate(ast)
    semantic_report.raise_if_failed()

    if perform_optimizations:
        pass_manager = PassManager[Module](Identifier("fhy_ast_pass_manager"))
        fixpoint_group = FixpointPassGroup[Module](
            name=Identifier("fhy_ast_fixpoint"),
        )
        fixpoint_group.add_pass(
            cast(
                CompilerPass[Module, Module],
                ConstantFoldingPass(),
            )
        )
        fixpoint_group.add_pass(
            cast(
                CompilerPass[Module, Module],
                AlgebraicSimplificationPass(),
            )
        )
        fixpoint_group.add_pass(
            cast(
                CompilerPass[Module, Module],
                DeadCodeEliminationPass(symbol_table),
            )
        )
        pass_manager.add_fixpoint_group(fixpoint_group)
        ast = pass_manager.run(ast).output

    return ast, symbol_table
