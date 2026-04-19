"""Validate the FhY AST."""

__all__ = [
    "validate_ast",
]

from fhy_core import FixpointPassGroup, Identifier, PassManager, SymbolTable

from .node import Module
from .passes import (
    DeadCodeEliminationPass,
    build_symbol_table,
    validate_call_sites,
    validate_expression_statement_lhs,
    validate_for_all_statements,
    validate_index_domains,
    validate_operations,
    validate_reductions,
    validate_type_qualifiers,
    validate_types,
)


def _perform_structural_validation(ast: Module, symbol_table: SymbolTable) -> None:
    validate_expression_statement_lhs(ast)
    validate_for_all_statements(ast, symbol_table)
    validate_reductions(ast, symbol_table)
    validate_call_sites(ast, symbol_table)
    validate_operations(ast)


def _perform_type_checking(ast: Module, symbol_table: SymbolTable) -> None:
    validate_type_qualifiers(ast, symbol_table)
    validate_types(ast, symbol_table)


def _perform_semantic_validation(ast: Module, symbol_table: SymbolTable) -> None:
    validate_index_domains(ast, symbol_table)


def validate_ast(
    ast: Module, perform_optimizations: bool = True
) -> tuple[Module, SymbolTable]:
    """Validate the FhY AST.

    High-level steps (not completely disparate):
        1. Symbol table construction
        2. Structural validation
        3. Type checking
        4. Semantic validation
        5. Optimization (optional)

    Each check below is tagged as either [IMPLEMENTED] or [NOT IMPLEMENTED].
    When implemented, the pass enforcing the check is named in brackets; when
    not, a brief sketch of the enforcement approach is given so it can be
    picked up later.

    Symbol table construction:
        [IMPLEMENTED] (build_symbol_table)
            - Throws an error if a symbol is already defined.

    Structural validation:
        1. Expression statement LHS validation [IMPLEMENTED]
           (validate_expression_statement_lhs)
            - [IMPLEMENTED] Any expression other than an array access
              expression or an identifier expression on the LHS is invalid.
            - [IMPLEMENTED] The array expression of an array-access LHS must
              be an identifier expression.
            - [IMPLEMENTED] An `ExpressionStatement` with `left is None`
              must have its `right` be a `FunctionExpression` (otherwise the
              statement has no effect). Generates a warning.
        2. For-all statement validation [IMPLEMENTED]
           (validate_for_all_statements)
            - [IMPLEMENTED] Index expression is an identifier expression.
            - [IMPLEMENTED] The identifier resolves to an index variable via
              the symbol table.
        3. Reduction validation [IMPLEMENTED] (validate_reductions)
            - [IMPLEMENTED] Indices passed to a reduction are identifier
              expressions resolving to index variables.
            - [IMPLEMENTED] A reduction is passed exactly one argument.
            - [IMPLEMENTED] Indices are distinct.
            - [IMPLEMENTED] Indices are used within the reduction argument.
        4. Call-site validation [IMPLEMENTED] (validate_call_sites)
            - [IMPLEMENTED] The function name is an identifier expression.
            - [IMPLEMENTED] The identifier resolves to a function frame
              (user function or builtin import) via the symbol table.
            - [IMPLEMENTED] The function call has the correct number of
              arguments (for user-defined functions).
            - [IMPLEMENTED] Non-reduction functions do not receive indices.
            - [IMPLEMENTED] A procedure is not called with a left-hand side
              expression.
            - [IMPLEMENTED] A procedure is not called in a value /
              expression position (e.g., inside a `BinaryExpression` or a
              `TernaryExpression` branch).
            - [IMPLEMENTED] Calling an operation as a bare statement
              (no LHS) discards its return value. Generates a warning.
        5. Operation validation [IMPLEMENTED] (validate_operations)
            - [IMPLEMENTED] The operation's arguments must all be scalars.
            - [IMPLEMENTED] The operation's return type must be a scalar OUTPUT.
            - [IMPLEMENTED] The operation's body must contain a return statement.
        6. Control-flow validation [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Every execution path through an `Operation`
              body must reach a `ReturnStatement`. Fires on `Operation.body`
              (including `SelectionStatement` and `ForAllStatement` bodies).
              Needs a CFG / structured-reachability walk.
            - [NOT IMPLEMENTED] A `Procedure.body` must not contain a
              `ReturnStatement`. Trivial visitor scan.
            - [NOT IMPLEMENTED] Flag unreachable statements after a
              `ReturnStatement` in any body (low priority; disjoint from the
              dead-code elimination optimization pass).
            - [NOT IMPLEMENTED] Detect direct and indirect recursion: a
              function whose body transitively calls itself. Fires on
              `FunctionExpression.function.identifier`. Needs a call graph
              built from resolved `FunctionSymbolTableFrame` entries +
              SCC detection.

    Type checking:
        1. Qualifier validation [IMPLEMENTED] (validate_type_qualifiers)
            - [IMPLEMENTED] INPUTs only defined in argument lists and are
              read-only (cannot be assigned to).
            - [IMPLEMENTED] TEMPs only defined in declaration statements.
            - [IMPLEMENTED] OUTPUTs only defined in argument lists or return
              types.
            - [IMPLEMENTED] PARAMs are compile-time constants (cannot be
              assigned to).
            - [NOT IMPLEMENTED] `Native.args` qualifier restrictions are
              handled by the shared `visit_argument` in this pass; add a
              focused test to lock in that behaviour.
        2. Type checking [IMPLEMENTED] (validate_types)
            - [IMPLEMENTED] Expression statement LHS and RHS types are
              compatible (structural element-type match with primitive
              data-type promotion, matching free-index sets).
            - [IMPLEMENTED] Declaration-statement initializer types match
              the declared type.
            - [IMPLEMENTED] Return-statement expression types match the
              operation's declared return type.
            - [IMPLEMENTED] Ternary-branch and binary-operand types promote
              to a common primitive data type with matching shapes.
            - [NOT IMPLEMENTED] `UnaryExpression` operator constraints: `!`
              (logical not) requires a bool-convention value; `~` (bitwise
              not) must reject floats/complex data types. Add operator-
              aware dispatch in `_infer_type`.
            - [NOT IMPLEMENTED] `TupleAccessExpression.element_index` must
              be a non-negative `IntLiteral` strictly less than the arity
              of the `tuple_expression`'s `TupleType`. Extend `_infer_type`.
            - [NOT IMPLEMENTED] Selection/ternary condition must be a
              scalar bool-convention value (once the bool convention is
              canonical). Extend `_infer_ternary` / add a selection check.
            - [NOT IMPLEMENTED] `Procedure.templates` / `Operation.templates`
              parameters must be uniquely named and actually referenced in
              the signature or body's `TemplateDataType` nodes. Collect
              declared templates, walk the function body, error on
              unreferenced or re-defined templates.
            - [NOT IMPLEMENTED] Shape-dimension identifier coherence: the
              same shape identifier used in multiple argument types must
              refer to the same symbol-table entry with identical bounds.

    Semantic validation:
        1. Index-domain validation [IMPLEMENTED] (validate_index_domains)
            - [IMPLEMENTED] Number of indices matches the array's
              dimensionality.
            - [IMPLEMENTED] Each index is either an `IndexType` variable or
              a scalar unsigned-integer `PARAM` expression.
            - [IMPLEMENTED] Each array access is within the array's bounds
              (universally, via a z3 check on `lower >= 1 AND upper <= dim`).
            - [NOT IMPLEMENTED] Tuple-access bounds — see tuple rule above.
        2. Definite assignment [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Every `TEMP` variable is assigned before it
              is read. Partially supported by `LivenessAnalysis` (live-in
              sets) but no pass currently raises on use-before-def.
            - [NOT IMPLEMENTED] Every `OUTPUT` argument is assigned on every
              control-flow path before the function returns. Needs forward
              data-flow over the body's CFG.
        3. Constant safety [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Division / modulo by a compile-time literal
              zero. Trivial structural check over
              `BinaryExpression(DIVISION | FLOORDIV | MODULO, _, IntLiteral(0))`
              (and the equivalent `FloatLiteral(0.0)` and negative-zero
              forms). Low priority but cheap to add.

    Optimizations:
        1. Dead code elimination [IMPLEMENTED] (DeadCodeEliminationPass)
            - [IMPLEMENTED] Removes `ExpressionStatement`s that assign to a
              TEMP identifier target whose value is not in the statement's
              live-out set and whose RHS has no function calls.
            - [IMPLEMENTED] Removes initialized TEMP `DeclarationStatement`s
              under the same liveness / side-effect conditions.
            - [NOT IMPLEMENTED] Drop uninitialized TEMP `DeclarationStatement`s
              whose variable is never assigned and never read after other
              DCE iterations converge. Extend `_is_dead_declaration` to
              track never-written TEMPs using a forward "ever-assigned"
              scan, or a second liveness-style analysis over declarations.
            - [NOT IMPLEMENTED] Remove `ExpressionStatement`s with an
              `ArrayAccessExpression` LHS when the array is a TEMP and the
              written element is provably not in live-out (requires
              per-element / per-index liveness, not just per-variable).
        2. Constant folding [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Fold `BinaryExpression` and `UnaryExpression`
              nodes whose operands are all `IntLiteral` / `FloatLiteral` /
              `ComplexLiteral` into a single literal of the promoted
              primitive data type. Pure `Transformer` pass — reuse
              `promote_primitive_data_types` / `promote_core_data_types`
              from fhy_core to pick the result type. Place in the DCE
              fixpoint group so folded expressions create fresh dead code.
        3. Algebraic simplification [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Pattern-match identity / absorbing cases on
              `BinaryExpression`: `x + 0 -> x`, `0 + x -> x`, `x - 0 -> x`,
              `x * 1 -> x`, `1 * x -> x`, `x * 0 -> 0` (only when `x` is
              side-effect-free), `x / 1 -> x`, `x ** 1 -> x`, `x || true
              -> true`, `x && false -> false`. Also collapse `UnaryExpression`
              pairs (`!!x -> x`, `~~x -> x`, `--x -> x`). Pure `Transformer`
              pass; depends on constant folding to normalize literals first.
        4. Strength reduction [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Rewrite integer `BinaryExpression`s to
              cheaper equivalents: `x * 2**k -> x << k`, `x / 2**k -> x >> k`
              (signed-right-shift caveat), `x % 2**k -> x & (2**k - 1)` for
              unsigned. Check the LHS operand's `CoreDataType` via the
              symbol table; restrict to integer data types.
        5. Canonicalization [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] For commutative `BinaryOperation`s
              (`+`, `*`, `&`, `|`, `^`, `==`, `!=`, `&&`, `||`), move
              literal operands to the right and sort identifier operands
              by identifier id. Normalize `FunctionExpression.indices`
              within reductions by sorted index id. Enables CSE and
              cheaper structural equivalence.
        6. Copy / constant propagation [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Build a forward "reaching definitions"
              `Analysis` (analogous to `LivenessAnalysis`, same block /
              selection / for-all handling) that maps every
              `IdentifierExpression` use to the set of possibly-reaching
              `ExpressionStatement` / `DeclarationStatement` definitions.
              When a use has exactly one reaching definition whose RHS is
              an `IdentifierExpression` or a literal, rewrite the use via
              a `Transformer`. Place in the DCE fixpoint group — the
              propagation creates more dead stores for DCE to drop.
        7. Common subexpression elimination [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Per straight-line block, hash pure
              sub-expressions (reuse the existing `_FunctionCallFinder`
              side-effect check from DCE); on the second occurrence,
              introduce a synthesized TEMP `DeclarationStatement` +
              `ExpressionStatement` before the first use and substitute
              the TEMP at every occurrence. Benefits greatly from
              canonicalization running first.
        8. Unreachable code elimination [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Collapse `SelectionStatement` whose
              condition folds to a literal bool: replace with the taken
              body. Drop statements that follow a `ReturnStatement` in the
              same block. Pure `Transformer` pass; gains most value when
              paired with constant folding in the DCE fixpoint group.
        9. Loop-invariant code motion [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] For each `ForAllStatement`, identify
              sub-expressions whose free identifiers do not intersect the
              loop index (and any variables written inside the body).
              `IdentifierCollector` already provides the free-identifier
              walk; combine with a "written-in-body" set derived from
              `LivenessAnalysis`'s `def` logic. Hoist invariant
              sub-expressions into fresh TEMP declarations immediately
              preceding the loop.
       10. Operation inlining [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] At each `FunctionExpression` whose target
              resolves to an `Operation` (guaranteed pure by the qualifier
              validator's `OUTPUT`-return-type rule), substitute the
              callee body at the call site: alpha-rename declarations via
              `IdentifierReplacer`, map callee arguments to actual
              expressions, and replace the call with the renamed return
              expression. Size-threshold and recursion check (see the
              call-graph item under Control-flow validation) gate the
              rewrite.
       11. Dead declaration cleanup [NOT IMPLEMENTED]
            - [NOT IMPLEMENTED] Follow-up to DCE once the fixpoint
              converges: any `DeclarationStatement` whose variable is
              never written (after all dead stores have been eliminated)
              and never read is removable. Needs a simple pair of
              per-function identifier-use / identifier-write scans.

    Args:
        ast: The FhY AST to validate.
        perform_optimizations: Whether to perform optimizations on the AST.

    Returns:
        A tuple containing the validated AST and the symbol table.

    Raises:
        Exception: If the FhY AST is invalid.

    """
    symbol_table = build_symbol_table(ast)
    _perform_structural_validation(ast, symbol_table)
    _perform_type_checking(ast, symbol_table)
    _perform_semantic_validation(ast, symbol_table)

    if perform_optimizations:
        pass_manager = PassManager[Module](Identifier("fhy_ast_pass_manager"))
        dce_fixpoint_group = FixpointPassGroup[Module](
            name=Identifier("fhy_ast_dead_code_elimination_fixpoint"),
        )
        dce_fixpoint_group.add_pass(
            DeadCodeEliminationPass(pass_manager.analysis_manager, symbol_table)
        )
        pass_manager.add_fixpoint_group(dce_fixpoint_group)
        ast = pass_manager.run(ast).output

    return ast, symbol_table
