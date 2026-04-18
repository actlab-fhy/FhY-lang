"""Validate the FhY AST."""

__all__ = [
    "validate_ast",
]

from fhy_core import SymbolTable

from .node import Module
from .passes import (
    build_symbol_table,
    validate_expression_statement_lhs,
    validate_type_qualifiers,
)


def validate_ast(ast: Module) -> tuple[Module, SymbolTable]:
    """Validate the FhY AST.

    Steps:
        1. Symbol table construction
            - Throws an error if a symbol is already defined.
        2. Expression statement LHS validation
            - Throws an error if the left-hand side of an expression statement is
              invalid.
                - Any expression other than an array access expression or an identifier
                  expression is invalid.
                - In the case of an array access expression, the array expression must
                  be an identifier expression.
        3. Qualifier validation
            - Throws an error if type qualifier rules are violated.
                - INPUTs are read-only and only defined in argument lists.
                - TEMPs are read-write and only defined in declaration statements.
                - OUTPUTs are write-only and only defined in argument lists or return
                  types.
                - PARAMs are compile-time constants.
        4. For-all statement validation
            - Throws an error if the index expression is not an identifier expression.
            - Throws an error if the identifier is not an index via the symbol table.
        5. Reduction validation
            - Throws an error if the expressions for indices passed to a reduction
              are not identifier expressions and the identifiers are not indices via
              the symbol table.
        6. Index-domain validation
            - Throws an error if the an array access is out of bounds.
        7. Call-site validation
            - Throws an error if a function call is performed in an invalid manner.
                - The expression passed as the function name is not an identifier
                  expression.
                - The function name is not defined as a function via the symbol table.
                - A non-reduction function does not have any indices passed to it.
                - A procedure is not used with a left-hand side expression.
        8. Type checking
            - Throws an error if the type of an expression is not
              compatible with the type of the variable it is assigned to.

    Args:
        ast: The FhY AST to validate.

    Returns:
        A tuple containing the validated AST and the symbol table.

    Raises:
        Exception: If the FhY AST is invalid.

    """
    symbol_table = build_symbol_table(ast)
    validate_expression_statement_lhs(ast)
    validate_type_qualifiers(ast, symbol_table)
    return ast, symbol_table
