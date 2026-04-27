"""Shared utilities for unit tests."""

from typing import Any, cast

from fhy_core import (
    CompilerPass,
    Identifier,
    Type,
    TypeQualifier,
    ValidationManager,
)

from fhy_lang.ast import (
    Argument,
    DeclarationStatement,
    ExpressionStatement,
    IdentifierExpression,
    Module,
    Procedure,
    QualifiedType,
    Statement,
)


def run_validator(validator: CompilerPass[Any, Any], module: Module) -> None:
    """Run ``validator`` against ``module`` and raise on ERROR diagnostics.

    Test-only helper that wraps the validator in a single-pass
    :class:`~fhy_core.pass_infrastructure.ValidationManager` so every
    "run one validator in isolation" call goes through the same
    diagnostic-aggregation machinery as the production pipelines in
    :mod:`fhy_lang.ast.validate`. The cast is always safe at runtime
    because a :class:`Module` is a :class:`Node`.

    Args:
        validator: The validator pass to run.
        module: The AST module to validate.

    Raises:
        ValidationFailedError: If the validator emitted any ERROR-level
            diagnostic.

    """
    manager = ValidationManager[Module](
        Identifier(f"test::{validator.get_pass_name()}")
    )
    manager.add(cast("CompilerPass[Module, Any]", validator))
    manager.validate(module).raise_if_failed()


def make_argument(
    name: Identifier, qualifier: TypeQualifier, base_type: Type
) -> Argument:
    """Construct an `Argument` with the given name, qualifier, and base type."""
    return Argument(
        name=name,
        qualified_type=QualifiedType(
            base_type=base_type,
            type_qualifier=qualifier,
        ),
    )


def make_uninitialized_temp_declaration(
    name: Identifier, base_type: Type
) -> DeclarationStatement:
    """Construct an uninitialized TEMP `DeclarationStatement`."""
    return DeclarationStatement(
        variable_name=name,
        variable_type=QualifiedType(
            base_type=base_type,
            type_qualifier=TypeQualifier.TEMP,
        ),
    )


def make_initialized_temp_declaration(
    name: Identifier, base_type: Type, initializer_identifier: Identifier
) -> DeclarationStatement:
    """Construct a TEMP `DeclarationStatement` initialized from an identifier."""
    return DeclarationStatement(
        variable_name=name,
        variable_type=QualifiedType(
            base_type=base_type,
            type_qualifier=TypeQualifier.TEMP,
        ),
        expression=IdentifierExpression(identifier=initializer_identifier),
    )


def make_identifier_assignment(
    target: Identifier, source: Identifier
) -> ExpressionStatement:
    """Construct an `target = source` assignment between two identifiers."""
    return ExpressionStatement(
        left=IdentifierExpression(identifier=target),
        right=IdentifierExpression(identifier=source),
    )


def make_procedure(
    name: Identifier,
    args: tuple[Argument, ...],
    body: tuple[Statement, ...],
) -> Procedure:
    """Construct a `Procedure` with the given name, args, and body."""
    return Procedure(
        name=name,
        args=args,
        body=body,
    )


def make_module_with_statement(statement: Statement) -> Module:
    """Wrap a single top-level statement in a new `Module`."""
    return Module(statements=(statement,))
