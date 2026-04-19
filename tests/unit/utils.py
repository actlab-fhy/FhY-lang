"""Shared utilities for unit tests."""

from fhy.lang.ast import (
    Argument,
    DeclarationStatement,
    ExpressionStatement,
    IdentifierExpression,
    Module,
    Procedure,
    QualifiedType,
    Statement,
)
from fhy_core import (
    Identifier,
    Provenance,
    Type,
    TypeQualifier,
)


def make_argument(
    name: Identifier, qualifier: TypeQualifier, base_type: Type
) -> Argument:
    """Construct an `Argument` with the given name, qualifier, and base type."""
    return Argument(
        name=name,
        qualified_type=QualifiedType(
            base_type=base_type,
            type_qualifier=qualifier,
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
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
            provenance=Provenance.unknown(),
        ),
        provenance=Provenance.unknown(),
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
            provenance=Provenance.unknown(),
        ),
        expression=IdentifierExpression(
            identifier=initializer_identifier, provenance=Provenance.unknown()
        ),
        provenance=Provenance.unknown(),
    )


def make_identifier_assignment(
    target: Identifier, source: Identifier
) -> ExpressionStatement:
    """Construct an `target = source` assignment between two identifiers."""
    return ExpressionStatement(
        left=IdentifierExpression(identifier=target, provenance=Provenance.unknown()),
        right=IdentifierExpression(identifier=source, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
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
        provenance=Provenance.unknown(),
    )


def make_module_with_statement(statement: Statement) -> Module:
    """Wrap a single top-level statement in a new `Module`."""
    return Module(statements=(statement,), provenance=Provenance.unknown())
