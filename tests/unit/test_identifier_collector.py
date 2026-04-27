"""Tests the identifier collector AST pass."""

from fhy_core import (
    CoreDataType,
    Identifier,
    NumericalType,
    TemplateDataType,
    TypeQualifier,
)
from fhy_core import IdentifierExpression as CoreIdentifierExpression

from fhy_lang.ast import (
    Argument,
    ArrayAccessExpression,
    DeclarationStatement,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Operation,
    Procedure,
    QualifiedType,
)
from fhy_lang.ast.passes.identifier_collector import (
    collect_identifiers,
)


def test_empty_module(empty_module_ast):
    """Test an empty module returns empty set."""
    identifiers = collect_identifiers(empty_module_ast)
    assert len(identifiers) == 1
    identifier = next(iter(identifiers))
    assert identifier == empty_module_ast.name


def test_declaration_statement():
    """Test retrieval of identifiers from a declaration statement."""
    x = Identifier("x")
    N = Identifier("N")
    statement = DeclarationStatement(
        variable_name=x,
        variable_type=QualifiedType(
            base_type=NumericalType(
                CoreDataType.INT32, shape=[CoreIdentifierExpression(N)]
            ),
            type_qualifier=TypeQualifier.TEMP,
        ),
        expression=IntLiteral(value=5),
    )

    result = collect_identifiers(statement)
    assert result == {x, N}


def test_expression_statement(x_plus_y_expression_ast):
    """Test retrieval of identifiers from an expression statement."""
    right, x, y = x_plus_y_expression_ast
    statement = ExpressionStatement(
        right=right,
    )
    result = collect_identifiers(statement)
    assert result == {x, y}


def test_function_expression():
    """Test retrieval of identifiers from a function expression."""
    x = Identifier("x")
    y = Identifier("y")
    i = Identifier("i")
    foo = Identifier("foo")
    func = FunctionExpression(
        function=IdentifierExpression(identifier=foo),
        indices=[IdentifierExpression(identifier=i)],
        template_types=[],
        args=[
            Argument(
                name=x,
                qualified_type=QualifiedType(
                    base_type=NumericalType(CoreDataType.INT32),
                    type_qualifier=TypeQualifier.TEMP,
                ),
            ),
            Argument(
                name=y,
                qualified_type=QualifiedType(
                    base_type=NumericalType(CoreDataType.INT32),
                    type_qualifier=TypeQualifier.TEMP,
                ),
            ),
        ],
    )
    result = collect_identifiers(func)
    assert result == {x, y, foo, i}


def test_array_access_expression():
    """Test retrieval of identifiers from an array access expression."""
    i = Identifier("i")
    arr = Identifier("arr")
    arr_access = ArrayAccessExpression(
        array_expression=IdentifierExpression(identifier=arr),
        indices=[IdentifierExpression(identifier=i)],
    )
    result = collect_identifiers(arr_access)
    assert result == {i, arr}


def test_empty_procedure():
    """Test retrieval of identifiers from an empty procedure."""
    foo = Identifier("foo")
    proc = Procedure(name=foo, templates=[], args=[], body=[])
    result = collect_identifiers(proc)
    assert result == {foo}


def test_empty_operation():
    """Test retrieval of identifiers from an empty operation."""
    foo = Identifier("foo")
    op = Operation(
        name=foo,
        templates=[],
        args=[],
        body=[],
        return_type=QualifiedType(
            base_type=NumericalType(data_type=CoreDataType.INT32),
            type_qualifier=TypeQualifier.TEMP,
        ),
    )
    result = collect_identifiers(op)
    assert result == {foo}


def test_function_arguments():
    """Test retrieval of identifiers from arguments."""
    x = Identifier("x")
    y = Identifier("y")
    bar = Identifier("bar")
    proc = Procedure(
        name=bar,
        templates=[],
        args=[
            Argument(
                name=x,
                qualified_type=QualifiedType(
                    base_type=NumericalType(CoreDataType.INT32),
                    type_qualifier=TypeQualifier.TEMP,
                ),
            ),
            Argument(
                name=y,
                qualified_type=QualifiedType(
                    base_type=NumericalType(CoreDataType.INT32),
                    type_qualifier=TypeQualifier.TEMP,
                ),
            ),
        ],
        body=[],
    )
    result = collect_identifiers(proc)
    assert result == {x, y, bar}


def test_function_template_types():
    """Test retrieval of identifiers from template types."""
    T = Identifier("T")
    bar = Identifier("bar")
    proc = Procedure(
        name=bar,
        templates=[TemplateDataType(T)],
        args=[],
        body=[],
    )
    result = collect_identifiers(proc)
    assert result == {T, bar}
