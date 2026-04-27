"""Tests the identifier replacer AST pass."""

from fhy_core import (
    CoreDataType,
    Identifier,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TypeQualifier,
)
from fhy_core import IdentifierExpression as CoreIdentifierExpression

from fhy_lang import replace_identifiers
from fhy_lang.ast import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    DeclarationStatement,
    ExpressionStatement,
    FunctionExpression,
    IdentifierExpression,
    IntLiteral,
    Module,
    Operation,
    Procedure,
    QualifiedType,
)


def _make_id_expr(identifier: Identifier) -> IdentifierExpression:
    return IdentifierExpression(identifier=identifier)


def test_empty_module(empty_module_ast):
    """Test replacing on an empty module returns a new module with the new name."""
    new_module_name = Identifier("new_module_name")
    result = replace_identifiers(
        empty_module_ast, {empty_module_ast.name: new_module_name}
    )
    assert isinstance(result, Module)
    assert result.name == new_module_name
    assert len(result.statements) == 0


def test_empty_map_is_identity(x_plus_y_expression_ast):
    """Test that an empty map leaves identifiers unchanged."""
    right, x, y = x_plus_y_expression_ast
    statement = ExpressionStatement(
        right=right,
    )
    result = replace_identifiers(statement, {})
    assert isinstance(result, ExpressionStatement)
    assert isinstance(result.right, BinaryExpression)
    assert result.right.left.identifier == x
    assert result.right.right.identifier == y


def test_unmapped_identifier_is_preserved(x_plus_y_expression_ast):
    """Test identifiers absent from the map are left alone."""
    right, x, y = x_plus_y_expression_ast
    x_new = Identifier("x_new")
    statement = ExpressionStatement(
        right=right,
    )
    result = replace_identifiers(statement, {x: x_new})
    assert result.right.left.identifier == x_new
    assert result.right.right.identifier == y


def test_procedure_name():
    """Test replacing the name of a procedure."""
    procedure = Procedure(
        name=Identifier("procedure_name"),
        templates=(),
        args=(),
        body=(),
    )
    new_procedure_name = Identifier("new_procedure_name")
    result = replace_identifiers(procedure, {procedure.name: new_procedure_name})
    assert isinstance(result, Procedure)
    assert result.name == new_procedure_name


def test_operation_name():
    """Test replacing the name of an operation."""
    operation = Operation(
        name=Identifier("operation_name"),
        templates=(),
        args=(),
        body=(),
        return_type=QualifiedType(
            base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            type_qualifier=TypeQualifier.OUTPUT,
        ),
    )
    new_operation_name = Identifier("new_operation_name")
    result = replace_identifiers(operation, {operation.name: new_operation_name})
    assert isinstance(result, Operation)
    assert result.name == new_operation_name


def test_arguments():
    """Test replacing identifiers in an argument."""
    n = Identifier("n")
    argument = Argument(
        name=Identifier("argument_name"),
        qualified_type=QualifiedType(
            base_type=NumericalType(
                PrimitiveDataType(CoreDataType.INT32),
                shape=[CoreIdentifierExpression(n)],
            ),
            type_qualifier=TypeQualifier.TEMP,
        ),
    )
    procedure = Procedure(
        name=Identifier("procedure_name"),
        templates=(),
        args=(argument,),
        body=(),
    )
    new_argument_name = Identifier("new_argument_name")
    new_n = Identifier("new_n")
    result = replace_identifiers(
        procedure, {argument.name: new_argument_name, n: new_n}
    )
    assert isinstance(result, Procedure)
    assert result.args[0].name == new_argument_name
    assert result.args[0].qualified_type.base_type.shape[0].identifier == new_n


def test_template_types():
    """Test replacing identifiers in a template type."""
    T = Identifier("T")
    T_new = Identifier("T_new")
    template_type = TemplateDataType(T)
    procedure = Procedure(
        name=Identifier("procedure_name"),
        templates=(template_type,),
        args=(),
        body=(),
    )
    result = replace_identifiers(procedure, {T: T_new})
    assert isinstance(result, Procedure)
    assert result.templates[0].data_type == T_new


def test_declaration_statement_expression():
    """Test replacing identifiers within the initializer expression of a
    declaration statement."""
    x = Identifier("x")
    y = Identifier("y")
    y_new = Identifier("y_new")
    statement = DeclarationStatement(
        variable_name=x,
        variable_type=QualifiedType(
            base_type=NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            type_qualifier=TypeQualifier.TEMP,
        ),
        expression=_make_id_expr(y),
    )
    result = replace_identifiers(statement, {y: y_new})
    assert isinstance(result, DeclarationStatement)
    assert result.variable_name == x
    assert isinstance(result.expression, IdentifierExpression)
    assert result.expression.identifier == y_new


def test_binary_expression_statement(x_plus_y_expression_ast):
    """Test replacing both operands of a binary expression."""
    right, x, y = x_plus_y_expression_ast
    x_new = Identifier("x_new")
    y_new = Identifier("y_new")
    statement = ExpressionStatement(right=right)
    result = replace_identifiers(statement, {x: x_new, y: y_new})
    assert isinstance(result, ExpressionStatement)
    assert isinstance(result.right, BinaryExpression)
    assert result.right.left.identifier == x_new
    assert result.right.right.identifier == y_new
    assert result.right.operation == BinaryOperation.ADDITION


def test_function_expression():
    """Test replacing identifiers in a function expression."""
    foo = Identifier("foo")
    i = Identifier("i")
    x = Identifier("x")
    foo_new = Identifier("foo_new")
    i_new = Identifier("i_new")
    x_new = Identifier("x_new")
    func = FunctionExpression(
        function=_make_id_expr(foo),
        indices=[_make_id_expr(i)],
        template_types=[],
        args=[_make_id_expr(x)],
    )
    result = replace_identifiers(func, {foo: foo_new, i: i_new, x: x_new})
    assert isinstance(result, FunctionExpression)
    assert result.function.identifier == foo_new
    assert result.indices[0].identifier == i_new
    assert result.args[0].identifier == x_new


def test_array_access_expression():
    """Test replacing identifiers in an array access expression."""
    i = Identifier("i")
    arr = Identifier("arr")
    i_new = Identifier("i_new")
    arr_new = Identifier("arr_new")
    arr_access = ArrayAccessExpression(
        array_expression=_make_id_expr(arr),
        indices=[_make_id_expr(i)],
    )
    result = replace_identifiers(arr_access, {i: i_new, arr: arr_new})
    assert isinstance(result, ArrayAccessExpression)
    assert result.array_expression.identifier == arr_new
    assert result.indices[0].identifier == i_new


def test_module_with_nested_replacement():
    """Test replacement propagates into nested module statements."""
    x = Identifier("x")
    y = Identifier("y")
    x_new = Identifier("x_new")
    module = Module(
        statements=(
            ExpressionStatement(
                right=BinaryExpression(
                    left=_make_id_expr(x),
                    right=_make_id_expr(y),
                    operation=BinaryOperation.ADDITION,
                ),
            ),
        ),
    )
    result = replace_identifiers(module, {x: x_new})
    assert isinstance(result, Module)
    assert len(result.statements) == 1
    stmt = result.statements[0]
    assert stmt.right.left.identifier == x_new
    assert stmt.right.right.identifier == y


def test_returns_new_node_when_mapped():
    """Test that an IdentifierExpression is rebuilt with the mapped identifier."""
    x = Identifier("x")
    x_new = Identifier("x_new")
    expr = _make_id_expr(x)
    result = replace_identifiers(expr, {x: x_new})
    assert isinstance(result, IdentifierExpression)
    assert result.identifier == x_new


def test_int_literal_unchanged():
    """Test that literals pass through unchanged."""
    literal = IntLiteral(value=42)
    result = replace_identifiers(literal, {Identifier("x"): Identifier("y")})
    assert isinstance(result, IntLiteral)
    assert result.value == 42
