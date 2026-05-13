"""Tests conversion of FhY source code from CST to AST."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from fhy_core import (
    CoreDataType,
    DataType,
    FileProvenance,
    Identifier,
    IndexType,
    NumericalType,
    Position,
    PrimitiveDataType,
    Provenance,
    Span,
    TemplateDataType,
    Type,
    TypeQualifier,
)
from fhy_core import (
    Expression as CoreExpression,
)
from fhy_core import (
    IdentifierExpression as CoreIdentifierExpression,
)
from fhy_core import (
    LiteralExpression as CoreLiteralExpression,
)

from fhy_lang import FhYSyntaxError, TupleType
from fhy_lang.ast import node as ast_node
from fhy_lang.ast.passes import collect_identifiers
from fhy_lang.ast.pprint import pformat_ast
from fhy_lang.converter.from_fhy_source import from_fhy_source
from fhy_lang.converter.from_parse_tree import _get_source_info

from ..utils import assert_name, assert_sequence_type, assert_type

_ConstructAst = Callable[[str], ast_node.Module]


def _is_expressions_exactly_equal(
    expr1: ast_node.Expression, expr2: ast_node.Expression
) -> bool:
    return expr1.is_structurally_equivalent(expr2)


def _assert_expressions_exactly_equal(
    expr1: ast_node.Expression, expr2: ast_node.Expression, what_it_is: str
) -> None:
    assert _is_expressions_exactly_equal(expr1, expr2), (
        f"Expected {what_it_is} expressions to be exactly equal "
        f"(expected: {pformat_ast(expr1, show_id=True)}, "
        f"actual: {pformat_ast(expr2, show_id=True)})"
    )


def _is_core_expressions_exactly_equal(
    expression1: CoreExpression, expression2: CoreExpression
) -> bool:
    return expression1.is_structurally_equivalent(expression2)


def _create_identifier_map(node: ast_node.Node) -> dict[str, Identifier]:
    identifiers = collect_identifiers(node)
    ret = {}
    for identifier in identifiers:
        if identifier.name_hint not in ret:
            ret[identifier.name_hint] = identifier
        else:
            raise ValueError(f"Duplicate identifier: {identifier.name_hint}")
    return ret


def _assert_is_expected_module(
    node: ast_node.Node, expected_num_statements: int
) -> None:
    assert_type(node, ast_node.Module, "AST node")
    assert_sequence_type(node.statements, ast_node.Statement, "module statements")
    assert len(node.statements) == expected_num_statements


def _assert_is_expected_import(node: ast_node.Node, expected_import: str) -> None:
    assert_type(node, ast_node.Import, "AST node")
    assert_type(node.name, Identifier, "imported name")
    assert_name(node.name, expected_import, what_it_is="imported name")


def _assert_is_expected_function_like(
    node: ast_node.Node,
    expected_cls: type[ast_node.Procedure] | type[ast_node.Operation],
    expected_name: Identifier,
    expected_num_templates: int,
    expected_num_args: int,
    expected_num_statements: int,
) -> None:
    label = expected_cls.__name__.lower()
    assert_type(node, expected_cls, "AST node")
    assert_type(node.name, Identifier, f"{label} name")
    assert_name(node.name, expected_name, what_it_is=f"{label} name")
    assert_sequence_type(node.templates, TemplateDataType, f"{label} templates")
    assert len(node.templates) == expected_num_templates
    assert_sequence_type(node.args, ast_node.Argument, f"{label} arguments")
    assert len(node.args) == expected_num_args
    assert_sequence_type(node.body, ast_node.Statement, f"{label} statements")
    assert len(node.body) == expected_num_statements


def _assert_is_expected_procedure(
    node: ast_node.Node,
    expected_name: Identifier,
    expected_num_templates: int,
    expected_num_args: int,
    expected_num_statements: int,
) -> None:
    _assert_is_expected_function_like(
        node,
        ast_node.Procedure,
        expected_name,
        expected_num_templates,
        expected_num_args,
        expected_num_statements,
    )


def _assert_is_expected_operation(
    node: ast_node.Node,
    expected_name: Identifier,
    expected_num_templates: int,
    expected_num_args: int,
    expected_num_statements: int,
) -> None:
    _assert_is_expected_function_like(
        node,
        ast_node.Operation,
        expected_name,
        expected_num_templates,
        expected_num_args,
        expected_num_statements,
    )


def _assert_is_expected_qualified_type(
    node: ast_node.Node,
    expected_type_qualifier: TypeQualifier,
    expected_base_type_cls: type[Type],
) -> None:
    assert_type(node, ast_node.QualifiedType, "qualified type")
    assert node.type_qualifier == expected_type_qualifier
    assert_type(node.base_type, expected_base_type_cls, "qualified type base type")


def _assert_is_expected_argument(
    node: ast_node.Node,
    expected_name: Identifier,
) -> None:
    assert_type(node, ast_node.Argument, "argument")
    assert_type(node.name, Identifier, "argument name")
    assert_name(node.name, expected_name, what_it_is="argument name")


def _assert_is_expected_shape(
    shape: list[ast_node.Expression], expected_shape: list[ast_node.Expression]
) -> None:
    assert_type(shape, list, "shape")
    assert_sequence_type(shape, CoreExpression, "shape")
    assert len(shape) == len(expected_shape)
    for i, shape_component in enumerate(shape):
        assert _is_core_expressions_exactly_equal(shape_component, expected_shape[i]), (
            f"Expected shape component {i} to be equal "
            f"(expected: {expected_shape[i]}, actual: {shape_component})"
        )


def _assert_is_expected_numerical_type(
    numerical_type: NumericalType,
    expected_core_data_type: CoreDataType,
    expected_shape: list[CoreExpression],
) -> None:
    assert_type(numerical_type, NumericalType, "numerical type")
    assert_type(numerical_type.data_type, DataType, "numerical type data type")
    assert numerical_type.data_type.core_data_type == expected_core_data_type
    assert_sequence_type(numerical_type.shape, CoreExpression, "numerical type shape")
    _assert_is_expected_shape(numerical_type.shape, expected_shape)


def _assert_is_expected_index_type(
    index_type: IndexType,
    expected_low: CoreExpression,
    expected_high: CoreExpression,
    expected_stride: CoreExpression,
) -> None:
    assert_type(index_type, IndexType, "index type")
    assert_type(index_type.lower_bound, CoreExpression, "index type lower bound")
    assert _is_core_expressions_exactly_equal(index_type.lower_bound, expected_low), (
        "Expected lower bound to be equal "
        + f"(expected: {expected_low}, actual: {index_type.lower_bound})"
    )
    assert_type(index_type.upper_bound, CoreExpression, "index type upper bound")
    assert _is_core_expressions_exactly_equal(index_type.upper_bound, expected_high), (
        "Expected upper bound to be equal "
        + f"(expected: {expected_high}, actual: {index_type.upper_bound})"
    )
    assert_type(index_type.stride, CoreExpression, "index type stride")
    assert _is_core_expressions_exactly_equal(index_type.stride, expected_stride), (
        "Expected stride to be equal "
        + f"(expected: {expected_stride}, actual: {index_type.stride})"
    )


def _assert_is_expected_declaration_statement(
    node: ast_node.Node,
    expected_variable_name: Identifier,
    expected_expression: ast_node.Expression | None,
) -> None:
    assert_type(node, ast_node.DeclarationStatement, "declaration statement")
    assert_type(node.variable_name, Identifier, "variable name")
    assert_name(node.variable_name, expected_variable_name, what_it_is="variable name")
    assert_type(node.variable_type, ast_node.QualifiedType, "variable type")
    if node.expression is not None:
        assert_type(node.expression, ast_node.Expression, "expression")
    if expected_expression is not None:
        _assert_expressions_exactly_equal(
            node.expression, expected_expression, "declaration statement expression"
        )


def _assert_is_expected_expression_statement(
    node: ast_node.Node,
    expected_left_expression: ast_node.Expression | None,
    expected_right_expression: ast_node.Expression,
) -> None:
    assert_type(node, ast_node.ExpressionStatement, "expression statement")
    if expected_left_expression is not None:
        assert_type(node.left, ast_node.Expression, "left expression")
        _assert_expressions_exactly_equal(
            node.left, expected_left_expression, "left expression"
        )
    assert_type(node.right, ast_node.Expression, "right expression")
    _assert_expressions_exactly_equal(
        node.right, expected_right_expression, "right expression"
    )


def _assert_is_expected_return_statement(
    node: ast_node.Node, expected_expression: ast_node.Expression
) -> None:
    assert_type(node, ast_node.ReturnStatement, "return statement")
    assert_type(node.expression, ast_node.Expression, "expression")
    assert _is_expressions_exactly_equal(node.expression, expected_expression), (
        "Expected expression to be equal "
        + f"(expected: {expected_expression}, actual: {node.expression})"
    )


# ====
# CORE
# ====
def test_empty_file(construct_ast: _ConstructAst) -> None:
    """Test that an empty file is converted correctly."""
    source: str = ""
    ast = construct_ast(source)
    _assert_is_expected_module(ast, 0)


# =========
# FUNCTIONS
# =========
@pytest.mark.parametrize(
    ["source"],
    [
        ("proc foo(){}",),  # only proc
        ("proc foo<>(){}",),  # proc with empty template types
        ("proc foo[](){}",),  # proc with empty index types
        ("proc foo<>[](){}",),  # proc with both empty template and index types
    ],
)
def test_empty_procedure(construct_ast: _ConstructAst, source: str) -> None:
    """Test that an empty procedure is converted correctly."""
    ast = construct_ast(source)
    _assert_is_expected_module(ast, 1)
    procedure = ast.statements[0]
    _assert_is_expected_procedure(procedure, "foo", 0, 0, 0)


@pytest.mark.parametrize(
    ["name"],
    [
        ("x",),
        ("arg",),
        ("arg1",),
        ("arg_1",),
        # Check Identity Names similar to Keywords
        ("importer",),
        ("from_there",),
        ("astype",),
        ("tuples",),
        ("indexed",),
        ("proctor",),
        ("operator",),
        ("natives",),
        ("reduction",),
        ("if_true",),
        ("else_if",),
        ("return_value",),
    ],
)
def test_empty_procedure_with_scalar_argument(
    construct_ast: _ConstructAst, name: str
) -> None:
    """Test an empty procedure with a single scalar argument and argument names."""
    source: str = f"proc foo(input int32 {name}){{}}"
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    procedure = ast.statements[0]
    _assert_is_expected_procedure(procedure, identifier_map["foo"], 0, 1, 0)
    arg = procedure.args[0]
    _assert_is_expected_argument(arg, name)
    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(
        arg_qualified_type, TypeQualifier.INPUT, NumericalType
    )
    arg_base_type = arg_qualified_type.base_type
    _assert_is_expected_numerical_type(arg_base_type, CoreDataType.INT32, [])


@pytest.mark.parametrize(
    ["type_qualifier", "expected_type_qualifier"],
    [
        ("input", TypeQualifier.INPUT),
        ("output", TypeQualifier.OUTPUT),
        ("temp", TypeQualifier.TEMP),
        ("param", TypeQualifier.PARAM),
        ("state", TypeQualifier.STATE),
    ],
)
def test_empty_procedure_with_scalar_argument_qualifiers(
    construct_ast: _ConstructAst,
    type_qualifier: str,
    expected_type_qualifier: TypeQualifier,
) -> None:
    """Test an empty procedure with a single scalar argument with varying
    type qualifiers.
    """
    source: str = f"proc foo({type_qualifier} int32 x){{}}"
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    procedure = ast.statements[0]
    _assert_is_expected_procedure(procedure, identifier_map["foo"], 0, 1, 0)
    arg = procedure.args[0]
    _assert_is_expected_argument(arg, identifier_map["x"])
    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(
        arg_qualified_type, expected_type_qualifier, NumericalType
    )
    arg_base_type = arg_qualified_type.base_type
    _assert_is_expected_numerical_type(arg_base_type, CoreDataType.INT32, [])


@pytest.mark.parametrize(
    ["data_type", "expected_core_data_type"],
    [
        ("uint8", CoreDataType.UINT8),
        ("uint16", CoreDataType.UINT16),
        ("uint32", CoreDataType.UINT32),
        ("int8", CoreDataType.INT8),
        ("int16", CoreDataType.INT16),
        ("int32", CoreDataType.INT32),
        ("int64", CoreDataType.INT64),
        ("float16", CoreDataType.FLOAT16),
        ("float32", CoreDataType.FLOAT32),
        ("float64", CoreDataType.FLOAT64),
        ("complex64", CoreDataType.COMPLEX64),
        ("complex128", CoreDataType.COMPLEX128),
    ],
)
def test_empty_procedure_with_scalar_argument_data_types(
    construct_ast: _ConstructAst,
    data_type: str,
    expected_core_data_type: CoreDataType,
) -> None:
    """Test an empty procedure with a single scalar argument with varying
    data types.
    """
    source: str = f"proc foo(input {data_type} x){{}}"
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    procedure = ast.statements[0]
    _assert_is_expected_procedure(procedure, identifier_map["foo"], 0, 1, 0)
    arg = procedure.args[0]
    _assert_is_expected_argument(arg, identifier_map["x"])
    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(
        arg_qualified_type, TypeQualifier.INPUT, NumericalType
    )
    arg_base_type = arg_qualified_type.base_type
    _assert_is_expected_numerical_type(arg_base_type, expected_core_data_type, [])


def test_empty_procedure_with_array_argument(construct_ast: _ConstructAst) -> None:
    """Test an empty procedure containing an array argument."""
    source: str = "proc foo(input int32[m, n] x){}"
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    procedure = ast.statements[0]
    _assert_is_expected_procedure(procedure, identifier_map["foo"], 0, 1, 0)
    arg = procedure.args[0]
    _assert_is_expected_argument(arg, identifier_map["x"])
    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(
        arg_qualified_type, TypeQualifier.INPUT, NumericalType
    )
    arg_type_shape = arg_qualified_type.base_type.shape
    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_shape(
        arg_type_shape,
        [
            CoreIdentifierExpression(identifier_map["m"]),
            CoreIdentifierExpression(identifier_map["n"]),
        ],
    )


@pytest.mark.parametrize(
    ["source"],
    [
        ("op bar() -> output int32 {}",),  # only op
        ("op bar<>() -> output int32 {}",),  # op with empty template types
        ("op bar[]() -> output int32 {}",),  # op with empty index types
        ("op bar<>[]() -> output int32 {}",),  # op with both empty template
        # and index types
    ],
)
def test_empty_operation(construct_ast: _ConstructAst, source: str) -> None:
    """Test that an empty operation is converted correctly."""
    ast = construct_ast(source)
    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    operation: ast_node.Operation = ast.statements[0]
    _assert_is_expected_operation(operation, identifier_map["bar"], 0, 0, 0)


def test_empty_operation_return_type(construct_ast: _ConstructAst) -> None:
    """Test that an empty operation with a return type is converted correctly."""
    source: str = "op foo(input int32[n, m] x) -> output int32[n, m] {}"
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    operation: ast_node.Operation = ast.statements[0]
    _assert_is_expected_operation(operation, identifier_map["foo"], 0, 1, 0)
    arg = operation.args[0]
    _assert_is_expected_argument(arg, identifier_map["x"])
    arg_qualified_type = arg.qualified_type
    _assert_is_expected_qualified_type(
        arg_qualified_type, TypeQualifier.INPUT, NumericalType
    )
    arg_base_type: Type = arg_qualified_type.base_type
    _assert_is_expected_numerical_type(
        arg_base_type,
        CoreDataType.INT32,
        [
            CoreIdentifierExpression(identifier_map["n"]),
            CoreIdentifierExpression(identifier_map["m"]),
        ],
    )
    return_type = operation.return_type
    _assert_is_expected_qualified_type(return_type, TypeQualifier.OUTPUT, NumericalType)
    return_type_shape = return_type.base_type.shape
    _assert_is_expected_shape(
        return_type_shape,
        [
            CoreIdentifierExpression(identifier_map["n"]),
            CoreIdentifierExpression(identifier_map["m"]),
        ],
    )


@pytest.mark.parametrize(
    ["templates"],
    [(["T"],), (["T", "K"],), (["V", "Ex", "F"],)],
)
def test_operation_template_types(
    construct_ast: _ConstructAst, templates: list[str]
) -> None:
    """Test that an empty operation with template types is returned correctly."""
    source: str = f"op foo<{', '.join(templates)}>(input int32 x) \
-> output int32 {{}}"
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    operation: ast_node.Operation = ast.statements[0]
    _assert_is_expected_operation(
        operation, identifier_map["foo"], len(templates), 1, 0
    )
    assert len(operation.templates) == len(templates)
    for j, k in zip(operation.templates, templates):
        assert_type(j, TemplateDataType, "template type")
        assert_name(j.data_type, identifier_map[k], what_it_is="template type")


def test_operation_template_type_body(construct_ast: _ConstructAst) -> None:
    """Test that an template type identifiers are equivalent."""
    source: str = "op foo<T>(input T[n, m] x) -> output int32[n, m] {temp T[n, m] A;}"
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    operation: ast_node.Operation = ast.statements[0]
    _assert_is_expected_operation(operation, identifier_map["foo"], 1, 1, 1)
    template = operation.templates[0]
    assert_type(template, TemplateDataType, "template type")
    arg_base_type = operation.args[0].qualified_type.base_type
    assert_type(arg_base_type, NumericalType, "numerical type")
    assert_type(arg_base_type.data_type, TemplateDataType, "template type")
    assert_name(
        arg_base_type.data_type.data_type,
        template.data_type,
        what_it_is="template type",
    )
    statement: ast_node.Statement = operation.body[0]
    _assert_is_expected_declaration_statement(statement, identifier_map["A"], None)
    numerical_type = statement.variable_type.base_type
    assert_type(numerical_type, NumericalType, "numerical type")
    assert_name(
        numerical_type.data_type.data_type,
        template.data_type,
        what_it_is="template type",
    )


def test_operation_template_type_call(construct_ast: _ConstructAst) -> None:
    """Test that a template type can be instantiated and used in a call."""
    source: str = """
    op foo<T>(input T[N1, M1] a) -> output T[N1, M1] {
        temp T[N1, M1] b;
        return a;
    }

    op bar<P>() -> output P {
        temp int32[N, M] c;
        temp int32[N, M] d = foo<P>(c);
    }

    proc baz() {
        temp int32 z = bar<int32>();
    }
"""
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 3)
    operation: ast_node.Operation = ast.statements[1]
    _assert_is_expected_operation(operation, identifier_map["bar"], 1, 0, 2)
    statement = operation.body[1]
    _assert_is_expected_declaration_statement(
        statement,
        identifier_map["d"],
        ast_node.FunctionExpression(
            function=ast_node.IdentifierExpression(identifier=identifier_map["foo"]),
            template_types=[TemplateDataType(data_type=identifier_map["P"])],
            args=[ast_node.IdentifierExpression(identifier=identifier_map["c"])],
        ),
    )
    procedure: ast_node.Procedure = ast.statements[2]
    _assert_is_expected_procedure(procedure, identifier_map["baz"], 0, 0, 1)
    statement = procedure.body[0]
    _assert_is_expected_declaration_statement(
        statement,
        identifier_map["z"],
        ast_node.FunctionExpression(
            function=ast_node.IdentifierExpression(identifier=identifier_map["bar"]),
            template_types=[PrimitiveDataType(core_data_type=CoreDataType.INT32)],
            args=[],
        ),
    )


# # ==========
# # STATEMENTS
# # ==========
@pytest.mark.xfail(reason="Import statements are not supported.")
def test_absolute_import(construct_ast: _ConstructAst) -> None:
    """Test absolute import statement is converted correctly."""
    source: str = "import foo.bar;"
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_import(statement, identifier_map["foo.bar"])


def test_declaration_statement_without_assignment(construct_ast: _ConstructAst) -> None:
    """Test converting a single declaration statement without assignment."""
    source: str = "temp int32 i;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(statement, identifier_map["i"], None)
    qualified = statement.variable_type
    _assert_is_expected_qualified_type(qualified, TypeQualifier.TEMP, NumericalType)
    _assert_is_expected_shape(qualified.base_type.shape, [])


def test_declaration_statement_with_assignment(construct_ast: _ConstructAst) -> None:
    """Test converting a single declaration statement with assignment."""
    source: str = "temp int32 i = 5;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(
        statement,
        identifier_map["i"],
        ast_node.IntLiteral(value=5),
    )


def test_array_declaration_statement(construct_ast: _ConstructAst) -> None:
    """Test converting a single array declaration statement."""
    source: str = "temp int32[A, B] c;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(statement, identifier_map["c"], None)
    array_type = statement.variable_type
    _assert_is_expected_qualified_type(array_type, TypeQualifier.TEMP, NumericalType)
    _assert_is_expected_shape(
        array_type.base_type.shape,
        [
            CoreIdentifierExpression(identifier_map["A"]),
            CoreIdentifierExpression(identifier_map["B"]),
        ],
    )


def test_index_variable_declaration_statement(construct_ast: _ConstructAst) -> None:
    """Test converting a single index variable declaration statement."""
    source: str = "temp index[1:N] i;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(statement, identifier_map["i"], None)
    index_type = statement.variable_type
    _assert_is_expected_qualified_type(index_type, TypeQualifier.TEMP, IndexType)
    _assert_is_expected_index_type(
        index_type.base_type,
        CoreLiteralExpression(1),
        CoreIdentifierExpression(identifier_map["N"]),
        CoreLiteralExpression(1),
    )


def test_expression_statement_without_assignment(construct_ast: _ConstructAst) -> None:
    """Test converting a simple binary expression statements."""
    source = "5 + 5;"
    ast: ast_node.Module = construct_ast(source)

    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        None,
        ast_node.BinaryExpression(
            operation=ast_node.BinaryOperation.ADDITION,
            left=ast_node.IntLiteral(value=5),
            right=ast_node.IntLiteral(value=5),
        ),
    )


def test_expression_statement_with_assignment(construct_ast: _ConstructAst) -> None:
    """Test converting a simple binary expression statement with assignment."""
    source = "A = 5 + 5;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        ast_node.IdentifierExpression(identifier=identifier_map["A"]),
        ast_node.BinaryExpression(
            operation=ast_node.BinaryOperation.ADDITION,
            left=ast_node.IntLiteral(value=5),
            right=ast_node.IntLiteral(value=5),
        ),
    )


@pytest.mark.xfail(reason="Selection statements are not supported.")
def test_selection_statement(construct_ast: _ConstructAst) -> None:
    """Test conversion of an if (selection) statement."""
    source: str = "if (1) {i = 1;} else {j = 1;}"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    assert_type(statement, ast_node.SelectionStatement, "selection statement")
    assert_type(
        statement.condition, ast_node.IntLiteral, "selection statement condition"
    )
    assert_sequence_type(
        statement.true_body, ast_node.Statement, "selection statement true body"
    )
    assert len(statement.true_body) == 1
    assert_sequence_type(
        statement.false_body, ast_node.Statement, "selection statement false body"
    )
    assert len(statement.false_body) == 1
    _assert_is_expected_expression_statement(
        statement.true_body[0],
        ast_node.IdentifierExpression(identifier=identifier_map["i"]),
        ast_node.IntLiteral(value=1),
    )
    _assert_is_expected_expression_statement(
        statement.false_body[0],
        ast_node.IdentifierExpression(identifier=identifier_map["j"]),
        ast_node.IntLiteral(value=1),
    )


def test_for_all_statement(construct_ast: _ConstructAst) -> None:
    """Test conversion of a forall statement."""
    source: str = "forall (i) {}"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    assert_type(statement, ast_node.ForAllStatement, "forall statement")
    assert_type(statement.index, ast_node.Expression, "forall statement index")
    _assert_expressions_exactly_equal(
        statement.index,
        ast_node.IdentifierExpression(identifier=identifier_map["i"]),
        "forall statement index",
    )
    assert_sequence_type(statement.body, ast_node.Statement, "forall statement body")
    assert len(statement.body) == 0


def test_return_statement(construct_ast: _ConstructAst) -> None:
    """Test a conversion of a return statement."""
    source: str = "return i;"  # Semantically Incorrect.
    ast: ast_node.Module = construct_ast(source)

    identifer_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_return_statement(
        statement,
        ast_node.IdentifierExpression(identifier=identifer_map["i"]),
    )


# ===========
# EXPRESSIONS
# ===========
@pytest.mark.parametrize(["operator"], [(op,) for op in ast_node.UnaryOperation])
def test_unary_expression(
    construct_ast: _ConstructAst, operator: ast_node.UnaryOperation
) -> None:
    """Test conversion of a unary expression with correct operator."""
    source: str = f"temp int32 i = {operator.value}5;"
    ast: ast_node.Module = construct_ast(source)

    identifer_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(
        statement,
        identifer_map["i"],
        ast_node.UnaryExpression(
            operation=operator,
            expression=ast_node.IntLiteral(value=5),
        ),
    )


@pytest.mark.parametrize(["operator"], [(op,) for op in ast_node.BinaryOperation])
def test_binary_expressions(
    construct_ast: _ConstructAst, operator: ast_node.BinaryOperation
) -> None:
    """Test conversion of binary expressions with correct operators."""
    source: str = f"temp float32 i = 5 {operator.value} 6;"  # Semantically Incorrect
    ast: ast_node.Module = construct_ast(source)

    identifer_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(
        statement,
        identifer_map["i"],
        ast_node.BinaryExpression(
            operation=operator,
            left=ast_node.IntLiteral(value=5),
            right=ast_node.IntLiteral(value=6),
        ),
    )


def test_ternary_expressions(construct_ast: _ConstructAst) -> None:
    """Test converting a ternary expression."""
    source: str = "temp float32 i = 5 < 6 ? 7 : 8;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(
        statement,
        identifier_map["i"],
        ast_node.TernaryExpression(
            condition=ast_node.BinaryExpression(
                operation=ast_node.BinaryOperation.LESS_THAN,
                left=ast_node.IntLiteral(value=5),
                right=ast_node.IntLiteral(value=6),
            ),
            true=ast_node.IntLiteral(value=7),
            false=ast_node.IntLiteral(value=8),
        ),
    )


@pytest.mark.parametrize(["name"], [("A",), ("A1",), ("A_",)])
def test_tuple_access_expression(construct_ast: _ConstructAst, name: str) -> None:
    """Test conversion of a tuple access expression."""
    source: str = f"x = {name}.1;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]

    _assert_is_expected_expression_statement(
        statement,
        ast_node.IdentifierExpression(identifier=identifier_map["x"]),
        ast_node.TupleAccessExpression(
            tuple_expression=ast_node.IdentifierExpression(
                identifier=identifier_map[name]
            ),
            element_index=ast_node.IntLiteral(value=1),
        ),
    )


def test_tuple_access_function_expression(construct_ast: _ConstructAst) -> None:
    """Test conversion of a tuple access expression returned from an operation."""
    source: str = "x = f().1;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        ast_node.IdentifierExpression(identifier=identifier_map["x"]),
        ast_node.TupleAccessExpression(
            tuple_expression=ast_node.FunctionExpression(
                function=ast_node.IdentifierExpression(identifier=identifier_map["f"]),
            ),
            element_index=ast_node.IntLiteral(value=1),
        ),
    )


@pytest.mark.parametrize(
    ["source", "nargs", "name"],
    [
        ("temp int32 i = foo();", 0, "foo"),  # only function call
        ("temp int32 i = foo(A);", 1, "foo"),  # only function call
        ("temp int32 i = module.method();", 0, "module.method"),
        ("temp int32 i = module.method(A);", 1, "module.method"),
        ("temp int32 i = foo[]();", 0, "foo"),  # with index
        ("temp int32 i = foo[](A);", 1, "foo"),  # with index
        ("temp int32 i = module.method[]();", 0, "module.method"),
        ("temp int32 i = module.method[](A);", 1, "module.method"),
        ("temp int32 i = foo<>();", 0, "foo"),  # with template types
        ("temp int32 i = foo<>(A);", 1, "foo"),  # with template types
        ("temp int32 i = module.method<>();", 0, "module.method"),
        ("temp int32 i = module.method<>(A);", 1, "module.method"),
        ("temp int32 i = foo<>[]();", 0, "foo"),  # both template types and index
        ("temp int32 i = foo<>[](A);", 1, "foo"),  # both template types and index
        ("temp int32 i = module.method<>[]();", 0, "module.method"),
        ("temp int32 i = module.method<>[](A);", 1, "module.method"),
    ],
)
def test_function_expression(
    construct_ast: _ConstructAst, source: str, nargs: int, name: str
) -> None:
    """Test conversion of a function call expression with a declaration statement."""
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(
        statement,
        identifier_map["i"],
        ast_node.FunctionExpression(
            function=ast_node.IdentifierExpression(identifier=identifier_map[name]),
            args=[
                ast_node.IdentifierExpression(identifier=identifier_map["A"])
                for _ in range(nargs)
            ],
        ),
    )


@pytest.mark.parametrize(
    ["source"],
    [
        ("foo();",),  # only function call
        ("foo<>();",),  # with template types
        ("foo[]();",),  # with index
        ("foo<>[]();",),  # both template types and index
    ],
)
def test_function_expression_as_expression_statement(
    construct_ast: _ConstructAst, source: str
) -> None:
    """Test conversion of a function call expression as an expression statement."""
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        None,
        ast_node.FunctionExpression(
            function=ast_node.IdentifierExpression(identifier=identifier_map["foo"]),
        ),
    )


def test_tensor_access_expression(construct_ast: _ConstructAst) -> None:
    """Test conversion of a tensor access expression."""
    source: str = "A[i] = 1;"  # Semantically Invalid
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        ast_node.ArrayAccessExpression(
            array_expression=ast_node.IdentifierExpression(
                identifier=identifier_map["A"]
            ),
            indices=[ast_node.IdentifierExpression(identifier=identifier_map["i"])],
        ),
        ast_node.IntLiteral(value=1),
    )


def test_tuple_expression(construct_ast: _ConstructAst) -> None:
    """Test conversion of a tuple expression."""
    source: str = "b = (a,);"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        ast_node.IdentifierExpression(identifier=identifier_map["b"]),
        ast_node.TupleExpression(
            expressions=[ast_node.IdentifierExpression(identifier=identifier_map["a"])],
        ),
    )


# =====
# TYPES
# =====
@pytest.mark.parametrize(
    ["source"],
    [
        ("output tuple[int32[m, n], int32] i;",),
        ("output tuple[int32[m, n], int32,] i;",),
    ],
)
def test_tuple_type(construct_ast: _ConstructAst, source: str) -> None:
    """Test conversion of a tuple type."""
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(statement, identifier_map["i"], None)
    _assert_is_expected_qualified_type(
        statement.variable_type, TypeQualifier.OUTPUT, TupleType
    )
    tuple_type = statement.variable_type.base_type
    assert len(tuple_type.types) == 2
    t1, t2 = tuple_type.types
    _assert_is_expected_numerical_type(
        t1,
        CoreDataType.INT32,
        [
            CoreIdentifierExpression(identifier_map["m"]),
            CoreIdentifierExpression(identifier_map["n"]),
        ],
    )
    _assert_is_expected_numerical_type(t2, CoreDataType.INT32, [])


@pytest.mark.parametrize(
    ["source", "value"],
    [
        ("1;", 1),
        ("0b0101;", 5),
        ("0B01;", 1),
        ("0x1;", 1),
        ("0XFF;", 255),
        ("0o1;", 1),
        ("0O7;", 7),
    ],
)
def test_int_literal(construct_ast: _ConstructAst, source: str, value: int) -> None:
    """Test conversion of int literals in different formats."""
    ast: ast_node.Module = construct_ast(source)

    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        None,
        ast_node.IntLiteral(value=value),
    )


@pytest.mark.parametrize(
    ["source", "value"],
    [
        ("1.0;", 1.0),
        (".2;", 0.2),
        (" 1.;", 1.0),
        (" 1e2;", 100.0),
        ("1.2e3;", 1200.0),
    ],
)
def test_float_literal(construct_ast: _ConstructAst, source: str, value: float) -> None:
    """Test conversion of float literals in different formats."""
    ast: ast_node.Module = construct_ast(source)

    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        None,
        ast_node.FloatLiteral(value=value),
    )


@pytest.mark.parametrize(
    ["source", "value"],
    [
        ("1.0j;", 1.0j),
        ("1j;", 1j),
        ("1e10j;", 1e10j),
        ("0.2j;", 0.2j),
        (".2j;", 0.2j),
    ],
)
def test_complex_literal(
    construct_ast: _ConstructAst, source: str, value: complex
) -> None:
    """Test conversion of complex literals in different formats."""
    ast: ast_node.Module = construct_ast(source)

    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        None,
        ast_node.ComplexLiteral(value=value),
    )


# =============
# MISCELLANEOUS
# =============
def test_line_comment(construct_ast: _ConstructAst) -> None:
    """Test that comments are skipped during conversion, creating an empty module."""
    source: str = "# this is a comment!"
    ast = construct_ast(source)
    _assert_is_expected_module(ast, 0)


# ===============
# EXPECTED ERRORS
# ===============
def test_syntax_error_no_argument_name(construct_ast: _ConstructAst) -> None:
    """Test that FhYSyntaxError is raised when a function argument has no name."""
    source: str = "op foo(input int32[m,n]) -> output int32 {}"
    with pytest.raises(FhYSyntaxError, match="Function Argument Name not Provided"):
        construct_ast(source)


def test_syntax_error_no_operation_return_type(construct_ast: _ConstructAst) -> None:
    """Test that FhYSyntaxError is raised when an operation has no return type."""
    source: str = "op func(input int32[m,n] A) {}"
    with pytest.raises(
        FhYSyntaxError, match="Operation Functions Require Return Types"
    ):
        construct_ast(source)


def test_invalid_function_keyword(construct_ast: _ConstructAst) -> None:
    """Test that FhYSyntaxError is raised when a function uses an invalid keyword."""
    source: str = "def foo(input int32[m,n] A) -> output int32[m,n] {}"
    with pytest.raises(FhYSyntaxError):
        construct_ast(source)


@pytest.mark.parametrize(
    ["source"],
    [
        ("lorem ipsum dolor sit amet;",),  # With Semicolon
        ("lorem ipsum dolor sit amet",),  # No Semicolon
    ],
)
def test_gibberish(construct_ast: _ConstructAst, source: str) -> None:
    """Test gibberish (unrecognized text per the grammar) raises FhYSyntaxError."""
    with pytest.raises(FhYSyntaxError):
        construct_ast(source)


# ============================
# INDEX TYPE STRIDE
# ============================
def test_index_variable_with_stride(construct_ast: _ConstructAst) -> None:
    """Test conversion of an index type with an explicit stride expression."""
    source: str = "temp index[1:N:2] i;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(statement, identifier_map["i"], None)
    index_type = statement.variable_type
    _assert_is_expected_qualified_type(index_type, TypeQualifier.TEMP, IndexType)
    _assert_is_expected_index_type(
        index_type.base_type,
        CoreLiteralExpression(1),
        CoreIdentifierExpression(identifier_map["N"]),
        CoreLiteralExpression(2),
    )


# ==================================
# UNSUPPORTED FEATURES
# ==================================
@pytest.mark.parametrize(
    ["source"],
    [
        ("proc foo[i]() {}",),
        ("proc foo[i, j]() {}",),
        ("op foo[i]() -> output int32 {}",),
        ("op foo[i, j]() -> output int32 {}",),
    ],
)
def test_function_with_indices_raises_not_implemented(
    construct_ast: _ConstructAst, source: str
) -> None:
    """Test that a function with non-empty index list raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="indices"):
        construct_ast(source)


# ====================
# PROVENANCE
# ====================
def test_provenance_unknown_when_constructed_with_unknown_provenance(
    construct_ast: _ConstructAst,
) -> None:
    """Test nodes carry Provenance.unknown() for an unknown parse provenance."""
    source: str = "temp int32 i = 5;"
    ast: ast_node.Module = construct_ast(source)

    assert ast.provenance == Provenance.unknown()
    statement = ast.statements[0]
    assert statement.provenance == Provenance.unknown()


def test_get_source_info_walks_parent_chain_to_find_tokens() -> None:
    """Test _get_source_info walks ancestors until it finds a token-bearing context."""
    grandparent = SimpleNamespace(
        start=SimpleNamespace(line=1, column=0),
        stop=SimpleNamespace(line=1, column=10),
        parentCtx=None,
    )
    parent = SimpleNamespace(start=None, stop=None, parentCtx=grandparent)
    child = SimpleNamespace(start=None, stop=None, parentCtx=parent)

    provenance = _get_source_info(child, FileProvenance("test.fhy"))

    assert isinstance(provenance, FileProvenance)
    assert provenance.span.start_position.line == 1
    assert provenance.span.start_position.column == 1
    assert provenance.span.end_position.line == 1
    assert provenance.span.end_position.column == 11


def test_get_source_info_returns_unknown_when_no_ancestor_has_tokens() -> None:
    """Test _get_source_info returns unknown when no ctx in the chain has tokens."""
    parent = SimpleNamespace(start=None, stop=None, parentCtx=None)
    child = SimpleNamespace(start=None, stop=None, parentCtx=parent)

    provenance = _get_source_info(child, FileProvenance("test.fhy"))

    assert provenance == Provenance.unknown()


def test_get_source_info_returns_unknown_for_non_file_provenance() -> None:
    """Test _get_source_info returns unknown when parse provenance is non-file."""
    ctx = SimpleNamespace(
        start=SimpleNamespace(line=1, column=0),
        stop=SimpleNamespace(line=1, column=10),
        parentCtx=None,
    )

    provenance = _get_source_info(ctx, Provenance.unknown())

    assert provenance == Provenance.unknown()


# ===================================
# BUILT-IN TYPE IDENTIFIER IDENTITY
# ===================================
def test_built_in_type_identifier_identity_is_shared_across_parses(
    construct_ast: _ConstructAst,
) -> None:
    """Test built-in type names resolve to the same Identifier object across parses."""
    ast1: ast_node.Module = construct_ast("int32;")
    ast2: ast_node.Module = construct_ast("int32;")

    expr1 = ast1.statements[0].right
    expr2 = ast2.statements[0].right

    assert expr1.identifier is expr2.identifier


# ============================
# UNSUPPORTED FEATURE PROVENANCE
# ============================


def _file_provenance(file_path: str = "/tmp/fake.fhy") -> FileProvenance:
    """Helper: a FileProvenance with a known file path and a wide span."""
    return FileProvenance(
        Path(file_path),
        Span(
            start_position=Position(line=1, column=1),
            end_position=Position(line=99, column=80),
        ),
    )


def test_import_statement_raises_not_implemented_with_provenance() -> None:
    """Test import statements raise NotImplementedError tagged with source location."""
    with pytest.raises(NotImplementedError, match=r"fake\.fhy:\d+:\d+"):
        from_fhy_source("import foo.bar;", provenance=_file_provenance())


def test_selection_statement_raises_not_implemented_with_provenance() -> None:
    """Test selection statements raise NotImplementedError tagged with location."""
    source: str = "proc main(output int32 b) { if (1) { b = 1; } else { b = 0; } }"
    with pytest.raises(NotImplementedError, match=r"fake\.fhy:\d+:\d+"):
        from_fhy_source(source, provenance=_file_provenance())


def test_function_declaration_raises_not_implemented_with_provenance() -> None:
    """Test bodyless function declarations raise NotImplementedError with location."""
    with pytest.raises(NotImplementedError, match=r"fake\.fhy:\d+:\d+"):
        from_fhy_source("proc main();", provenance=_file_provenance())


def test_function_indices_raise_not_implemented_with_provenance() -> None:
    """Test function indices raise NotImplementedError tagged with source location."""
    source: str = "op f<>[i, j]() -> output int32 { return 0; }"
    with pytest.raises(NotImplementedError, match=r"fake\.fhy:\d+:\d+"):
        from_fhy_source(source, provenance=_file_provenance())


def test_dtype_template_parameters_raise_not_implemented_with_provenance() -> None:
    """Test custom dtype template params raise NotImplementedError with location."""
    with pytest.raises(NotImplementedError, match=r"fake\.fhy:\d+:\d+"):
        from_fhy_source("temp myparam<5> x;", provenance=_file_provenance())
