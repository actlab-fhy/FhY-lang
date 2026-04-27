"""Tests conversion of FhY source code from CST to AST."""

from collections.abc import Sequence

import pytest
from fhy_core import (
    BinaryExpression as CoreBinaryExpression,
)
from fhy_core import (
    CoreDataType,
    DataType,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    Provenance,
    Stack,
    TemplateDataType,
    TupleType,
    Type,
    TypeQualifier,
    VisitablePass,
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
from fhy_core import (
    UnaryExpression as CoreUnaryExpression,
)

from fhy_lang.lang import FhYSyntaxError
from fhy_lang.lang.ast import node as ast_node
from fhy_lang.lang.ast.passes import collect_identifiers
from fhy_lang.lang.ast.pprint import pformat_ast

from ..utils import assert_name, assert_sequence_type, assert_type

# TODO: switch to using structural equivalence

# TODO: make all identifier name equality not in terms of name hint after scope and
#       loading identifiers with table is implemented


# TODO: Use expression only base pass when implemented
class ExpressionExactEqualityGetter(VisitablePass[ast_node.Expression, bool]):
    """Pass to determine if two expressions are exactly equal."""

    _stack: Stack[ast_node.Expression | Sequence[ast_node.Expression]]

    def __init__(self, other: ast_node.Expression) -> None:
        self._stack = Stack[ast_node.Expression]()
        self._stack.push(other)

    def get_noop_output(self, ir: ast_node.Expression) -> bool:
        return True

    def visit_sequence(self, node: Sequence[ast_node.Expression]) -> bool:
        other = self._stack.pop()
        if not isinstance(other, Sequence):
            return False
        if len(node) != len(other):
            return False
        for expression, other_expression in zip(node, other):
            self._stack.push(other_expression)
            if not self.visit(expression):
                return False
        return True

    def visit_unary_expression(self, node: ast_node.UnaryExpression) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.UnaryExpression):
            return False
        else:
            if node.operation != other.operation:
                return False
            self._stack.push(other.expression)
            return self.visit(node.expression)

    def visit_binary_expression(self, node: ast_node.BinaryExpression) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.BinaryExpression):
            return False
        else:
            if node.operation != other.operation:
                return False
            self._stack.push(other.right)
            is_right_equal = self.visit(node.right)
            self._stack.push(other.left)
            is_left_equal = self.visit(node.left)
            return is_left_equal and is_right_equal

    def visit_ternary_expression(self, node: ast_node.TernaryExpression) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.TernaryExpression):
            return False
        else:
            self._stack.push(other.false)
            is_false_equal = self.visit(node.false)
            self._stack.push(other.true)
            is_true_equal = self.visit(node.true)
            self._stack.push(other.condition)
            is_condition_equal = self.visit(node.condition)
            return is_condition_equal and is_true_equal and is_false_equal

    def visit_function_expression(self, node: ast_node.FunctionExpression) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.FunctionExpression):
            return False
        else:
            self._stack.push(other.function)
            is_function_equal = self.visit(node.function)
            self._stack.push(other.indices)
            is_indices_equal = self.visit_sequence(node.indices)
            self._stack.push(other.template_types)
            is_template_types_equal = len(node.template_types) == len(
                other.template_types
            ) and all(
                template_type.is_structurally_equivalent(other_template_type)
                for template_type, other_template_type in zip(
                    node.template_types, other.template_types
                )
            )
            self._stack.push(other.args)
            is_args_equal = self.visit_sequence(node.args)
            return (
                is_function_equal
                and is_indices_equal
                and is_template_types_equal
                and is_args_equal
            )

    def visit_array_access_expression(
        self, node: ast_node.ArrayAccessExpression
    ) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.ArrayAccessExpression):
            return False
        else:
            self._stack.push(other.array_expression)
            is_array_expression_equal = self.visit(node.array_expression)
            self._stack.push(other.indices)
            is_indices_equal = self.visit_sequence(node.indices)
            return is_array_expression_equal and is_indices_equal

    def visit_tuple_expression(self, node: ast_node.TupleExpression) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.TupleExpression):
            return False
        else:
            self._stack.push(other.expressions)
            return self.visit_sequence(node.expressions)

    def visit_tuple_access_expression(
        self, node: ast_node.TupleAccessExpression
    ) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.TupleAccessExpression):
            return False
        else:
            self._stack.push(other.tuple_expression)
            is_tuple_expression_equal = self.visit(node.tuple_expression)
            self._stack.push(other.element_index)
            is_element_index_equal = self.visit(node.element_index)
            return is_tuple_expression_equal and is_element_index_equal

    def visit_identifier_expression(self, node: ast_node.IdentifierExpression) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.IdentifierExpression):
            return False
        else:
            return node.identifier == other.identifier

    def visit_int_literal(self, node: ast_node.IntLiteral) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.IntLiteral):
            return False
        else:
            return node.value == other.value

    def visit_float_literal(self, node: ast_node.FloatLiteral) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.FloatLiteral):
            return False
        else:
            return node.value == other.value

    def visit_complex_literal(self, node: ast_node.ComplexLiteral) -> bool:
        other = self._stack.pop()
        if not isinstance(other, ast_node.ComplexLiteral):
            return False
        else:
            return node.value == other.value


def _is_expressions_exactly_equal(
    expr1: ast_node.Expression, expr2: ast_node.Expression
) -> bool:
    return ExpressionExactEqualityGetter(expr2)(expr1)


def _assert_expressions_exactly_equal(
    expr1: ast_node.Expression, expr2: ast_node.Expression, what_it_is: str
) -> None:
    assert _is_expressions_exactly_equal(expr1, expr2), (
        f"Expected {what_it_is} expressions to be exactly equal \
(expected: {pformat_ast(expr1, show_id=True)}, \
actual: {pformat_ast(expr2, show_id=True)})"
    )


def _is_core_expressions_exactly_equal(
    expression1: CoreExpression, expression2: CoreExpression
) -> None:
    if isinstance(expression1, CoreLiteralExpression) and isinstance(
        expression2, CoreLiteralExpression
    ):
        return expression1.value == expression2.value
    elif isinstance(expression1, CoreIdentifierExpression) and isinstance(
        expression2, CoreIdentifierExpression
    ):
        return expression1.identifier == expression2.identifier
    elif isinstance(expression1, CoreUnaryExpression) and isinstance(
        expression2, CoreUnaryExpression
    ):
        return (
            expression1.operation == expression2.operation
            and _is_core_expressions_exactly_equal(
                expression1.operand, expression2.operand
            )
        )
    elif isinstance(expression1, CoreBinaryExpression) and isinstance(
        expression2, CoreBinaryExpression
    ):
        return (
            expression1.operation == expression2.operation
            and _is_core_expressions_exactly_equal(expression1.left, expression2.left)
            and _is_core_expressions_exactly_equal(expression1.right, expression2.right)
        )
    else:
        return False


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


def _assert_is_expected_procedure(
    node: ast_node.Node,
    expected_name: Identifier,
    expected_num_templates: int,
    expected_num_args: int,
    expected_num_statements: int,
) -> None:
    assert_type(node, ast_node.Procedure, "AST node")
    assert_type(node.name, Identifier, "procedure name")
    assert_name(node.name, expected_name, what_it_is="procedure name")
    assert_sequence_type(node.templates, TemplateDataType, "procedure templates")
    assert len(node.templates) == expected_num_templates
    assert_sequence_type(node.args, ast_node.Argument, "procedure arguments")
    assert len(node.args) == expected_num_args
    assert_sequence_type(node.body, ast_node.Statement, "procedure statements")
    assert len(node.body) == expected_num_statements


def _assert_is_expected_operation(
    node: ast_node.Node,
    expected_name: Identifier,
    expected_num_templates: int,
    expected_num_args: int,
    expected_num_statements: int,
) -> None:
    assert_type(node, ast_node.Operation, "AST node")
    assert_type(node.name, Identifier, "operation name")
    assert_name(node.name, expected_name, what_it_is="operation name")
    assert_sequence_type(node.templates, TemplateDataType, "procedure templates")
    assert len(node.templates) == expected_num_templates
    assert_sequence_type(node.args, ast_node.Argument, "operation arguments")
    assert len(node.args) == expected_num_args
    assert_sequence_type(node.body, ast_node.Statement, "operation statements")
    assert len(node.body) == expected_num_statements


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
        (
            _is_core_expressions_exactly_equal(shape_component, expected_shape[i]),
            (
                f"Expected shape component {i} to be equal "
                + f"(expected: {expected_shape[i]}, actual: {shape_component})"
            ),
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
def test_empty_file(construct_ast):
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
def test_empty_procedure(construct_ast, source: str):
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
def test_empty_procedure_with_scalar_argument(construct_ast, name: str):
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
    construct_ast, type_qualifier: str, expected_type_qualifier: TypeQualifier
):
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
    construct_ast, data_type: str, expected_core_data_type: CoreDataType
):
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


def test_empty_procedure_with_array_argument(construct_ast):
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
def test_empty_operation(construct_ast, source: str):
    """Test that an empty operation is converted correctly."""
    ast = construct_ast(source)
    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    operation: ast_node.Operation = ast.statements[0]
    _assert_is_expected_operation(operation, identifier_map["bar"], 0, 0, 0)


def test_empty_operation_return_type(construct_ast):
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
            ast_node.IdentifierExpression(
                identifier=identifier_map["n"], provenance=Provenance.unknown()
            ),
            ast_node.IdentifierExpression(
                identifier=identifier_map["m"], provenance=Provenance.unknown()
            ),
        ],
    )


@pytest.mark.parametrize(
    ["templates"],
    [(["T"],), (["T", "K"],), (["V", "Ex", "F"],)],
)
def test_operation_template_types(construct_ast, templates: list[str]):
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


def test_operation_template_type_body(construct_ast):
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


def test_operation_template_type_call(construct_ast):
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
            function=ast_node.IdentifierExpression(
                identifier=identifier_map["foo"], provenance=Provenance.unknown()
            ),
            template_types=[TemplateDataType(data_type=identifier_map["P"])],
            args=[
                ast_node.IdentifierExpression(
                    identifier=identifier_map["c"], provenance=Provenance.unknown()
                )
            ],
            provenance=Provenance.unknown(),
        ),
    )
    procedure: ast_node.Procedure = ast.statements[2]
    _assert_is_expected_procedure(procedure, identifier_map["baz"], 0, 0, 1)
    statement = procedure.body[0]
    _assert_is_expected_declaration_statement(
        statement,
        identifier_map["z"],
        ast_node.FunctionExpression(
            function=ast_node.IdentifierExpression(
                identifier=identifier_map["bar"], provenance=Provenance.unknown()
            ),
            template_types=[PrimitiveDataType(core_data_type=CoreDataType.INT32)],
            args=[],
            provenance=Provenance.unknown(),
        ),
    )


# # ==========
# # STATEMENTS
# # ==========
@pytest.mark.xfail(reason="Import statements are not supported.")
def test_absolute_import(construct_ast):
    """Test absolute import statement is converted correctly."""
    source: str = "import foo.bar;"
    ast = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_import(statement, identifier_map["foo.bar"])


def test_declaration_statement_without_assignment(construct_ast):
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


def test_declaration_statement_with_assignment(construct_ast):
    """Test converting a single declaration statement with assignment."""
    source: str = "temp int32 i = 5;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(
        statement,
        identifier_map["i"],
        ast_node.IntLiteral(value=5, provenance=Provenance.unknown()),
    )


def test_array_declaration_statement(construct_ast):
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
            ast_node.IdentifierExpression(
                identifier=identifier_map["A"], provenance=Provenance.unknown()
            ),
            ast_node.IdentifierExpression(
                identifier=identifier_map["B"], provenance=Provenance.unknown()
            ),
        ],
    )


def test_index_variable_declaration_statement(construct_ast):
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


def test_expression_statement_without_assignment(construct_ast):
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
            left=ast_node.IntLiteral(value=5, provenance=Provenance.unknown()),
            right=ast_node.IntLiteral(value=5, provenance=Provenance.unknown()),
            provenance=Provenance.unknown(),
        ),
    )


def test_expression_statement_with_assignment(construct_ast):
    """Test converting a simple binary expression statement with assignment."""
    source = "A = 5 + 5;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        ast_node.IdentifierExpression(
            identifier=identifier_map["A"], provenance=Provenance.unknown()
        ),
        ast_node.BinaryExpression(
            operation=ast_node.BinaryOperation.ADDITION,
            left=ast_node.IntLiteral(value=5, provenance=Provenance.unknown()),
            right=ast_node.IntLiteral(value=5, provenance=Provenance.unknown()),
            provenance=Provenance.unknown(),
        ),
    )


@pytest.mark.xfail(reason="Selection statements are not supported.")
def test_selection_statement(construct_ast):
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
        ast_node.IntLiteral(value=1, provenance=Provenance.unknown()),
    )
    _assert_is_expected_expression_statement(
        statement.false_body[0],
        ast_node.IdentifierExpression(
            identifier=identifier_map["j"], provenance=Provenance.unknown()
        ),
        ast_node.IntLiteral(value=1, provenance=Provenance.unknown()),
        provenance=Provenance.unknown(),
    )


def test_for_all_statement(construct_ast):
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
        ast_node.IdentifierExpression(
            identifier=identifier_map["i"], provenance=Provenance.unknown()
        ),
        "forall statement index",
    )
    assert_sequence_type(statement.body, ast_node.Statement, "forall statement body")
    assert len(statement.body) == 0


def test_return_statement(construct_ast):
    """Test a conversion of a return statement."""
    source: str = "return i;"  # Semantically Incorrect.
    ast: ast_node.Module = construct_ast(source)

    identifer_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_return_statement(
        statement,
        ast_node.IdentifierExpression(
            identifier=identifer_map["i"], provenance=Provenance.unknown()
        ),
    )


# ===========
# EXPRESSIONS
# ===========
@pytest.mark.parametrize(["operator"], [(op,) for op in ast_node.UnaryOperation])
def test_unary_expression(construct_ast, operator: ast_node.UnaryOperation):
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
            expression=ast_node.IntLiteral(value=5, provenance=Provenance.unknown()),
            provenance=Provenance.unknown(),
        ),
    )


@pytest.mark.parametrize(["operator"], [(op,) for op in ast_node.BinaryOperation])
def test_binary_expressions(construct_ast, operator: ast_node.BinaryOperation):
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
            left=ast_node.IntLiteral(value=5, provenance=Provenance.unknown()),
            right=ast_node.IntLiteral(value=6, provenance=Provenance.unknown()),
            provenance=Provenance.unknown(),
        ),
    )


def test_ternary_expressions(construct_ast):
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
                left=ast_node.IntLiteral(value=5, provenance=Provenance.unknown()),
                right=ast_node.IntLiteral(value=6, provenance=Provenance.unknown()),
                provenance=Provenance.unknown(),
            ),
            true=ast_node.IntLiteral(value=7, provenance=Provenance.unknown()),
            false=ast_node.IntLiteral(value=8, provenance=Provenance.unknown()),
            provenance=Provenance.unknown(),
        ),
    )


@pytest.mark.parametrize(["name"], [("A",), ("A1",), ("A_",)])
def test_tuple_access_expression(construct_ast, name: str):
    """Test conversion of a tuple access expression."""
    source: str = f"x = {name}.1;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]

    _assert_is_expected_expression_statement(
        statement,
        ast_node.IdentifierExpression(
            identifier=identifier_map["x"], provenance=Provenance.unknown()
        ),
        ast_node.TupleAccessExpression(
            tuple_expression=ast_node.IdentifierExpression(
                identifier=identifier_map[name], provenance=Provenance.unknown()
            ),
            element_index=ast_node.IntLiteral(value=1, provenance=Provenance.unknown()),
            provenance=Provenance.unknown(),
        ),
    )


def test_tuple_access_function_expression(construct_ast):
    """Test conversion of a tuple access expression returned from an operation."""
    source: str = "x = f().1;"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        ast_node.IdentifierExpression(
            identifier=identifier_map["x"], provenance=Provenance.unknown()
        ),
        ast_node.TupleAccessExpression(
            tuple_expression=ast_node.FunctionExpression(
                function=ast_node.IdentifierExpression(
                    identifier=identifier_map["f"], provenance=Provenance.unknown()
                ),
                provenance=Provenance.unknown(),
            ),
            element_index=ast_node.IntLiteral(value=1, provenance=Provenance.unknown()),
            provenance=Provenance.unknown(),
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
def test_function_expression(construct_ast, source: str, nargs: int, name: str):
    """Test conversion of a function call expression with a declaration statement."""
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_declaration_statement(
        statement,
        identifier_map["i"],
        ast_node.FunctionExpression(
            function=ast_node.IdentifierExpression(
                identifier=identifier_map[name], provenance=Provenance.unknown()
            ),
            args=[
                ast_node.IdentifierExpression(
                    identifier=identifier_map["A"], provenance=Provenance.unknown()
                )
                for _ in range(nargs)
            ],
            provenance=Provenance.unknown(),
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
def test_function_expression_as_expression_statement(construct_ast, source: str):
    """Test conversion of a function call expression as an expression statement."""
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        None,
        ast_node.FunctionExpression(
            function=ast_node.IdentifierExpression(
                identifier=identifier_map["foo"], provenance=Provenance.unknown()
            ),
            provenance=Provenance.unknown(),
        ),
    )


def test_tensor_access_expression(construct_ast):
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
                identifier=identifier_map["A"], provenance=Provenance.unknown()
            ),
            indices=[
                ast_node.IdentifierExpression(
                    identifier=identifier_map["i"], provenance=Provenance.unknown()
                )
            ],
            provenance=Provenance.unknown(),
        ),
        ast_node.IntLiteral(value=1, provenance=Provenance.unknown()),
    )


def test_tuple_expression(construct_ast):
    """Test conversion of a tuple expression."""
    source: str = "b = (a,);"
    ast: ast_node.Module = construct_ast(source)

    identifier_map = _create_identifier_map(ast)
    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        ast_node.IdentifierExpression(
            identifier=identifier_map["b"], provenance=Provenance.unknown()
        ),
        ast_node.TupleExpression(
            expressions=[
                ast_node.IdentifierExpression(
                    identifier=identifier_map["a"], provenance=Provenance.unknown()
                )
            ],
            provenance=Provenance.unknown(),
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
def test_tuple_type(construct_ast, source: str):
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
def test_int_literal(construct_ast, source: str, value: int):
    """Test conversion of int literals in different formats."""
    ast: ast_node.Module = construct_ast(source)

    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        None,
        ast_node.IntLiteral(value=value, provenance=Provenance.unknown()),
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
def test_float_literal(construct_ast, source: str, value: float):
    """Test conversion of float literals in different formats."""
    ast: ast_node.Module = construct_ast(source)

    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        None,
        ast_node.FloatLiteral(value=value, provenance=Provenance.unknown()),
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
def test_complex_literal(construct_ast, source: str, value: complex):
    """Test conversion of complex literals in different formats."""
    ast: ast_node.Module = construct_ast(source)

    _assert_is_expected_module(ast, 1)
    statement = ast.statements[0]
    _assert_is_expected_expression_statement(
        statement,
        None,
        ast_node.ComplexLiteral(value=value, provenance=Provenance.unknown()),
    )


# =============
# MISCELLANEOUS
# =============
def test_line_comment(construct_ast):
    """Test that comments are skipped during conversion, creating an empty module."""
    source: str = "# this is a comment!"
    ast = construct_ast(source)
    _assert_is_expected_module(ast, 0)


# ===============
# EXPECTED ERRORS
# ===============
def test_syntax_error_no_argument_name(construct_ast):
    """Test that FhYSyntaxError is raised when a function argument has no name."""
    source: str = "op foo(input int32[m,n]) -> output int32 {}"
    with pytest.raises(FhYSyntaxError):
        ast = construct_ast(source)


def test_syntax_error_no_procedure_name(construct_ast):
    """Test that FhYSyntaxError is raised when a procedure has no name."""
    source: str = "proc () {}"
    # NOTE: This raises the ANTLR Syntax Error, not from our visitor class.
    #       This means we do not gain coverage in parse tree converter for this case.
    with pytest.raises(FhYSyntaxError):
        ast = construct_ast(source)


def test_syntax_error_no_operation_name(construct_ast):
    """Test that FhYSyntaxError is raised when an operation has no name."""
    source: str = "op (input int32[m,n] A) -> output int32 {}"
    # NOTE: This raises the ANTLR Syntax Error, not from our visitor class.
    #       This means we do not gain coverage in parse tree converter for this case.
    with pytest.raises(FhYSyntaxError):
        ast = construct_ast(source)


def test_syntax_error_no_operation_return_type(construct_ast):
    """Test that FhYSyntaxError is raised when an operation has no return type."""
    source: str = "op func(input int32[m,n] A) {}"
    with pytest.raises(FhYSyntaxError):
        ast = construct_ast(source)


def test_invalid_function_keyword(construct_ast):
    """Test that FhYSyntaxError is raised when a function uses an invalid keyword."""
    source: str = "def foo(input int32[m,n] A) -> output int32[m,n] {}"
    with pytest.raises(FhYSyntaxError):
        ast = construct_ast(source)


@pytest.mark.parametrize(
    ["source"],
    [
        ("lorem ipsum dolor sit amet;",),  # With Semicolon
        ("lorem ipsum dolor sit amet",),  # No Semicolon
    ],
)
def test_gibberish(construct_ast, source: str):
    """Test gibberish (unrecognized text per the grammar) raises FhYSyntaxError."""
    with pytest.raises(FhYSyntaxError):
        ast = construct_ast(source)
