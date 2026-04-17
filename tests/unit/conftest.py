"""Pytest Unit Test Fixtures and Utilities.

Fixtures present in this file are innately available to all modules present within this
directory. This is not true of Subdirectories, which will need it's own conftest.py
file.

"""

from collections.abc import Callable, Generator
from typing import TypeVar

import pytest
from fhy.lang.ast import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    BinaryOperation,
    ComplexLiteral,
    DeclarationStatement,
    ExpressionStatement,
    FloatLiteral,
    ForAllStatement,
    FunctionExpression,
    IdentifierExpression,
    Import,
    IntLiteral,
    Module,
    Node,
    Operation,
    Procedure,
    QualifiedType,
    ReturnStatement,
    SelectionStatement,
    TernaryExpression,
    TupleAccessExpression,
    TupleExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy.lang.converter.from_fhy_source import from_fhy_source as fhy_source
from fhy_core import (
    CoreDataType,
    Identifier,
    IndexType,
    NumericalType,
    Position,
    PrimitiveDataType,
    Provenance,
    Span,
    TemplateDataType,
    TupleType,
    TypeQualifier,
    get_logger,
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

logger = get_logger(__name__)
logger.setLevel(10)

TLiteral = TypeVar("TLiteral", IntLiteral, FloatLiteral, ComplexLiteral)
T = TypeVar("T")
fixture_node_names: list[str] = []


def add_fixture_node(f: T) -> T:
    """Simple wrapper to collect statically constructed ast node fixtures.

    This is used for ease of testing where we can instead grab the `fixture_node_names`
    variable from this module.

    """
    fixture_node_names.append(f.__name__)

    return f


@pytest.fixture
def construct_ast() -> Callable[[str], Node]:
    """Construct an abstract syntax tree (AST) from a raw text file source."""

    def _inner(source: str) -> Node:
        return fhy_source(source, provenance=Provenance.unknown(), logger=logger)

    return _inner


# TODO: Just get rid of all this... these tests are horrible.


@pytest.fixture
def construct_id() -> Generator[Callable[[str], tuple[dict, Provenance]], None, None]:
    def inner(name: str) -> tuple[dict, Provenance]:
        """Build an Identifier from a Name Hint."""
        _id = Identifier(name)
        obj = dict(cls_name="Identifier", attributes=dict(name_hint=name, _id=_id._id))

        return obj, _id

    yield inner


@pytest.fixture
def core_literal_expression() -> (
    Callable[[int | float | complex], tuple[dict, TLiteral]]
):
    def _build(value: int | float | complex):  # noqa: PYI041
        _literal: TLiteral
        result = dict(value=value)
        _literal = CoreLiteralExpression(value=value)

        name = _literal.__class__.__qualname__
        obj = dict(cls_name=name, attributes=dict(value=result))
        return obj, _literal

    return _build


@add_fixture_node
@pytest.fixture
def provenance() -> tuple[dict, Provenance]:
    a, b, c, d, e = 1, 10, 1, 10, "test"
    obj = dict(
        cls_name="Provenance",
        attributes=dict(
            span=dict(
                cls_name="Span",
                attributes=dict(
                    start_line=a,
                    end_line=b,
                    start_column=c,
                    end_column=d,
                ),
            ),
        ),
    )
    sp = Provenance(
        span=Span(
            file_path="test.fhy",
            start_position=Position(line=a, column=c),
            end_position=Position(line=b, column=d),
        )
    )

    return obj, sp


@pytest.fixture
def literals(provenance) -> Callable[[int | float | complex], tuple[dict, TLiteral]]:
    def _build(value: int | float | complex):  # noqa: PYI041
        span_obj, span_cls = provenance
        _literal: TLiteral
        if isinstance(value, complex):
            result = dict(real=value.real, imag=value.imag)
            _literal = ComplexLiteral(provenance=span_cls, value=value)
        elif isinstance(value, float):
            result = value
            _literal = FloatLiteral(provenance=span_cls, value=value)
        elif isinstance(value, int):
            result = value
            _literal = IntLiteral(provenance=span_cls, value=value)
        else:
            raise TypeError(f"Invalid Value Type: {value}")

        name = _literal.__class__.__qualname__
        obj = dict(cls_name=name, attributes=dict(provenance=span_obj, value=result))
        return obj, _literal

    return _build


@pytest.fixture
def build_numerical_type() -> (
    Generator[
        Callable[
            [CoreDataType, list[dict], list[CoreExpression]],
            tuple[dict, NumericalType],
        ],
        None,
        None,
    ]
):
    def inner(
        core: CoreDataType,
        shape_objs: list[dict],
        shape_cls: list[CoreExpression],
    ) -> tuple[dict, NumericalType]:
        """Builds a Numerical Type obj and node."""
        obj = dict(
            cls_name="NumericalType",
            attributes=dict(
                data_type=dict(
                    cls_name="PrimitiveDataType",
                    attributes=dict(core_data_type=core.value),
                ),
                shape=shape_objs,
            ),
        )

        numerical = NumericalType(
            data_type=PrimitiveDataType(core_data_type=core), shape=shape_cls
        )

        return obj, numerical

    yield inner


@add_fixture_node
@pytest.fixture
def index_type(core_literal_expression, construct_id) -> tuple[dict, IndexType]:
    text = "index[1:k:1]"
    one_obj, one_cls = core_literal_expression(1)
    upper_obj, upper_cls = construct_id("k")

    obj = dict(
        cls_name="IndexType",
        attributes=dict(
            lower_bound=one_obj,
            upper_bound=upper_obj,
            stride=one_obj,
        ),
    )

    index = IndexType(
        lower_bound=one_cls,
        upper_bound=CoreIdentifierExpression(upper_cls),
        stride=one_cls,
    )

    return obj, index


@add_fixture_node
@pytest.fixture
def tuple_type(construct_id, build_numerical_type) -> tuple[dict, TupleType]:
    text = "tuple ( int32[m], )"
    shape_1_obj, shape_1_id = construct_id("m")
    num_obj, numerical = build_numerical_type(
        CoreDataType.INT32, [shape_1_obj], [CoreIdentifierExpression(shape_1_id)]
    )

    obj = dict(cls_name="TupleType", attributes=dict(types=[num_obj]))

    tup = TupleType(types=[numerical])

    return obj, tup


@add_fixture_node
@pytest.fixture
def qualified(
    provenance, construct_id, build_numerical_type
) -> tuple[dict, QualifiedType]:
    text = "input int32[m, n]"
    span_obj, span_cls = provenance
    shape_1_obj, shape_1_id = construct_id("m")
    shape_2_obj, shape_2_id = construct_id("n")
    num_obj, num_cls = build_numerical_type(
        CoreDataType.INT32,
        [shape_1_obj, shape_2_obj],
        [CoreIdentifierExpression(shape_1_id), CoreIdentifierExpression(shape_2_id)],
    )

    obj = dict(
        cls_name="QualifiedType",
        attributes=dict(
            provenance=span_obj,
            base_type=num_obj,
            type_qualifier="input",
        ),
    )

    qualified_type = QualifiedType(
        provenance=span_cls,
        base_type=num_cls,
        type_qualifier=TypeQualifier.INPUT,
    )

    return obj, qualified_type


@add_fixture_node
@pytest.fixture
def arg1(provenance, qualified, construct_id) -> tuple[dict, Argument]:
    text: str = "input int32[m, n] rupaul"
    span_obj, span_cls = provenance
    qtype_obj, qtype_cls = qualified
    arg_id_obj, arg_id = construct_id("rupaul")

    obj = dict(
        cls_name="Argument",
        attributes=dict(
            provenance=span_obj,
            name=arg_id_obj,
            qualified_type=qtype_obj,
        ),
    )

    arg = Argument(
        provenance=span_cls,
        name=arg_id,
        qualified_type=qtype_cls,
    )

    return obj, arg


# EXPRESSIONS
@add_fixture_node
@pytest.fixture
def unary(provenance, literals) -> tuple[dict, UnaryExpression]:
    text: str = "-(5)"
    span_obj, span_cls = provenance
    negative = UnaryOperation.NEGATION
    literal_obj, literal_cls = literals(5)

    obj = dict(
        cls_name="UnaryExpression",
        attributes=dict(
            provenance=span_obj,
            operation=negative.value,
            expression=literal_obj,
        ),
    )

    _unary = UnaryExpression(
        provenance=span_cls, operation=negative, expression=literal_cls
    )

    return obj, _unary


@add_fixture_node
@pytest.fixture
def binary(provenance, literals) -> tuple[dict, BinaryExpression]:
    text: str = "(5 + 10.0)"
    span_obj, span_cls = provenance
    addition = BinaryOperation.ADDITION
    lit_obj_left, lit_cls_left = literals(5)
    lit_obj_right, lit_cls_right = literals(10.0)

    obj = dict(
        cls_name="BinaryExpression",
        attributes=dict(
            provenance=span_obj,
            operation=addition.value,
            left=lit_obj_left,
            right=lit_obj_right,
        ),
    )

    bin_express = BinaryExpression(
        provenance=span_cls, operation=addition, left=lit_cls_left, right=lit_cls_right
    )

    return obj, bin_express


@add_fixture_node
@pytest.fixture
def ternary(provenance, literals, binary, unary) -> tuple[dict, TernaryExpression]:
    text: str = "(1 ? (5 + 10.0) : -(5))"
    span_obj, span_cls = provenance
    cond_obj, cond_cls = literals(1)
    binary_obj, binary_cls = binary
    unary_obj, unary_cls = unary

    obj = dict(
        cls_name="TernaryExpression",
        attributes=dict(
            provenance=span_obj,
            condition=cond_obj,
            true=binary_obj,
            false=unary_obj,
        ),
    )

    expression = TernaryExpression(
        provenance=span_cls, condition=cond_cls, true=binary_cls, false=unary_cls
    )

    return obj, expression


@add_fixture_node
@pytest.fixture
def tuple_access(
    provenance, literals, construct_id
) -> tuple[dict, TupleAccessExpression]:
    text: str = "A.1"
    span_obj, span_cls = provenance
    one_obj, one_cls = literals(1)
    name_obj, name_cls = construct_id("A")

    obj = dict(
        cls_name="TupleAccessExpression",
        attributes=dict(
            provenance=span_obj,
            tuple_expression=name_obj,
            element_index=one_obj,
        ),
    )
    access = TupleAccessExpression(
        provenance=span_cls,
        tuple_expression=name_cls,
        element_index=one_cls,
    )

    return obj, access


@add_fixture_node
@pytest.fixture
def function_call(provenance, construct_id) -> tuple[dict, FunctionExpression]:
    text: str = "funky<>[]()"
    span_obj, span_cls = provenance
    name_obj, name_cls = construct_id("funky")

    obj = dict(
        cls_name="FunctionExpression",
        attributes=dict(
            provenance=span_obj,
            function=name_obj,
            template_types=[],
            indices=[],
            args=[],
        ),
    )
    function = FunctionExpression(
        provenance=span_cls, function=name_cls, template_types=[], indices=[], args=[]
    )

    return obj, function


@add_fixture_node
@pytest.fixture
def array_access(provenance, construct_id) -> tuple[dict, ArrayAccessExpression]:
    text: str = "array[j, k]"
    span_obj, span_cls = provenance
    array_id_obj, array_id_cls = construct_id("array")
    index_1_obj, index_1_cls = construct_id("j")
    index_2_obj, index_2_cls = construct_id("k")

    obj = dict(
        cls_name="ArrayAccessExpression",
        attributes=dict(
            provenance=span_obj,
            array_expression=array_id_obj,
            indices=[index_1_obj, index_2_obj],
        ),
    )
    array = ArrayAccessExpression(
        provenance=span_cls,
        array_expression=array_id_cls,
        indices=[index_1_cls, index_2_cls],
    )

    return obj, array


@add_fixture_node
@pytest.fixture
def tuple_express(provenance) -> tuple[dict, TupleExpression]:
    text: str = "(  )"
    span_obj, span_cls = provenance

    obj = dict(
        cls_name="TupleExpression",
        attributes=dict(
            provenance=span_obj,
            expressions=[],
        ),
    )
    array = TupleExpression(
        provenance=span_cls,
        expressions=[],
    )

    return obj, array


@add_fixture_node
@pytest.fixture
def id_express(provenance, construct_id) -> tuple[dict, IdentifierExpression]:
    text: str = "name"
    span_obj, span_cls = provenance
    id_obj, id_cls = construct_id("name")

    obj = dict(
        cls_name="IdentifierExpression",
        attributes=dict(
            provenance=span_obj,
            identifier=id_obj,
        ),
    )
    expression = IdentifierExpression(
        provenance=span_cls,
        identifier=id_cls,
    )

    return obj, expression


# STATEMENTS
@add_fixture_node
@pytest.fixture
def declaration(
    provenance, qualified, binary, construct_id
) -> tuple[dict, DeclarationStatement]:
    text: str = "input int32[m, n] bar = (5 + 10.0);"
    span_obj, span_cls = provenance
    qtype_obj, qtype_cls = qualified
    binary_obj, binary_cls = binary
    varname_obj, varname_id = construct_id("bar")

    obj = dict(
        cls_name="DeclarationStatement",
        attributes=dict(
            provenance=span_obj,
            variable_name=varname_obj,
            variable_type=qtype_obj,
            expression=binary_obj,
        ),
    )

    statement = DeclarationStatement(
        provenance=span_cls,
        variable_name=varname_id,
        variable_type=qtype_cls,
        expression=binary_cls,
    )

    return obj, statement


@add_fixture_node
@pytest.fixture
def express_state(provenance, unary, array_access) -> tuple[dict, ExpressionStatement]:
    text: str = "array[j, k] = -(5);"
    span_obj, span_cls = provenance
    array_obj, array_cls = array_access
    unary_obj, unary_cls = unary

    obj = dict(
        cls_name="ExpressionStatement",
        attributes=dict(
            provenance=span_obj,
            left=array_obj,
            right=unary_obj,
        ),
    )

    statement = ExpressionStatement(
        provenance=span_cls,
        left=array_cls,
        right=unary_cls,
    )

    return obj, statement


@add_fixture_node
@pytest.fixture
def iteration_state(provenance, construct_id) -> tuple[dict, ForAllStatement]:
    text: str = "forall (elements) {\n\n}"
    span_obj, span_cls = provenance
    index_obj, index_cls = construct_id("elements")

    obj = dict(
        cls_name="ForAllStatement",
        attributes=dict(provenance=span_obj, index=index_obj, body=[]),
    )

    statement = ForAllStatement(provenance=span_cls, index=index_cls, body=[])

    return obj, statement


@add_fixture_node
@pytest.fixture
def select_state(provenance, binary) -> tuple[dict, SelectionStatement]:
    text: str = "if (5 + 10.0) {\n\n}"
    span_obj, span_cls = provenance
    binary_obj, binary_cls = binary

    obj = dict(
        cls_name="SelectionStatement",
        attributes=dict(
            provenance=span_obj, condition=binary_obj, true_body=[], false_body=[]
        ),
    )

    statement = SelectionStatement(
        provenance=span_cls, condition=binary_cls, true_body=[], false_body=[]
    )

    return obj, statement


@add_fixture_node
@pytest.fixture
def return_state(provenance, unary) -> tuple[dict, ReturnStatement]:
    text: str = "return -(5);"
    span_obj, span_cls = provenance
    unary_obj, unary_cls = unary

    obj = dict(
        cls_name="ReturnStatement",
        attributes=dict(provenance=span_obj, expression=unary_obj),
    )

    statement = ReturnStatement(
        provenance=span_cls,
        expression=unary_cls,
    )

    return obj, statement


@add_fixture_node
@pytest.fixture
def import_node(provenance, construct_id) -> tuple[dict, Import]:
    text: str = "import x.y;"
    span_obj, span_cls = provenance
    id_obj, id_cls = construct_id("x.y")
    obj = dict(cls_name="Import", attributes=dict(provenance=span_obj, name=id_obj))
    import_statement = Import(provenance=span_cls, name=id_cls)

    return obj, import_statement


# FUNCTIONS
@add_fixture_node
@pytest.fixture
def operation(
    provenance, arg1, qualified, express_state, construct_id
) -> tuple[dict, Operation]:
    text: str = (
        "op foobar<>(input int32[m, n] rupaul) -> input int32[m, n]"
        " {\n  array[j, k] = -(5);\n}"
    )
    span_obj, span_cls = provenance
    arg1_obj, arg1_cls = arg1
    qtype_obj, qtype_cls = qualified
    name_id_obj, name_id = construct_id("foobar")
    e_state_obj, e_state_cls = express_state

    obj = dict(
        cls_name="Operation",
        attributes=dict(
            provenance=span_obj,
            name=name_id_obj,
            templates=[],
            args=[arg1_obj],
            body=[e_state_obj],
            return_type=qtype_obj,
        ),
    )

    op = Operation(
        provenance=span_cls,
        name=name_id,
        args=[arg1_cls],
        body=[e_state_cls],
        return_type=qtype_cls,
    )

    return obj, op


@add_fixture_node
@pytest.fixture
def procedure_with_templates(construct_id) -> tuple[dict, Procedure]:
    text: str = "proc mumu<T>() {\n\n}"
    name_id_obj, name_id = construct_id("mumu")
    tobj, tid = construct_id("T")

    obj = dict(
        cls_name="Procedure",
        attributes=dict(
            name=name_id_obj,
            templates=[
                dict(cls_name="TemplateDataType", attributes=dict(data_type=tobj))
            ],
            args=[],
            body=[],
        ),
    )
    proc = Procedure(
        provenance=None,
        name=name_id,
        templates=[TemplateDataType(data_type=tid)],
        args=[],
        body=[],
    )

    return obj, proc


@add_fixture_node
@pytest.fixture
def procedure(provenance, arg1, declaration, construct_id) -> tuple[dict, Procedure]:
    text: str = (
        "proc buzz<>(input int32[m, n] rupaul) "
        "{\n  input int32[m, n] bar = (5 + 10.0);\n}"
    )
    span_obj, span_cls = provenance
    arg1_obj, arg1_cls = arg1
    declare_obj, declare_cls = declaration
    name_id_obj, name_id = construct_id("buzz")

    obj = dict(
        cls_name="Procedure",
        attributes=dict(
            provenance=span_obj,
            name=name_id_obj,
            templates=[],
            args=[arg1_obj],
            body=[declare_obj],
        ),
    )

    proc = Procedure(
        provenance=span_cls,
        name=name_id,
        templates=[],
        args=[arg1_cls],
        body=[declare_cls],
    )

    return obj, proc


# MODULE
@add_fixture_node
@pytest.fixture
def module(provenance, operation, procedure) -> tuple[dict, Module]:
    text: str = (
        "op foobar<>(input int32[m, n] rupaul) -> input int32[m, n]"
        " {\n  array[j, k] = -(5);\n}\n"
        "proc buzz<>(input int32[m, n] rupaul) "
        "{\n  input int32[m, n] bar = (5 + 10.0);\n}"
    )
    span_obj, span_cls = provenance
    op_obj, op_cls = operation
    proc_obj, proc_cls = procedure

    obj = dict(
        cls_name="Module",
        attributes=dict(provenance=span_obj, statements=[op_obj, proc_obj]),
    )

    module = Module(provenance=span_cls, statements=[op_cls, proc_cls])

    return obj, module
