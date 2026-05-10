"""Collect symbol identifiers from an AST node."""

__all__ = ["collect_identifiers"]

from fhy_core import (
    AnalysisVisitablePass,
    DataType,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    register_pass,
)
from fhy_core import collect_identifiers as collect_core_identifiers

from fhy_lang.ast.node import (
    Argument,
    DeclarationStatement,
    FunctionExpression,
    IdentifierExpression,
    Module,
    Node,
    Operation,
    Procedure,
    QualifiedType,
)
from fhy_lang.ast.shape import narrow_shape
from fhy_lang.types import TupleType


def _collect_identifiers_from_numerical_type(
    numerical_type: NumericalType,
) -> set[Identifier]:
    identifiers: set[Identifier] = set()
    for dimension in narrow_shape(numerical_type.shape):
        identifiers.update(collect_core_identifiers(dimension))
    return identifiers


def _collect_identifiers_from_index_type(
    index_type: IndexType,
) -> set[Identifier]:
    identifiers: set[Identifier] = set()
    identifiers.update(collect_core_identifiers(index_type.lower_bound))
    identifiers.update(collect_core_identifiers(index_type.upper_bound))
    identifiers.update(collect_core_identifiers(index_type.stride))
    return identifiers


def _collect_identifiers_from_type(node_type: Type) -> set[Identifier]:
    if isinstance(node_type, NumericalType):
        return _collect_identifiers_from_numerical_type(node_type)
    elif isinstance(node_type, IndexType):
        return _collect_identifiers_from_index_type(node_type)
    elif isinstance(node_type, TupleType):
        identifiers: set[Identifier] = set()
        for inner_type in node_type.types:
            identifiers.update(_collect_identifiers_from_type(inner_type))
        return identifiers
    else:
        raise NotImplementedError(
            f"Collecting identifiers from type {node_type} is not supported."
        )


def _collect_identifiers_from_data_type(data_type: DataType) -> set[Identifier]:
    if isinstance(data_type, TemplateDataType):
        return {data_type.data_type}
    elif isinstance(data_type, PrimitiveDataType):
        return set()
    else:
        raise NotImplementedError(
            f"Collecting identifiers from data type {data_type} is not supported."
        )


@register_pass(
    "fhy_ast_identifier_collector",
    "Collects all identifiers in the FhY AST for any given node.",
)
class _IdentifierCollector(AnalysisVisitablePass[Node]):
    """Collect all identifiers in the AST for any given node."""

    _identifiers: set[Identifier]

    def __init__(self) -> None:
        super().__init__()
        self._identifiers = set()

    @property
    def identifiers(self) -> frozenset[Identifier]:
        return frozenset(self._identifiers)

    def visit_module(self, module: Module) -> None:
        self._identifiers.add(module.name)

    def visit_procedure(self, procedure: Procedure) -> None:
        self._identifiers.add(procedure.name)
        for template in procedure.templates:
            self._identifiers.update(_collect_identifiers_from_data_type(template))

    def visit_operation(self, operation: Operation) -> None:
        self._identifiers.add(operation.name)
        for template in operation.templates:
            self._identifiers.update(_collect_identifiers_from_data_type(template))

    def visit_argument(self, argument: Argument) -> None:
        self._identifiers.add(argument.name)

    def visit_declaration_statement(
        self, declaration_statement: DeclarationStatement
    ) -> None:
        self._identifiers.add(declaration_statement.variable_name)

    def visit_function_expression(
        self, function_expression: FunctionExpression
    ) -> None:
        if isinstance(function_expression.function, IdentifierExpression):
            self._identifiers.add(function_expression.function.identifier)
        for template in function_expression.template_types:
            self._identifiers.update(_collect_identifiers_from_data_type(template))

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> None:
        self._identifiers.add(identifier_expression.identifier)

    def visit_qualified_type(self, qualified_type: QualifiedType) -> None:
        self._identifiers.update(
            _collect_identifiers_from_type(qualified_type.base_type)
        )


def collect_identifiers(node: Node) -> frozenset[Identifier]:
    """Return the set of identifier objects referenced by the given AST node."""
    collector = _IdentifierCollector()
    collector(node)
    return collector.identifiers
