"""Transformer pattern for the FhY AST nodes."""

__all__ = ["Transformer"]

from collections.abc import Callable, Sequence
from functools import singledispatchmethod
from typing import TypeVar, cast

from fhy_core import (
    DataType,
    Identifier,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    Type,
    TypeQualifier,
    VisitablePass,
)

from .node import (
    Argument,
    ArrayAccessExpression,
    BinaryExpression,
    ComplexLiteral,
    DeclarationStatement,
    Expression,
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
    Statement,
    TernaryExpression,
    TupleAccessExpression,
    TupleExpression,
    UnaryExpression,
)

Statements = Statement | list[Statement]
_T = TypeVar("_T", bound=Node)


# TODO: do not create new objects when not necessary;
#       test this behavior too!


class Transformer(VisitablePass[Node, Node]):
    """AST node transformer."""

    def get_noop_output(self, ir: Node) -> Node:
        return ir

    def visit_sequence(
        self,
        nodes: Sequence[_T],
        visit_fn: Callable[[_T], _T] | Callable[[_T], _T | Sequence[_T]],
        is_length_same: bool = True,
    ) -> tuple[_T, ...]:
        """Visit a list of nodes or structures.

        Args:
            nodes: Nodes to visit.
            visit_fn: Function to visit each node.
            is_length_same: Whether the length of the transformed nodes or
                nodes should be the same as the input nodes.

        Returns:
            Transformed nodes or structures.

        """
        new_nodes: list[_T] = []
        for node in nodes:
            new_node = visit_fn(node)
            if isinstance(new_node, Sequence):
                if is_length_same:
                    raise RuntimeError(
                        "visit_fn returned a sequence, but is_length_same is True."
                    )
                else:
                    new_nodes.extend(new_node)
            else:
                new_nodes.append(new_node)
        return tuple(new_nodes)

    def visit_module(self, node: Module) -> Module:
        """Transform a module node.

        Args:
            node: Module node to transform.

        """
        new_name = self.visit_identifier(node.name)
        new_statements = self.visit_sequence(
            cast(Sequence[Statement], node.statements),
            self.visit_statement,
            is_length_same=False,
        )

        return Module(
            name=new_name, statements=new_statements, provenance=node.provenance
        )

    def visit_statement(self, node: Statement) -> Statements:
        """Transform a statement node.

        Args:
            node: Statement node to transform.

        """
        if isinstance(node, Import):
            return self.visit_import(node)
        elif isinstance(node, Operation):
            return self.visit_operation(node)
        elif isinstance(node, Procedure):
            return self.visit_procedure(node)
        elif isinstance(node, DeclarationStatement):
            return self.visit_declaration_statement(node)
        elif isinstance(node, ExpressionStatement):
            return self.visit_expression_statement(node)
        elif isinstance(node, SelectionStatement):
            return self.visit_selection_statement(node)
        elif isinstance(node, ForAllStatement):
            return self.visit_for_all_statement(node)
        elif isinstance(node, ReturnStatement):
            return self.visit_return_statement(node)
        else:
            raise NotImplementedError(f'Statement "{type(node)}" is not supported.')

    def visit_import(self, node: Import) -> Import:
        """Transform an import node.

        Args:
            node: Import node to transform.

        """
        return node

    def visit_operation(self, node: Operation) -> Operation:
        """Transform an operation node.

        Args:
            node: Operation node to transform.

        """
        new_name = self.visit_identifier(node.name)
        new_templates = tuple(
            self.visit_template_data_type(template) for template in node.templates
        )
        new_args = self.visit_sequence(node.args, self.visit_argument)
        new_return_type: QualifiedType = self.visit_qualified_type(node.return_type)
        new_body = self.visit_sequence(
            cast(Sequence[Statement], node.body),
            self.visit_statement,
            is_length_same=False,
        )

        return Operation(
            name=new_name,
            templates=new_templates,
            args=new_args,
            return_type=new_return_type,
            body=new_body,
            provenance=node.provenance,
        )

    def visit_procedure(self, node: Procedure) -> Procedure:
        """Transform a Procedure node.

        Args:
            node: Procedure node to transform.

        """
        new_name = self.visit_identifier(node.name)
        new_templates = tuple(
            self.visit_template_data_type(template) for template in node.templates
        )
        new_args = self.visit_sequence(node.args, self.visit_argument)
        new_body = self.visit_sequence(
            node.body, self.visit_statement, is_length_same=False
        )

        return Procedure(
            name=new_name,
            templates=new_templates,
            args=new_args,
            body=new_body,
            provenance=node.provenance,
        )

    def visit_argument(self, node: Argument) -> Argument:
        """Transform an argument node.

        Args:
            node: Argument node to transform.

        """
        return Argument(
            name=self.visit_identifier(node.name),
            qualified_type=self.visit_qualified_type(node.qualified_type),
            provenance=node.provenance,
        )

    def visit_declaration_statement(self, node: DeclarationStatement) -> Statements:
        """Transform a declaration statement node.

        Args:
            node: Declaration statement node to transform.

        """
        new_expression = (
            self.visit_expression(node.expression)
            if node.expression is not None
            else None
        )
        return DeclarationStatement(
            variable_name=node.variable_name,
            variable_type=self.visit_qualified_type(node.variable_type),
            expression=new_expression,
            provenance=node.provenance,
        )

    def visit_expression_statement(self, node: ExpressionStatement) -> Statements:
        new_left = self.visit_expression(node.left) if node.left is not None else None
        return ExpressionStatement(
            left=new_left,
            right=self.visit_expression(node.right),
            provenance=node.provenance,
        )

    def visit_selection_statement(self, node: SelectionStatement) -> Statements:
        """Transform a selection statement node.

        Args:
            node: Selection statement node to transform.

        """
        return SelectionStatement(
            condition=self.visit_expression(node.condition),
            true_body=self.visit_sequence(
                node.true_body, self.visit_statement, is_length_same=False
            ),
            false_body=self.visit_sequence(
                node.false_body, self.visit_statement, is_length_same=False
            ),
            provenance=node.provenance,
        )

    def visit_for_all_statement(self, node: ForAllStatement) -> Statements:
        """Transform an iteration statement node.

        Args:
            node: For-all statement node to transform.

        """
        return ForAllStatement(
            index=self.visit_expression(node.index),
            body=self.visit_sequence(
                node.body, self.visit_statement, is_length_same=False
            ),
            provenance=node.provenance,
        )

    def visit_return_statement(self, node: ReturnStatement) -> Statements:
        """Transform a return statement node.

        Args:
            node: Return statement node to transform.

        """
        return ReturnStatement(
            expression=self.visit_expression(node.expression),
            provenance=node.provenance,
        )

    def visit_expression(self, node: Expression) -> Expression:  # noqa: C901
        """Transform an expression node.

        Args:
            node: Expression node to transform.

        """
        if isinstance(node, UnaryExpression):
            return self.visit_unary_expression(node)
        elif isinstance(node, BinaryExpression):
            return self.visit_binary_expression(node)
        elif isinstance(node, TernaryExpression):
            return self.visit_ternary_expression(node)
        elif isinstance(node, FunctionExpression):
            return self.visit_function_expression(node)
        elif isinstance(node, ArrayAccessExpression):
            return self.visit_array_access_expression(node)
        elif isinstance(node, TupleExpression):
            return self.visit_tuple_expression(node)
        elif isinstance(node, TupleAccessExpression):
            return self.visit_tuple_access_expression(node)
        elif isinstance(node, IdentifierExpression):
            return self.visit_identifier_expression(node)
        elif isinstance(node, IntLiteral):
            return self.visit_int_literal(node)
        elif isinstance(node, FloatLiteral):
            return self.visit_float_literal(node)
        elif isinstance(node, ComplexLiteral):
            return self.visit_complex_literal(node)
        else:
            raise NotImplementedError(f'Expression "{type(node)}" is not supported.')

    def visit_unary_expression(self, node: UnaryExpression) -> UnaryExpression:
        return UnaryExpression(
            operation=node.operation,
            expression=self.visit_expression(node.expression),
            provenance=node.provenance,
        )

    def visit_binary_expression(self, node: BinaryExpression) -> Expression:
        """Transform a binary expression node.

        Args:
            node: Binary expression node to transform.

        """
        return BinaryExpression(
            operation=node.operation,
            left=self.visit_expression(node.left),
            right=self.visit_expression(node.right),
            provenance=node.provenance,
        )

    def visit_ternary_expression(self, node: TernaryExpression) -> Expression:
        """Transform a ternary expression node.

        Args:
            node: Ternary expression node to transform.

        """
        return TernaryExpression(
            condition=self.visit_expression(node.condition),
            true=self.visit_expression(node.true),
            false=self.visit_expression(node.false),
            provenance=node.provenance,
        )

    def visit_function_expression(self, node: FunctionExpression) -> Expression:
        """Transform a function expression node.

        Args:
            node: Function expression node to transform.

        """
        return FunctionExpression(
            function=self.visit_expression(node.function),
            template_types=tuple(
                self.visit_data_type(template) for template in node.template_types
            ),
            indices=self.visit_sequence(
                cast(Sequence[Expression], node.indices), self.visit_expression
            ),
            args=self.visit_sequence(node.args, self.visit_expression),
            provenance=node.provenance,
        )

    def visit_array_access_expression(self, node: ArrayAccessExpression) -> Expression:
        """Transform an array access expression node.

        Args:
            node: Array access expression node to transform.

        """
        return ArrayAccessExpression(
            array_expression=self.visit_expression(node.array_expression),
            indices=self.visit_sequence(
                cast(Sequence[Expression], node.indices), self.visit_expression
            ),
            provenance=node.provenance,
        )

    def visit_tuple_expression(self, node: TupleExpression) -> Expression:
        """Transform a tuple expression node.

        Args:
            node: Tuple expression node to transform.

        """
        return TupleExpression(
            expressions=self.visit_sequence(
                cast(Sequence[Expression], node.expressions), self.visit_expression
            ),
            provenance=node.provenance,
        )

    def visit_tuple_access_expression(self, node: TupleAccessExpression) -> Expression:
        """Transform a tuple access expression node.

        Args:
            node: Tuple access expression node to transform.

        """
        return TupleAccessExpression(
            tuple_expression=self.visit_expression(node.tuple_expression),
            element_index=self.visit_int_literal(node.element_index),
            provenance=node.provenance,
        )

    def visit_identifier_expression(self, node: IdentifierExpression) -> Expression:
        """Transform an identifier expression node.

        Args:
            node: Identifier expression node to transform.

        """
        return IdentifierExpression(
            identifier=self.visit_identifier(node.identifier),
            provenance=node.provenance,
        )

    def visit_int_literal(self, node: IntLiteral) -> IntLiteral:
        """Transform an int literal node.

        Args:
            node: Int literal node to transform.

        """
        return node

    def visit_float_literal(self, node: FloatLiteral) -> FloatLiteral:
        """Transform a float literal node.

        Args:
            node: Float literal node to transform.

        """
        return node

    def visit_complex_literal(self, node: ComplexLiteral) -> ComplexLiteral:
        """Transform a complex literal node.

        Args:
            node: Complex literal node to transform.

        """
        return node

    def visit_qualified_type(self, node: QualifiedType) -> QualifiedType:
        """Transform a qualified type node.

        Args:
            node: Qualified type node to transform.

        """
        return QualifiedType(
            base_type=self.visit_type(node.base_type),
            type_qualifier=self.visit_type_qualifier(node.type_qualifier),
            provenance=node.provenance,
        )

    @singledispatchmethod
    def visit_type(self, node: Type) -> Type:
        """Transform a type.

        Args:
            node: Type to transform.

        """
        raise NotImplementedError(f'Type "{type(node)}" is not supported.')

    @visit_type.register(NumericalType)
    def _(self, numerical_type: NumericalType) -> NumericalType:
        return numerical_type

    @visit_type.register(TupleType)
    def _(self, tuple_type: TupleType) -> TupleType:
        return tuple_type

    @visit_type.register(IndexType)
    def _(self, index_type: IndexType) -> IndexType:
        return index_type

    @singledispatchmethod
    def visit_data_type(self, data_type: DataType) -> DataType:
        """Transform a data type.

        Args:
            data_type: Data type to transform.

        """
        raise NotImplementedError(f'Data type "{type(data_type)}" is not supported.')

    @visit_data_type.register(PrimitiveDataType)
    def _(self, primitive_data_type: PrimitiveDataType) -> PrimitiveDataType:
        return primitive_data_type

    @visit_data_type.register(TemplateDataType)
    def _(self, template_data_type: TemplateDataType) -> TemplateDataType:
        return self.visit_template_data_type(template_data_type)

    def visit_template_data_type(
        self, template_data_type: TemplateDataType
    ) -> TemplateDataType:
        """Transform a template data type node.

        Args:
            template_data_type: Template data type node to transform.

        """
        return template_data_type

    def visit_identifier(self, identifier: Identifier) -> Identifier:
        """Transform an identifier.

        Args:
            identifier: Identifier to transform.

        """
        return identifier

    def visit_type_qualifier(self, type_qualifier: TypeQualifier) -> TypeQualifier:
        """Transform a type qualifier.

        Args:
            type_qualifier: Type qualifier to transform.

        """
        return type_qualifier
