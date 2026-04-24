grammar FhY;

/*
 * Module Rules
 */

module
    : scope
    ;

/*
 * Statement Rules
 */

statement
    : import_statement
    | function_declaration
    | function_definition
    | declaration_statement
    | expression_statement
    | selection_statement
    | iteration_statement
    | return_statement
    ;

scope
    : statement*
    ;

import_statement
    : IMPORT identifier_expression SEMICOLON
    ;

function_declaration
    : function_header SEMICOLON
    ;

function_definition
    : function_header OPEN_BRACE function_body CLOSE_BRACE
    ;

function_header
    : function_type=FUNCTION_KEYWORD name=IDENTIFIER (LESS_THAN function_template_types=identifier_list GREATER_THAN)? (OPEN_BRACKET function_indices=function_args CLOSE_BRACKET)? OPEN_PARENTHESES function_args CLOSE_PARENTHESES (ARROW return_type=qualified_type)?
    ;

identifier_list
    : (IDENTIFIER (COMMA IDENTIFIER)*)?
    ;

function_args
    : (function_arg (COMMA function_arg)*)?
    ;

function_arg
    : qualified_type (name=IDENTIFIER)?
    ;

function_body
    : scope
    ;

declaration_statement
    : qualified_type name=IDENTIFIER (EQUALS_SIGN expression)? SEMICOLON
    ;

expression_statement
    : (primitive_expression EQUALS_SIGN)? expression SEMICOLON
    ;

selection_statement
    : IF OPEN_PARENTHESES expression CLOSE_PARENTHESES OPEN_BRACE scope CLOSE_BRACE (ELSE OPEN_BRACE scope CLOSE_BRACE)?
    ;

iteration_statement
    : FORALL OPEN_PARENTHESES expression CLOSE_PARENTHESES OPEN_BRACE scope CLOSE_BRACE
    ;

return_statement
    : RETURN expression SEMICOLON
    ;

/*
 * Type Rules
 */

qualified_type
    : (type_qualifier=IDENTIFIER)? type
    ;

type
    : tuple_type
    | numerical_type
    | index_type
    ;

tuple_type
    : TUPLE OPEN_BRACKET ((type COMMA) | (type (COMMA type)+ COMMA?))? CLOSE_BRACKET
    ;

numerical_type
    : dtype (OPEN_BRACKET expression_list CLOSE_BRACKET)?
    ;

dtype
    : base_dtype=IDENTIFIER (LESS_THAN expression_list GREATER_THAN)?
    ;

dtype_list
    : (dtype (COMMA dtype)* COMMA?)?
    ;

index_type
    : INDEX OPEN_BRACKET range CLOSE_BRACKET
    ;

range
    : expression COLON expression (COLON expression)?
    ;

/*
 * Expression Rules
 */

expression
    : nested_expression=OPEN_PARENTHESES expression CLOSE_PARENTHESES
    | unary_expression=(SUBTRACTION | BITWISE_NOT | LOGICAL_NOT) expression
    | power_expression=expression POWER expression
    | multiplicative_expression=expression (MULTIPLICATION | DIVISION | FLOORDIV | MODULO) expression
    | additive_expression=expression (ADDITION | SUBTRACTION) expression
    | shift_expression=expression (LEFT_SHIFT | RIGHT_SHIFT)expression
    | relational_expression=expression (LESS_THAN | LESS_THAN_OR_EQUAL | GREATER_THAN | GREATER_THAN_OR_EQUAL) expression
    | equality_expression=expression (EQUAL_TO | NOT_EQUAL_TO) expression
    | and_expression=expression AND expression
    | exclusive_or_expression=expression EXCLUSIVE_OR expression
    | or_expression=expression OR expression
    | logical_and_expression=expression LOGICAL_AND expression
    | logical_or_expression=expression LOGICAL_OR expression
    | ternary_expression=expression QUESTION_MARK expression COLON expression
    | primitive_expression
    ;

expression_list
    : (expression (COMMA expression)*)?
    ;

primitive_expression
    : tuple_access_expression=primitive_expression FLOAT_LITERAL
    | function_expression=primitive_expression (LESS_THAN dtype_list GREATER_THAN)? (OPEN_BRACKET expression_list CLOSE_BRACKET)? OPEN_PARENTHESES expression_list CLOSE_PARENTHESES
    | array_access_expression=primitive_expression OPEN_BRACKET expression_list CLOSE_BRACKET
    | atom
    ;

atom
    : tuple
    | identifier_expression
    | literal
    ;

tuple
    : OPEN_PARENTHESES expression COMMA CLOSE_PARENTHESES
    | OPEN_PARENTHESES expression (COMMA expression)+ COMMA? CLOSE_PARENTHESES
    ;

identifier_expression
    : IDENTIFIER (DOT IDENTIFIER)*
    ;

/*
 * Literal Rules
 */

literal
    : INT_LITERAL
    | FLOAT_LITERAL
    | COMPLEX_LITERAL
    ;

/*
 * Keyword Tokens
 */

IMPORT
    : 'import'
    ;

FROM
    : 'from'
    ;

AS
    : 'as'
    ;

TUPLE
    : 'tuple'
    ;

INDEX
    : 'index'
    ;

FUNCTION_KEYWORD
    : PROCEDURE
    | OPERATION
    | NATIVE
    ;

PROCEDURE
    : 'proc'
    ;

OPERATION
    : 'op'
    ;

REDUCTION
    : 'reduc'
    ;

NATIVE
    : 'native'
    ;

FORALL
    : 'forall'
    ;

IF
    : 'if'
    ;

ELSE
    : 'else'
    ;

RETURN
    : 'return'
    ;

/*
 * Foundational Tokens
 */

OPEN_PARENTHESES
    : '('
    ;

CLOSE_PARENTHESES
    : ')'
    ;

OPEN_BRACKET
    : '['
    ;

CLOSE_BRACKET
    : ']'
    ;

OPEN_BRACE
    : '{'
    ;

CLOSE_BRACE
    : '}'
    ;

ARROW
    : '->'
    ;

EQUALS_SIGN
    : '='
    ;

QUESTION_MARK
    : '?'
    ;

COMMA
    : ','
    ;

DOT
    : '.'
    ;

SEMICOLON
    : ';'
    ;

COLON
    : ':'
    ;

LOGICAL_NOT
    : '!'
    ;

BITWISE_NOT
    : '~'
    ;

POWER
    : '**'
    ;

MULTIPLICATION
    : '*'
    ;

DIVISION
    : '/'
    ;

FLOORDIV
    : '//'
    ;

MODULO
    : '%'
    ;

ADDITION
    : '+'
    ;

SUBTRACTION
    : '-'
    ;

LEFT_SHIFT
    : '<<'
    ;

RIGHT_SHIFT
    : '>>'
    ;

GREATER_THAN
    : '>'
    ;

LESS_THAN
    : '<'
    ;

GREATER_THAN_OR_EQUAL
    : '>='
    ;

LESS_THAN_OR_EQUAL
    : '<='
    ;

EQUAL_TO
    : '=='
    ;

NOT_EQUAL_TO
    : '!='
    ;

AND
    : '&'
    ;

EXCLUSIVE_OR
    : '^'
    ;

OR
    : '|'
    ;

LOGICAL_AND
    : '&&'
    ;

LOGICAL_OR
    : '||'
    ;

/*
 * Fundamental Tokens
 */

IDENTIFIER
    : NONDIGIT (NONDIGIT | DEC_DIGIT)*
    ;

INT_LITERAL
    : BINARY_INT_LITERAL
    | OCTAL_INT_LITERAL
    | DECIMAL_INT_LITERAL
    | HEXADECIMAL_INT_LITERAL
    ;

FLOAT_LITERAL
    : FRACTION_PART EXPONENT_PART?
    | DIGIT_SEQUENCE EXPONENT_PART
    ;

COMPLEX_LITERAL
    : FRACTION_PART EXPONENT_PART? [jJ]
    | DIGIT_SEQUENCE EXPONENT_PART? [jJ]
    ;

FRACTION_PART
    : DIGIT_SEQUENCE? DOT DIGIT_SEQUENCE
    | DIGIT_SEQUENCE DOT
    ;

NONDIGIT
    : [a-zA-Z_]
    ;

BINARY_INT_LITERAL
    : '0' [bB] (BIN_DIGIT | '_')+
    ;

OCTAL_INT_LITERAL
    : '0' [oO] (OCT_DIGIT | '_')+
    ;

DECIMAL_INT_LITERAL
    : DIGIT_SEQUENCE
    ;

HEXADECIMAL_INT_LITERAL
    : '0' [xX] HEX_DIGIT (HEX_DIGIT | '_')*
    ;

EXPONENT_PART
    : [eE] SIGN? DIGIT_SEQUENCE
    ;

DIGIT_SEQUENCE
    : DEC_DIGIT (DEC_DIGIT | '_')*
    ;

SIGN
    : [+-]
    ;

BIN_DIGIT
    : [0-1]
    ;

OCT_DIGIT
    : [0-7]
    ;

DEC_DIGIT
    : [0-9]
    ;

HEX_DIGIT
    : [0-9a-fA-F]
    ;

WHITESPACE
    : [ \t]+ -> skip
    ;

NEWLINE
    : ('\r' '\n'? | '\n') -> skip
    ;

LINECOMMENT
    : '#' ~ [\r\n]* -> skip
    ;
