"""FhY source code to AST module converter."""

import logging
from typing import cast

from antlr4 import (  # type: ignore[import-untyped]
    DFA,
    CommonTokenStream,
    InputStream,
)
from antlr4.atn.ATNConfigSet import ATNConfigSet  # type: ignore[import-untyped]
from antlr4.error.ErrorListener import (  # type: ignore[import-untyped]
    ErrorListener,
)
from fhy_core import Provenance, get_logger

from fhy_lang import ast
from fhy_lang.parser import FhYLexer, FhYParser

from .error import FhYSyntaxError
from .from_parse_tree import from_parse_tree

_logger = get_logger(__name__)


class ThrowingErrorListener(ErrorListener):  # type: ignore[misc]
    """ANTLR error listener that surfaces parser diagnostics as FhY errors.

    Overrides ANTLR's default error listener to raise FhYSyntaxError on lex
    or parse errors and to log ambiguity, full-context, and context-sensitivity
    reports at debug level.
    """

    def __init__(self, log: logging.Logger = _logger) -> None:
        super().__init__()
        self.logger = log

    def syntaxError(
        self,
        recognizer: FhYParser,
        offendingSymbol: str,
        line: int,
        column: int,
        msg: str,
        e: Exception,
    ) -> None:
        text = self.get_text(recognizer, None, None)
        context = type(recognizer._ctx).__name__
        message = (
            f'context={context}(Line {line}:{column}) input="{offendingSymbol}" '
            f'text="{text}" - msg={msg}'
        )

        self.logger.error(message)

        raise FhYSyntaxError(message) from e

    def reportAmbiguity(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        exact: bool,
        ambigAlts: set[int],
        configs: ATNConfigSet,
    ) -> None:
        report = self._report(
            recognizer,
            dfa,
            startIndex,
            stopIndex,
            configs,
        )
        report += f" exact={exact}"
        self.logger.debug(report)

    def reportAttemptingFullContext(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        conflictingAlts: set[int],
        configs: ATNConfigSet,
    ) -> None:
        report = self._report(
            recognizer,
            dfa,
            startIndex,
            stopIndex,
            configs,
        )
        self.logger.debug(report)

    def reportContextSensitivity(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        prediction: int,
        configs: ATNConfigSet,
    ) -> None:
        msg = self._report(
            recognizer,
            dfa,
            startIndex,
            stopIndex,
            configs,
        )
        msg += f" predict={prediction}"
        self.logger.debug(msg)

    def _report(
        self,
        recognizer: FhYParser,
        dfa: DFA,
        startIndex: int,
        stopIndex: int,
        configs: ATNConfigSet,
    ) -> str:
        decision = self.get_decision(recognizer, dfa)
        conflict = {i.alt for i in configs}
        text = self.get_text(recognizer, startIndex, stopIndex)
        if (ctx := recognizer._ctx) is None:
            context = ""
        else:
            context = ctx.getText()

        message = (
            f"context={context}, d={decision}: ambigAlts={conflict}, input='{text}'"
        )

        return message

    def get_text(
        self,
        recognizer: FhYParser,
        start: int | None = None,
        stop: int | None = None,
    ) -> str:
        stream = recognizer.getTokenStream()
        text = cast(str, stream.getText(start, stop))

        return text

    def get_decision(self, recognizer: FhYParser, dfa: DFA) -> str:
        decision: int = dfa.decision
        index: int = dfa.atnStartState.ruleIndex

        rules: list[str] = recognizer.ruleNames
        if index < 0 or index >= len(rules):
            return str(decision)

        name = rules[index]
        if not name:
            return str(decision)

        return name


def create_lexer(input_str: str) -> FhYLexer:
    """Construct the FhyLexer from input string source code."""
    input_stream = InputStream(input_str)
    lexer = FhYLexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(ThrowingErrorListener(_logger))

    return lexer


def create_parser(input_str: str) -> FhYParser:
    """Construct the FhyParser from input string source code."""
    lexer = create_lexer(input_str)
    token_stream = CommonTokenStream(lexer)
    parser = FhYParser(token_stream)
    parser.removeErrorListeners()
    parser.addErrorListener(ThrowingErrorListener(_logger))

    return parser


def _fhy_source_to_parse_tree(fhy_source_content: str) -> FhYParser.ModuleContext:
    fhy_parser = create_parser(fhy_source_content)
    tree = fhy_parser.module()  # type: ignore[no-untyped-call]
    return cast(FhYParser.ModuleContext, tree)


def from_fhy_source(
    fhy_source_content: str,
    provenance: Provenance,
) -> ast.Module:
    """Convert FhY source code into the corresponding AST module representation.

    Args:
        fhy_source_content: FhY source code text.
        provenance: Provenance of the source code.

    Returns:
        AST module representation of input source code.

    Raises:
        FhYSyntaxError: Lex, parse, or semantic-shape error in the source.
        NotImplementedError: Source uses a feature that is not yet supported
            (imports, selection statements, function declarations, function
            indices, dtype template parameters).

    """
    _logger.debug("Lexing and parsing FhY source (%d chars).", len(fhy_source_content))
    tree = _fhy_source_to_parse_tree(fhy_source_content)
    _logger.debug("Parse tree constructed. Converting to AST...")
    _ast = from_parse_tree(tree, provenance)
    _logger.debug(
        "AST module constructed: %d top-level statement(s).", len(_ast.statements)
    )

    return _ast
