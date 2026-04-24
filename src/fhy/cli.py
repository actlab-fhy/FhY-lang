"""FhY command line interface (CLI)."""

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from fhy_core import SerializationFormat as CoreSerializationFormat
from fhy_core import StrEnum, SymbolTable, add_file_handler, get_logger

from fhy import __version__
from fhy.driver import CompilationOptions, Workspace, compile_fhy
from fhy.lang import ASTModule, pformat_ast

app = typer.Typer(
    name="FhY Language and Frontend",
    no_args_is_help=True,
    add_completion=False,
)

_logger: logging.Logger = get_logger(__name__)


def report_version(value: bool) -> None:
    """Report version to stdout and exit if true.

    Args:
        value: Whether to report version and exit.

    """
    if value:
        sys.stdout.write(f"FhY Language and Frontend v{__version__}\n")
        sys.exit(0)


def _confirm_arg(value: bool, check: str) -> bool:
    return value or check in sys.argv


class SerializationOptions(StrEnum):
    """Serialization options for FhY AST nodes."""

    JSON = "json"
    PRETTY = "pretty"
    PRETTYID = "prettyid"


def compile_fhy_source(
    main_file: Path | None = None,
    *,
    verbose: bool = False,
    optimize: bool = False,
    log_file: Path | None = None,
) -> tuple[ASTModule, SymbolTable]:
    """Parse a FhY project, compile it, and return the final AST module."""
    if log_file is not None:
        add_file_handler(
            _logger,
            log_file,
            level=logging.DEBUG if verbose else logging.INFO,
        )

    if main_file is None:
        _logger.error(
            "Please provide a valid file path to the main FhY module source code."
        )
        sys.exit(1)

    workspace = Workspace(main_file)
    options = CompilationOptions(verbose=verbose, perform_optimizations=optimize)

    try:
        program, symbol_table = compile_fhy(workspace, options)

    except KeyboardInterrupt as e:
        _logger.error(
            "Compilation has been interrupted by client.",
            exc_info=e,
        )
        sys.exit(1)
    except Exception as e:
        _logger.error("Compilation has failed.", exc_info=e)
        sys.exit(1)
    else:
        _logger.info("Compilation completed successfully.")

    return program, symbol_table


@app.callback(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def version(
    version: Annotated[
        bool,
        typer.Option(
            "--version", help="Show version and exit.", callback=report_version
        ),
    ] = False,
) -> None:
    """Welcome to FhY!"""
    report_version(_confirm_arg(version, "--version"))


# TODO: Include config option when ready
@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def serialize(
    main_file: Annotated[
        Path | None,
        typer.Argument(help="Valid filepath to main FhY module source code."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Enable debugging.")
    ] = False,
    optimize: Annotated[
        bool, typer.Option("--optimize", help="Enable optimization.")
    ] = False,
    log_file: Annotated[
        Path | None, typer.Option(help="Provide a filepath to write logs to.")
    ] = None,
    format: Annotated[
        SerializationOptions | None,
        typer.Option(
            "--format", "-f", case_sensitive=False, help="Format to serialize to."
        ),
    ] = None,
    indent: Annotated[
        int | None,
        typer.Option(
            "--indent",
            "-i",
            case_sensitive=False,
            help="Indentation to apply to serialization for human readability.",
        ),
    ] = None,
) -> None:
    """Compile a FhY source program and print the result."""
    module, _ = compile_fhy_source(
        main_file, verbose=verbose, optimize=optimize, log_file=log_file
    )

    if format is None:
        return
    elif format == SerializationOptions.JSON:
        serialized = module.serialize(CoreSerializationFormat.JSON)
        if isinstance(serialized, str):
            sys.stdout.write(serialized)
        elif isinstance(serialized, bytes):
            sys.stdout.write(serialized.decode())
        else:
            _logger.error("Unexpected serialization payload type for JSON output.")
            sys.exit(1)
    elif format in (SerializationOptions.PRETTY, SerializationOptions.PRETTYID):
        space: str = (indent or 2) * " "
        show_id: bool = format == SerializationOptions.PRETTYID
        sys.stdout.write(pformat_ast(module, indent_char=space, show_id=show_id))
    else:
        _logger.error(f"Unsupported or invalid serialization format: {format.value}")
        sys.exit(1)
