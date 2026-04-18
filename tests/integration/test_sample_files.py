"""Tests compiling source code from file using FhY entry point."""

import os
import re
from glob import glob

import pytest

from .utils import access_cli, get_diff

_HERE = os.path.abspath(os.path.join(__file__, os.pardir))
_SAMPLES = os.path.join(_HERE, "data")
_OUTPUT = os.path.join(_SAMPLES, "output")
_INPUT = os.path.join(_SAMPLES, "input", "*.fhy")

examples = glob(_INPUT)


def _grab_expected_output_file(filepath: str) -> str:
    basename: str = os.path.basename(filepath).split(".")[0]
    name = f"{basename}_output.fhy"
    path_out = os.path.join(_OUTPUT, name)
    if not os.path.exists(path_out):
        raise FileNotFoundError(f"Expected output file does not exist: {basename}")
    return path_out


def _iter_lines(text: str):
    yield from re.split("[\r\n]+", text)


def _clean_pretty_print_output(output: str) -> str:
    generator = _iter_lines(output)
    return "\n".join(generator).strip()


@pytest.mark.parametrize("file", examples)
def test_single_file_examples_through_cli_pretty(file: str):
    """Test the FhY CLI using pretty print on a collection of example files."""
    code, output, _ = access_cli("serialize", file, "-f", "pretty")
    assert code == 0
    result = _clean_pretty_print_output(output)

    out_path = _grab_expected_output_file(file)
    with open(out_path) as st:
        expected = st.read()

    if result != expected:
        get_diff(result, expected)

    assert result == expected
