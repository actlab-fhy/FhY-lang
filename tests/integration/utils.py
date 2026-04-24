"""Common integration test utilities."""

import difflib
import os
import subprocess
import sys


def get_diff(a: str, b: str) -> None:
    """Print a diff between two strings."""
    for i, s in enumerate(difflib.ndiff(a, b)):
        if s[0] == " ":
            continue
        elif s[0] == "-":
            print(f'Delete "{s[-1]}" from position {i}')
        elif s[0] == "+":
            print(f'Add "{s[-1]}" to position {i}')


def access_cli(*args: str, cwd: str | None = None) -> tuple[int, str, str]:
    """Access FhY entry point using subprocess and return the decoded stdout.

    Args:
        args: Arguments to pass to the FhY CLI.
        cwd: Current working directory to run the CLI in.

    Returns:
        A tuple containing the return code, stdout, and stderr.

    """
    cli_executable = os.path.join(os.path.dirname(sys.executable), "fhy_lang")
    result = subprocess.run(
        [cli_executable, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    output = result.stdout.decode()
    errors = result.stderr.decode()

    return result.returncode, output, errors
