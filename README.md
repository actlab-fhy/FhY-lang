# FhY Language and Frontend

[![PyPI version](https://img.shields.io/pypi/v/fhy_lang.svg)](https://pypi.org/project/fhy_lang/)
[![Python versions](https://img.shields.io/pypi/pyversions/fhy_lang.svg)](https://pypi.org/project/fhy_lang/)
[![CI](https://github.com/actlab-fhy/FhY-lang/actions/workflows/python-package.yml/badge.svg)](https://github.com/actlab-fhy/FhY-lang/actions/workflows/python-package.yml)

*A Language for Modeling Physical Things*

*FhY* is a cross-domain language with mathematical foundations that moves beyond the
current paradigm of domain-specific languages to enable cross-domain multi-acceleration.

This package provides the user-facing surface of the compiler: the FhY source language, its grammar and parser (built on ANTLR), the AST definitions, and the driver that turns a FhY source program into a fully-constructed AST module and symbol table ready to be consumed by downstream compilation stages.

## Features

- FhY grammar and ANTLR-generated parser
- Abstract syntax tree (AST) definitions and pretty-printing utilities
- Parse-tree-to-AST conversion
- Symbol table construction
- Compilation driver with configurable options (verbose logging, optimization)
- AST serialization to JSON or human-readable pretty-printed formats

## Command Line Interface

Installing the package exposes the `compile-fhy-lang` command:

```bash
compile-fhy-lang --help
compile-fhy-lang --version
```

### Compile and serialize a FhY source program

```bash
compile-fhy-lang serialize path/to/main.fhy [OPTIONS]
```

Common options:

- `--format, -f {json,pretty,prettyid}` - serialization format for the resulting AST
- `--indent, -i N` - indentation width for human-readable output
- `--verbose` - enable debug-level logging
- `--optimize` - enable frontend optimizations
- `--log-file PATH` - write logs to the given file

If no `--format` is provided, the program is compiled but no serialized output is emitted.

## Currently supported language subset

FhY is pre-1.0; the parser accepts the full grammar but the parse-tree
converter has not yet been wired up for every construct. Sources that use a
not-yet-supported feature raise `NotImplementedError` from the converter
with a `<file>:<line>:<col>` location pointing at the offending construct.

Supported and converted to AST today:

- Procedures (`proc name(...) { ... }`) and operations (`op name(...) -> ... { ... }`)
- `forall` statements over an index expression
- Declaration statements (`temp T name`, `temp T name = expr`, including
  `index` and array element types)
- Expression statements with or without a left-hand side
- Return statements (in operations)
- Unary, binary, and ternary expressions
- Array access (`x[i, j]`) and tuple access (`x.0`, `x.1`)
- Function calls and operation/reduction calls
- Tuple types and tuple expressions

Not yet supported (currently raise `NotImplementedError`):

- `if` / `else` selection statements
- `import` statements
- Bodyless function declarations (signatures without `{ ... }`)
- Function indices in headers (`op f<>[i, j](...) -> ...`)
- Custom dtype template parameters (`temp myparam<5> x;`)
