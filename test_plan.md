# FhY Language Test Plan

This document enumerates the test cases that should exist to cover every feature and error path described in the FhY language reference (the `.rst` files alongside this document). It is feature-oriented, not implementation-oriented: concrete test files may bundle several cases together.

Each entry below is tagged with its expected outcome:

- **[OK]** — a well-formed program. The compiler should accept it and (where applicable) it should evaluate to the expected result.
- **[ERR]** — a malformed program. The compiler should reject it at compile time with the corresponding language-level error.
- **[WARN]** — a discouraged but accepted program. The compiler should compile it but emit the corresponding warning.

The grouping below mirrors the reference chapters in order.

---

## 1. Lexical structure

### 1.1 Whitespace and comments

- **[OK]** Tokens separated by any combination of spaces and tabs.
- **[OK]** Tokens separated by `\n`, `\r`, or `\r\n`.
- **[OK]** A `#` line comment is consumed up to the end of the line.
- **[OK]** A trailing comment on the same line as a statement does not change the meaning of the statement.
- **[OK]** A program with comments interspersed between every legal token boundary still parses.
- **[ERR]** A C-style block comment (`/* ... */`) is not recognized; using one is a lexical/parse error.

### 1.2 Identifiers

- **[OK]** Identifiers may begin with a letter or underscore and contain letters, digits, and underscores afterward.
- **[OK]** `X` and `x` are distinct identifiers (case sensitivity).
- **[OK]** Identifier length is unbounded (a very long identifier still parses).
- **[ERR]** An identifier that starts with a digit is a lexical error.

### 1.3 Keywords

- **[ERR]** Each active keyword used as a user-defined name: `tuple`, `index`, `proc`, `op`, `forall`, `return`.
- **[ERR]** Each reserved type-name keyword used as a user-defined name: every entry of `int`, `int8`, `int16`, `int32`, `int64`, `uint`, `uint8`, `uint16`, `uint32`, `float32`, `float64`, `complex64`, `complex128`.
- **[ERR]** Each reserved built-in reduction name used as a user-defined name: `sum`, `prod`, `min`, `max`.
- **[ERR]** Each reserved-but-currently-unused keyword used as a user-defined name: `native`, `if`, `else`, `import`, `from`, `as`, `reduc`.

### 1.4 Literals

- **[OK]** Decimal integer literal.
- **[OK]** Binary integer literal: `0b1010`, `0B1010`.
- **[OK]** Octal integer literal: `0o17`, `0O17`.
- **[OK]** Hexadecimal integer literal: `0x1A`, `0X1A`.
- **[OK]** Integer literal with underscore digit separators: `1_000`, `0xFF_FF`.
- **[OK]** Float literal with leading digit and fraction: `1.0`, `1.5e3`.
- **[OK]** Float literal with leading dot: `.5`.
- **[OK]** Float literal with trailing dot: `2.`.
- **[OK]** Float literal with exponent without fraction part: `2e10`.
- **[OK]** Complex literal: `1.0j`, `3J`, `2.5e-1j`.
- **[OK]** Integer literal weak-typing: the same literal is accepted by sites that demand narrower or wider integer types, and the narrowest faithful type is chosen when the literal is unconstrained.

---

## 2. Types

### 2.1 Numerical types

- **[OK]** Each built-in scalar dtype used as the type of an `input` argument (a value can be passed in and read).
- **[OK]** Each built-in scalar dtype used as the type of an `output` argument (a value can be written out).
- **[OK]** A tensor with all literal dimensions: `int32[1, 32, 32, 16]`.
- **[OK]** A tensor with a single placeholder dimension introduced implicitly: `int32[N]`.
- **[OK]** A tensor whose shape dimension is arithmetic over a placeholder and a literal: `int32[N + 4]`.
- **[OK]** A tensor whose shape references an existing module-level `param` constant rather than introducing a new placeholder.
- **[OK]** Scalar types: a numerical type with an empty shape, and one whose every shape dim is `1`, both behave as scalars.
- **[ERR]** A shape dimension expression that cannot be reduced to a constant at compile time is rejected.

### 2.2 Index types

- **[OK]** Half-open range: `index[1:N]` covers `1, …, N`.
- **[OK]** Range with stride: `index[1:N:2]`.
- **[OK]** Range whose bounds reference a `param` constant.
- **[ERR]** Bounds or stride that is not compile-time reducible is rejected.

### 2.3 Tuple types

- **[OK]** Tuple types of arities 1, 2, and ≥3.
- **[OK]** Heterogeneous tuple of mixed scalar dtypes.
- **[OK]** Tuple whose element type is itself a tensor.
- **[OK]** Tuple value built with `(e1, e2, …)`; singleton tuple `(x,)` with trailing comma.
- **[OK]** Tuple access `t.0`, `t.1`, … up to `t.<arity-1>`.
- **[ERR]** Out-of-bounds tuple access (`t.k` with `k >= arity`).
- **[ERR]** Negative tuple-access index (if syntactically reachable).

### 2.4 Type qualifiers

- **[OK]** `input` argument: readable inside the body.
- **[OK]** `output` argument: writable inside the body, fully covered on every path.
- **[OK]** `temp` declaration with an initializer is readable thereafter.
- **[OK]** `temp` declaration without an initializer is readable on any path that writes to it first.
- **[OK]** `param` declaration with a compile-time-constant initializer.
- **[OK]** `param` argument supplied by the caller, used as a constant.
- **[ERR]** Writing to an `input` variable (scalar or via indexed assignment).
- **[ERR]** `input` or `output` qualifier on a declaration statement in a function body.
- **[ERR]** `temp` qualifier on a function argument list.
- **[ERR]** `param` declaration assigned a value that is not compile-time reducible.
- **[ERR]** Read of a `temp` on a path that does not definitely assign it.
- **[ERR]** `output` array not fully written across all paths.

### 2.5 Type promotion and assignability

- **[OK]** Assignment with identical types on both sides.
- **[OK]** Assignment where the RHS is a weakly-typed integer literal that promotes exactly to the LHS dtype.
- **[OK]** Call-site shape compatibility when the actual and formal have matching literals or symbolic dims.
- **[ERR]** Assignment with different shape arity on the two sides.
- **[ERR]** Assignment where the RHS literal dimension disagrees with the LHS literal dimension.
- **[ERR]** Assignment where promotion from the RHS dtype to the LHS dtype is not legal.

---

## 3. Modules

### 3.1 Single-file program structure

- **[OK]** A module consisting of a single `proc main` compiles.
- **[OK]** A module with a `param` constant, several `op` definitions, and a `proc main` compiles.
- **[OK]** A module-level `param` is in scope inside every function in the module.
- **[ERR]** A module with no `main` procedure is rejected.
- **[ERR]** A non-`param` declaration at module scope.
- **[ERR]** A `param` declaration of a tensor type at module scope.
- **[ERR]** A `param` declaration of a tuple type at module scope.
- **[ERR]** A `param` declaration at module scope without an initializer.
- **[ERR]** An assignment statement at module scope.
- **[ERR]** A `forall` at module scope.
- **[ERR]** A bare expression statement at module scope.

### 3.2 Name resolution

- **[OK]** A name declared in a function shadows nothing from the built-in namespace.
- **[OK]** A built-in function name (e.g., `exp`) resolves at module scope without import.
- **[ERR]** A user-defined identifier whose name collides with a built-in function name (the user-defined name attempts to shadow the built-in).
- **[ERR]** Use of any reserved keyword as a user-defined name (covers both data-type names and the unused-keyword set).

---

## 4. Functions

### 4.1 Procedures (`proc`)

- **[OK]** `proc` with `input`, `output`, and `param` arguments compiles.
- **[OK]** `proc` body of declarations and indexed assignments compiles.
- **[OK]** Calling `matmul(A, B, C);` from `main` invokes the procedure for its side effects on the `output` argument.
- **[ERR]** Procedure header with a `-> return_type`.
- **[ERR]** Procedure body containing a `return` statement.
- **[ERR]** Procedure call appearing inside a larger expression (value position).

### 4.2 Operations (`op`)

- **[OK]** `op` with scalar arguments and a scalar `output` return type, ending in `return`.
- **[OK]** Operation called inside an indexed assignment as a value-producing expression.
- **[ERR]** Operation argument of any non-scalar numerical type.
- **[ERR]** Operation argument of tuple type.
- **[ERR]** Operation argument of index type.
- **[ERR]** Operation return type that is not a scalar numerical type.
- **[ERR]** Operation return type that does not carry the `output` qualifier.
- **[ERR]** Operation whose body fails to reach a `return` on some path (testable only once branching is added; for now, every `op` body should reach the single `return`).

### 4.3 Template type parameters

- **[OK]** `op` parameterized by `<T>` is callable as `f<int32>(...)`, `f<float32>(...)`.
- **[OK]** `proc` parameterized by `<T>` is callable with a vanilla data type as the template substitution.
- **[OK]** Argument arity and shape compatibility are checked against the substituted signature.
- **[ERR]** Template argument that is not one of the listed primitive data-type names.

### 4.4 Function arguments

- **[OK]** A named argument referenced by name in the body.
- **[OK]** A shape identifier inside an argument type becomes an implicit `param uint32` in scope when no outer binding exists.
- **[OK]** A shape identifier inside an argument type resolves to an existing module-level `param` when one exists, without redeclaring.

### 4.5 Call-site shape-placeholder resolution

- **[OK]** Actual argument whose shape consists of literal integers that match the formal's literal dimensions.
- **[OK]** Formal dimension expression mentioning exactly one placeholder is solved against the actual dimension.
- **[OK]** Formal dimension that is arithmetic over one placeholder (`N + 4`, `2 * M - 1`) is solved correctly.
- **[OK]** All placeholders in the signature are assigned by the end of the walk.
- **[ERR]** A placeholder is rebound to a value that is not structurally equivalent to its existing binding during the same call. (Rebinding to a structurally-equivalent value is permitted, so the natural pattern of a callee placeholder appearing in multiple argument shapes — e.g. `proc f(int32[N] x, int32[N, M] y)` — works.)
- **[ERR]** A placeholder is left unassigned after the entire actual/formal walk has finished.
- **[ERR]** A formal dimension mentions two or more placeholders, and no other dimension solves any of them on its own (the language does not solve systems across dimensions).

### 4.6 Recursion

- **[ERR]** A `proc` that calls itself directly.
- **[ERR]** An `op` that calls itself directly.
- **[ERR]** Two functions that call each other (mutual recursion).
- **[ERR]** A longer recursive cycle through three or more functions.

---

## 5. Statements

### 5.1 Declaration statement

- **[OK]** `temp T x;` and `temp T x = init;`.
- **[OK]** `param T x = const_expr;` with a compile-time-constant initializer.
- **[ERR]** Declaration with `input` or `output` qualifier.
- **[ERR]** `param` declaration without an initializer.
- **[ERR]** `param` declaration with an initializer that is not compile-time reducible.

### 5.2 Assignment statement

- **[OK]** `y = expr;` where `y` is a scalar variable in scope.
- **[OK]** `C[i, j] = expr;` where `C` is an array variable and the indices are index-typed.
- **[OK]** Indexed LHS triggers implicit unordered iteration over the named index variables.
- **[OK]** Assignment to a `temp`, `output`, or `param`, respecting that variable's value-domain rules.
- **[ERR]** LHS that is a function-call result.
- **[ERR]** LHS that is a tuple access (e.g., `t.0 = …;`).
- **[ERR]** LHS that is a chained array access (e.g., `a[i][j] = …;`).
- **[ERR]** Assignment whose LHS underlies an `input` variable.
- **[WARN]** A bare expression statement `<expr>;` whose value is computed and discarded (no LHS, no procedure call).

### 5.3 Procedure-call statement

- **[OK]** `proc_name(arg1, …);` invokes the named procedure for its side effects.
- **[WARN]** A statement in this position whose call target is an `op` instead of a `proc`: the return value of the `op` is discarded. The compiler emits a warning but accepts the program.

### 5.4 Iteration statement (`forall`)

- **[OK]** `forall (i) { … }` where `i` is bound to an `index`-typed variable.
- **[OK]** Inside the body, the named index is consumed (no implicit iteration over it); other index variables retain their implicit iteration.
- **[OK]** Successive iterations of `forall (e)` see writes from earlier iterations (ordered semantics).
- **[OK]** A `forall` introduces a new namespace scope.
- **[ERR]** `forall (1:N) { … }` (literal range in the head position).
- **[ERR]** `forall (i + 1) { … }` (compound expression in the head position).
- **[ERR]** `forall (x) { … }` where `x` is not bound to an index type.

### 5.5 Return statement

- **[OK]** `return expr;` inside an `op` returns a scalar of the declared return type.
- **[OK]** The returned expression's type must be assignable to the declared return type (covered by 2.5).
- **[ERR]** A `return` statement inside a `proc`.

---

## 6. Expressions

### 6.1 Atoms

- **[OK]** Integer, float, and complex literals are valid expressions.
- **[OK]** A bare identifier reference resolves to the in-scope binding.
- **[OK]** Tuple construction: `(a, b)`, `(a, b, c)`, and singleton `(a,)`.

### 6.2 Array and tuple access

- **[OK]** `a[i1, …, ik]` with rank matching the declared rank of `a`.
- **[OK]** `t.k` for `k` a literal non-negative integer less than the tuple's arity.
- **[OK]** `t.0` and the leading-dot float-literal trick: no whitespace between dot and digit.
- **[ERR]** Array access whose index expression at some position is provably outside the declared shape along that dimension.
- **[ERR]** Array access whose rank does not match the array's declared rank.
- **[ERR]** Tuple access where the literal index is out of bounds.

### 6.3 Function calls

- **[OK]** Plain function call `f(a, b)`.
- **[OK]** Templated call `f<int32>(a, b)`.
- **[OK]** Reduction call `sum[i, j](body)` (covered also in 6.4).
- **[ERR]** Chained call `f(a)(b)`.
- **[ERR]** Function position that is not a plain identifier.
- **[ERR]** Arity mismatch between actual and formal arguments.
- **[ERR]** Index arguments (`[ … ]`) on a non-reduction call.

### 6.4 Reductions

- **[OK]** `sum`, `prod`, `min`, `max` over a single index.
- **[OK]** Reduction over multiple distinct indices.
- **[OK]** Body expression that references all of the declared indices.
- **[ERR]** Reduction call with zero body arguments or more than one body argument.
- **[ERR]** Reduction with an index argument that is not a plain identifier.
- **[ERR]** Reduction whose index name is not bound to an index-typed variable.
- **[ERR]** Reduction with a repeated index name.
- **[ERR]** Reduction with a declared index that is not actually used in the body.

### 6.5 Operators

- **[OK]** Each binary operator parses and evaluates with the expected semantics on numerical operands: `+`, `-`, `*`, `/`, `//`, `%`, `**`, `<<`, `>>`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `&`, `^`, `|`, `&&`, `||`.
- **[OK]** Each unary operator parses and evaluates: `-`, `~`, `!`.
- **[OK]** Ternary `c ? a : b`.
- **[OK]** Operator precedence follows the spec table; golden test cases for each pair of adjacent precedence levels.
- **[OK]** Unary binds tighter than power: `-x ** 2` parses as `(-x) ** 2`.
- **[OK]** Power is left-associative: `a ** b ** c` parses as `(a ** b) ** c`.
- **[OK]** All binary operators are left-associative (golden cases for at least the multiplicative and additive levels).

### 6.6 Side effects and constant safety

- **[OK]** Re-evaluating the same expression twice in a row has no observable effect (purity).
- **[ERR]** Division by a literal-zero RHS: `x / 0`.
- **[ERR]** Floor-division by a literal-zero RHS: `x // 0`.
- **[ERR]** Modulo by a literal-zero RHS: `x % 0`.
- **[ERR]** Division (any of the above) by a negation chain that reduces to literal zero.

---

## 7. Iteration model

- **[OK]** An indexed assignment is iterated over the full Cartesian product of its LHS index variables.
- **[OK]** 1-based indexing: along a dimension of size `n`, the first valid index is `1` and the last is `n`.
- **[OK]** A reduction collapses each named index over its declared range.
- **[OK]** `forall (e)` enforces ordered iteration: writes performed at step `e` are observable to step `e + 1`.
- **[OK]** Inside `forall (e)`, the index `e` is consumed; other index variables present in inner assignments retain their implicit unordered iteration.
- **[ERR]** An index variable appears only on the right-hand side of an assignment and is not consumed by any enclosing reduction or `forall` (no inferred broadcast).
- **[ERR]** An assignment whose right-hand side reads the array being written at an index that differs from the LHS index for that iteration (cross-iteration state, e.g., `A[i] = A[i - 1] + 1;`).

---

## 8. Cross-cutting

Some rules require composing multiple features in one program; the cases below ensure they hold together.

- **[OK]** A module containing a module-level `param`, an `op`, and a `proc main` that calls the `op` from inside an indexed assignment, compiles and produces correct results.
- **[OK]** A module containing two `proc`s where `main` invokes the other as a procedure-call statement.
- **[OK]** A `proc` whose argument shapes are wholly described by module-level `param` constants (no implicit placeholders).
- **[OK]** A `proc` whose argument shapes mix module-level `param` constants and literal dimensions.
- **[OK]** A `forall(e) { reduce-and-assign }` pattern of the form used in stochastic-gradient-descent: the outer index is ordered, the inner indices are still implicit and unordered.
- **[ERR]** A program that uses every legal feature *except* `main` — confirms that the entry-point requirement is checked even in otherwise-valid programs.
