# Fuse DSL Reference

[[toc]]

The Fuse front-end is a compact DSL for wiring tensor equations. Two parsers
are available:

- Legacy line‑oriented grammar (default) — the original, minimal surface.
- Structured grammar (v2, opt‑in) — adds expressions and blocks while keeping
  the legacy syntax a strict subset.

To opt into v2 in Python, construct `Program` with `parser='v2'`:

```
from fuse.core.program import Program
prog = Program(src_text, parser='v2')
```

## Lexical structure

* Programs are plain text. Every non-empty line (ignoring `#` comments) is a
  statement.
* Statements can span multiple lines by balancing `()`, `[]`, or `{}`. Closing
  all delimiters flushes the pending statement.
* Identifiers begin with `A–Z`/`a–z`/`_` and may contain digits/underscores.
  Tensor indices use bare identifiers.
* String literals use double quotes and are primarily employed for source/sink
  file paths.

## Statements (legacy)

### Equations

```
Target[i, j] = LHS[i, k] RHS[k, j]
Target[i, j] += Bias[i, j]
Target[i] max= MaxSources[i, k]
Target[i] avg= MeanSources[i, k]
```

Supported projection operators on the left hand side:

| Operator | Projection | Semantics                                                  |
|----------|------------|------------------------------------------------------------|
| `=`      | `sum`      | Default contraction. RHS-only axes are summed out.         |
| `+=`     | `sum`      | Emits a fresh equation sharing the same LHS (sugared add). |
| `max=`   | `max`      | Projects RHS-only axes by `max`.                           |
| `avg=`   | `mean`     | Projects RHS-only axes by arithmetic mean.                 |

LHS indices marked with a trailing `.` designate *dotted axes* for reductions.
For example `Soft[p, q.] = softmax(Logits[p, q])` indicates the reduction axis
within the builtin call.

### Sources and sinks

```
Weights[h, d] = "ckpt/weights.npy"    # source
"runs/activations.npz" = Activation[i, j]    # sink
```

Sources must use `=`. Sinks always place the filename on the left and use the
default projection (`sum`).

### Boolean terms

Parenthesised names such as `Fact(i, j)` signal boolean tensors (as opposed to
dense numeric tensors `Fact[i, j]`). Both syntaxes share the same semantics once
parsed, but parentheses are a useful convention when mixing boolean logic with
dense math.

### Index functions

Fuse includes a small set of unary index functions that emit boolean masks:

```
Even[i]  = even(i)
Odd[i]   = odd(i)
```

These must be supplied an axis either positionally or via `axis=`. For example
`even(i)` and `even(axis=i)` are equivalent.

## Function calls and builtins

Common single-argument builtins include:

* `relu`, `sig`, `gelu`, `softmax`, `lnorm`, `layernorm`, `masked_softmax`,
  `attention`, `rope`, `concat`, `causal_mask`, `topk`, `const`, `reduce_max`,
  `reduce_mean`, `sin`, `cos`, `case`, `tucker_dense`.

Arguments can be a tensor expression, a tuple, or include keyword arguments:

```
Soft[p, q.] = softmax(Logits[p, q], axis="q")
Scaled[i, j] = concat(A[i, j], B[i, j], axis="j")
```

## Projection & axis semantics

Fuse determines contraction axes by comparing LHS and RHS indices:

* Axes that appear on the RHS but not on the LHS are *projected*.
* Projection behaviour depends on the operator (`sum`, `max`, `mean` from above).
* Index order matters. The LHS establishes the storage order for emitted tensors.
* Shorthand `+=` splits into separate equations, each sharing the same LHS and
  projection descriptor.

Boolean tensors (`Fact(i, j)`) follow the same projection rules. A projected
boolean index (e.g., `Fact(i, k)` with `avg=`) will coerce to numeric semantics,
so choose the projection that matches your intent.

## Grammar sketch

```
program        ::= statement*
statement      ::= equation | source | sink | export
equation       ::= lhs operator rhs
lhs            ::= IDENTIFIER index_spec?
index_spec     ::= '[' indices ']' | '(' indices ')'
indices        ::= IDENTIFIER (',' IDENTIFIER)* ('.')?
operator       ::= '=' | '+=' | 'max=' | 'avg='
rhs            ::= sum_term ('+' sum_term)*
sum_term       ::= product_term (product_term)*
product_term   ::= IDENTIFIER index_spec?
                 | literal
                 | function_call
                 | index_function
function_call  ::= IDENTIFIER '(' arguments? ')'
arguments      ::= expr (',' expr)*
index_function ::= IDENTIFIER '(' (IDENTIFIER | 'axis=' IDENTIFIER) ')'
source         ::= lhs '=' STRING
sink           ::= STRING '=' rhs
export         ::= 'export' IDENTIFIER
```

The grammar above is intentionally approximate: it omits precedence details and
the parser accepts extra whitespace and comments, but it captures the core shape
of the DSL.

## Quick checklist

* Use dotted axes on the LHS to indicate softmax/normalization axes.
* Remember that `+=` emits an independent equation; it does **not** mutate the
  previous LHS in place.
* Index functions (`even`, `odd`) require an explicit axis.
* File sources/sinks always work with `np.load`/`np.save` semantics (`.npy`,
  `.npz`, `.jsonl`, etc.), respecting runtime policies like memory mapping.

Keep this reference handy while authoring `.fuse` programs or embedding
equations inside Python helpers.

---

## Structured syntax (v2)

The v2 grammar adds expressions, blocks, pure functions, and macros. It compiles
down to the same IR, and the original line syntax remains valid.

Highlights:

- Let bindings and multi‑line blocks
  - `let sim[u,v] = Emb[u,d] * Emb[v,d];`
  - `{ let t = 1 + 2; A[i] = t * x[i]; }`

- Arithmetic and broadcasting
  - Elementwise `+ - * / **` with axis‑aware broadcasting.
  - Reductions via `reduce(op, axes) expr`, e.g. `reduce(sum, d) x[i,d]*x[i,d]`.

- Guards and selects
  - `score[u] = select(risky[u], hi[u], lo[u]);`
  - Guard sugar: `score[u] when risky[u] = hi[u];`
  - Piecewise: `case { risky[u] -> hi[u]; default -> lo[u]; }`

- Pure functions (inlineable)
  - `fn dot(a[x], b[x]) -> s[] { s[] = a[x] * b[x]; }`
  - `sim[i,j] = dot(Emb[i,d], Emb[j,d]);`
  - No side‑effects; recursion is forbidden initially.

- Macros (deterministic, hygienic)
  - `@softmax(x, axis=j)` → `softmax(x, axis=j)` or `masked_softmax(...)`.
  - `@layer_norm(x, axis=d, eps=1e-5)` → `layernorm(...)`.

### v2 grammar sketch (subset)

```
program     := (import | export | param | axis | const | fn | stmt)*
stmt        := equation | let | block
equation    := lhs '=' expr | lhs 'when' expr '=' expr
lhs         := IDENT dims?
dims        := '[' IDENT (',' IDENT)* ']'
expr        := ternary
ternary     := or_expr ('?' expr ':' expr)?
or_expr     := and_expr ('||' and_expr)*
and_expr    := add_expr ('&&' add_expr)*
add_expr    := mul_expr (('+'|'-') mul_expr)*
mul_expr    := pow_expr (('*'|'/') pow_expr)*
pow_expr    := unary ('**' pow_expr)?
unary       := primary | ('-'|'!') unary
primary     := NUMBER | IDENT | tensor_ref | call | macro | piecewise | reduce | '(' expr ')'
tensor_ref  := IDENT '[' index (',' index)* ']'
call        := IDENT '(' args? ')'
macro       := '@' IDENT '(' args? ')'
args        := (IDENT '=' expr | expr) (',' (IDENT '=' expr | expr))*
reduce      := 'reduce' '(' IDENT ',' axes ')' expr
```

### Migration (no rewrites)

- All existing `.fuse` files keep working (legacy is the default parser).
- You can start using `let` and arithmetic first—no semantic risk.
- Add `select` and `case` where you currently simulate branching via multiple equations.
- Introduce functions as inline sugar only; forbid recursion initially.
- Macros expand to pure calls before type/shape checking to preserve determinism.

See also: `Program(src, parser='v2')`.
