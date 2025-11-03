# Migration to structured syntax (v2)

This page outlines a no‑rewrite migration path from the legacy line‑oriented DSL to the new structured grammar with expressions and blocks. The goal is incremental adoption with zero breakage.

## Principles

- Backward compatible: all existing `.fuse` files keep working.
- Sugar only: v2 lowers to the same IR and backends — no semantic changes.
- Deterministic: no Turing‑complete control flow; macros expand to pure calls.
- Opt‑in: v2 is gated via `Program(..., parser='v2')`.

## Recommended steps

1) Start with `let` bindings and arithmetic
- Safely factor common subexpressions without changing semantics.
- Example: `let sim[u,v] = Emb[u,d] * Emb[v,d];`

2) Add `select` and `case`
- Replace multi‑equation branching with deterministic blends.
- Example: `score[u] = select(risky[u], hi[u], lo[u]);`

3) Introduce pure functions (inlineable)
- Refactor repeated patterns into `fn` bodies; forbid recursion.
- Example: `fn dot(a[x], b[x]) -> s[] { s[] = a[x] * b[x]; }`

4) Use named reductions and axis clauses
- Prefer `reduce(sum, k) expr` over implicit projection for readability.

5) Macros for ergonomics
- Use `@softmax` and `@layer_norm` as shorthand. Macros expand AST→AST before shape checking.

## How to opt‑in

- In Python: `Program(src, parser='v2')`
- The CLI continues to use the legacy parser to keep existing projects untouched. You can embed v2 via your Python harness.

## Negative guidance

- No recursion yet: functions are inline sugar only.
- Keep macros pure: they must expand to expressions (no side‑effects).
- Validate shapes: broadcasting must be explicit or axis‑aligned.

## Testing strategy

- Golden DSL: parse → pretty‑print → parse roundtrip to ensure stable formatting.
- Property tests: random shapes/axes to exercise broadcasting/union edge cases.
- Conformance: compare NumPy (spec) vs Torch FX vs JAX/XLA for equality/tolerance, including `select`, `case`, and function inlining.
- Negative tests: ambiguous axes, type mismatches, non‑broadcastable masks, and non‑pure function bodies.

## Troubleshooting

- If a function call fails to inline, ensure the callee is available (including namespaces from imports) and the argument axes are compatible.
- For masked ops, verify mask shapes broadcast to the data tensor.
- For reductions, pick an explicit op (`sum`, `max`, `mean`) and list the reduced axes.

