<p align="center">
  <img src="docs/assets/Fuse.png" alt="Fuse logo" width="160" />
</p>

# Fuse — a tensor‑equation DSL as a library

Fuse expresses AI as a small set of tensor equations: joins + projection (+ optional nonlinearity). The repo ships a NumPy execution engine, a Torch FX lowering, and a JAX path with an optional XLA callable. The code stays small and readable while covering end‑to‑end flows (sources → IR → backends → sinks).

> Logical rules ≙ Einstein sums. Anything not on the left‑hand side is implicitly projected (summed) out. Equations with the same LHS implicitly add. Files are first‑class sources/sinks.

<p align="center">
  <a href="https://pypi.org/project/fuse/"><img src="https://img.shields.io/pypi/v/fuse.svg?label=PyPI&color=blue" alt="PyPI"></a>
  <a href="https://github.com/mehrdadxzaker/fuse/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="License"></a>
  <a href="https://www.neuralint.io/fuse/docs/"><img src="https://img.shields.io/badge/docs-material%20mkdocs-informational" alt="Docs"></a>
  <a href="#backends"><img src="https://img.shields.io/badge/backends-numpy%20%7C%20torch%20%7C%20jax-7f52ff" alt="Backends"></a>
</p>

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]  # or: pip install fuse
```

Run an example:

```bash
python examples/01_attention_block.py
```

Minimal Python harness:

```python
from fuse.core.program import Program

pgm = Program.from_file("examples/01_attention_block.fuse")
runner = pgm.compile(backend="auto")
out = runner.run()
print(out)
```

## Install

- `pip install fuse` – core with NumPy backend
- `pip install fuse[torch]` – enable Torch FX backend
- `pip install fuse[jax]` – enable JAX/XLA path
- `pip install fuse[bench]` – Torch + JAX bundle for benchmarks
- `pip install fuse[dev]` – linting, typing, docs, tests

Using uv (reproducible environments):

```bash
uv sync --extra dev  # swap extras: --extra torch --extra jax
```

## Features

- Compact DSL with joins, projection, axis‑aware concat, reductions, and literal/keyword arguments.
- End‑to‑end path: Sources → IR → Backends → Sinks with caching hooks and runtime policies.
- Backends: NumPy evaluator, Torch FX lowering, and JAX executor with optional XLA callable.
- Clear execution controls via `ExecutionConfig` (precision, device, zero‑copy, XLA cache, validations).
- Quant/LoRA/sharding abstractions via `RuntimePolicies` and cache‑backed weight stores.

## Documentation

- Docs site: `docs/` (Material for MkDocs). Build with `mkdocs serve`.
- Handy pages: [DSL reference](docs/dsl_reference.md), [Backend matrix](docs/backend_matrix.md), [CLI usage](docs/cli.md).

## Runtime inputs

- Pass runtime tensors: `runner.run(inputs={"Source": array})`. File paths act as defaults.
- Demand mode and Monte‑Carlo projection fall back to NumPy for Torch/JAX today.
- Torch FX exports bake file‑backed defaults; prefer `runner.run` for dynamic feeds. JAX exposes `runner.xla_callable`.
- NumPy supports `ExecutionConfig(fixpoint_strategy="semi_naive")` and optional blocked einsums via `ExecutionConfig(block_size=...)`.

## Backends

- NumPy: reference evaluator with fixpoint forward/backward chaining, recursion, and rich `explain()`.
- Torch FX: graph module backed by `torch.einsum`/NN ops (see `runner.fx_module`).
- JAX: `jax.numpy` executor with an optional XLA callable for `jax.jit` export.

## Examples

```bash
python examples/01_attention_block.py
python examples/02_rules_and_embeddings.py
python examples/03_zero_input_lm.py
```

Artifacts are written into `runs/` under each example folder.

## Contributing

Issues and PRs are welcome! Use Python 3.9+, follow PEP 8, and prefer explicit imports (`from fuse.core.ir import ...`). See `AGENTS.md` for repo conventions. For development:

```bash
pip install -e .[dev]
pip install pre-commit && pre-commit install
pytest -q
```

## License

Apache 2.0 – see `LICENSE`.

## Limitations

- Parser is line‑oriented: no arithmetic, conditionals, or macro system yet.
- Fixpoint mode is synchronous (no semi‑naïve delta joins) so large recursive programs may run slowly.
- Torch/JAX currently embed sources as constants; dynamic data loaders need explicit inputs.
- Policy hooks surface structure but don’t yet include distributed sharding or quant‑aware training loops.
