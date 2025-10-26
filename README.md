# Fuse (starter) — a tensor-equation DSL as a library

**Fuse** expresses AI as a small set of **tensor equations**: *joins* + *projection* (+ optional nonlinearity).
This version ships a NumPy execution engine, a Torch FX lowering, and a JAX path that can be JIT-compiled to XLA.
It keeps the code small and readable while covering end-to-end flows (sources → IR → backends → sinks).

> Paper context: logical rules ≙ Einstein sums; anything not on the LHS is implicitly **projected** (summed) out;
> equations with the same LHS implicitly add; files are first-class **sources/sinks** (reading/writing tensors).

## Install

### Pip

```bash
pip install fuse-ai
pip install fuse-ai[torch]   # Torch FX backend
pip install fuse-ai[jax]     # JAX backend
pip install fuse-ai[bench]   # Torch + JAX bundle for benchmarks
pip install fuse-ai[dev]     # Linting, typing, tests
```

### Editable (local development)

```bash
pip install -e ".[dev]"
```

### Using uv

The repository ships a `uv.lock` for reproducible envs. To sync the dev environment:

```bash
uv sync --extra dev
```

You can swap in other extras (e.g., `--extra torch --extra jax` or `--all-extras`) to mirror the `pip install fuse-ai[...]` flows above.

## Linting & formatting

Install [pre-commit](https://pre-commit.com) and enable the hooks to run Ruff (formatting + lint), pyupgrade, and import sorting automatically:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Runtime inputs

- All backends accept source tensors at call time via `runner.run(inputs={"Source": array})`; file paths remain as defaults when no runtime tensors are provided.
- Torch and JAX backends currently defer demand-driven execution (`mode="demand"`) and Monte-Carlo projection to the NumPy engine, falling back to the reference evaluator in those configurations.
- Torch FX exports keep file-backed defaults baked into the traced graph today; prefer `runner.run` when you need to feed dynamic tensors. The JAX `runner.xla_callable` accepts pytrees of runtime inputs.
- The NumPy runner now supports `ExecutionConfig(fixpoint_strategy="semi_naive")` for delta-driven fixpoint scheduling plus optional blocked einsums via `ExecutionConfig(block_size=...)`.

## Run examples

```bash
python examples/01_attention_block.py
python examples/02_rules_and_embeddings.py
python examples/03_zero_input_lm.py
```

Artifacts (sources, IR, simple plans, outputs) are written into `runs/` under each example’s folder.

[Additional documentation](docs/)

| Topic | Description |
| ----- | ----------- |
| [DSL reference](docs/dsl_reference.md) | One-page grammar & operator cheatsheet. |
| [Backend matrix](docs/backend_matrix.md) | Backend capabilities and constraints at a glance. |
| [CLI usage](docs/cli.md) | Running programs quickly via `python -m fuse run`. |

## What’s here

- A `Program` that parses a **compact Fuse DSL**:
  - Lines: `T[i,j] = A[i,k] B[k,j]`, `Y[i.] = softmax(X[i])`, axis-aware `concat`, `amax/avg` projections, and literal/keyword arguments.
  - Sources: `T[i,j] = "file.npy"`, `"out.jsonl" = T[i,j]`, text/CSV autoloaders, plus pluggable weight stores via `RuntimePolicies`.
- Execution engines
  - **NumPy** runner with fixpoint forward/backward chaining, recursion, enhanced `explain()` (einsum canonicalization, projected indices, timing).
  - **Torch FX** backend that emits a graph module backed by `torch.einsum`/NN ops (access via `runner.fx_module`), honouring caching and policies.
  - **JAX** backend that evaluates with `jax.numpy` and exposes a lazily-built `runner.xla_callable` for `jax.jit` export.
- Execution controls via `ExecutionConfig`
  - `precision` defaults to `fp32`. Mixed-precision runs can request `bf16`, `fp16`, or `auto` (which selects the fastest supported dtype per backend/device). NumPy always stays in `fp32`; Torch refuses `fp16` on CPU and checks CUDA/MPS support; JAX only permits `fp16` on GPU and maps TPU/GPU `auto` runs to `bf16`.
  - `device` chooses where execution happens: `auto`, `cpu`, `cuda[:index]`, or `mps`. NumPy can only target CPU; Torch/JAX resolve and pin all compilation artifacts to the requested accelerator so FX graphs and XLA lowers stay aligned.
  - `zero_copy` keeps host↔device hand-offs lean. When `True` (default) the runners reuse host buffers on CPU and skip redundant `.tolist()` conversions; set `False` if you need defensive copies before handing tensors to external code.
  - For JAX you can opt into the experimental XLA cache with `ExecutionConfig(jax_enable_xla_cache=True, jax_cache_dir="~/.cache/fuse/jax")` (path optional) and grab the lazily-built `runner.xla_callable` for `jax.jit` execution.
  - `validate_device_transfers=True` raises if GPU/TPU runs would implicitly copy NumPy inputs to device memory, forcing explicit `jax.device_put` hand-offs when you want to audit data movement.
- Quantised weights retain scale/zero-point metadata. During dequantisation we enforce float32 accumulation (at least fp16) and broadcast-compatible shapes; values are assumed to be pre-saturated/rounded into the int8 range, so Fuse only rescales without introducing extra clipping.
- Caching and policies
  - `Program.compile(..., cache_dir="path")` stores backend artifacts via `CacheManager`.
  - `RuntimePolicies` captures weight stores, sharding metadata, quantisation (e.g. `int8` dequant) and LoRA adapter rules.
- Examples:
  1. Masked attention in ~3 equations.
  2. A rule (`Aunt`) + reasoning in embedding space with a temperature knob `T`.
  3. Zero-input LM head: sources/sinks live in the program; just run the artifact.

## Limitations

- Parser is still line-oriented: no arithmetic, conditionals, or macro system yet.
- Fixpoint mode is synchronous (no semi-naïve delta joins) so large recursive programs may run slowly.
- Torch/JAX backends embed sources as constants; dynamic data loaders will need additional plumbed inputs.
- Policy hooks surface structure but do not yet include end-to-end distributed sharding or quant-aware training loops.

This remains a **starter** that you can extend toward production backends.
