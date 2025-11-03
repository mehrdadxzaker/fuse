# Concepts

[[toc]]

Understand how Fuse models tensor programs, how execution policies work, and where each backend fits.

## Program, IR, and evaluation

Fuse parses `.fuse` files into an intermediate representation (IR) consisting of equations, sources, sinks, and exports. Programs compile down to backend-specific runners.

* **Parser** — Converts the DSL into a structured IR, validating syntax and index semantics.
* **IR** — Captures tensor expressions, projections, and metadata needed for execution.
* **Evaluators** — Execute the IR using NumPy (always available) or optional Torch/JAX integrations.

Refer to [`fuse.core.ir`](../reference/fuse/core/ir/index.md) for IR data structures.

## Execution configuration

`ExecutionConfig` objects let you control precision, device selection, demand/fixpoint modes, and Monte Carlo projection strategies.

!!! example "Configure execution"
    ```python
    from pathlib import Path

    from fuse import ExecutionConfig, Program

    program = Program(Path("examples/01_attention_block.fuse").read_text())
    config = ExecutionConfig(device="cpu", demand_mode=True)
    runner = program.compile(backend="numpy", config=config)
    ```

Policies supplement configuration:

* **RuntimePolicies** — Compose caching behaviour, manifest weight stores, and quantisation options.
* **Temperature schedules** — Drive annealing-style loops with `make_schedule` helpers.
* **Explainability hooks** — `Program.explain()` emits structured traces per iteration.

## Backends at a glance

All backends share the parser, IR, caching, and policy layers. Key differences lie in execution engines:

* **NumPy evaluator** — Reference implementation with complete DSL coverage.
* **Torch FX backend** — Lowers to `torch.fx.GraphModule` objects and integrates with `torch.compile`.
* **JAX backend** — Emits pure JAX functions with optional XLA compilation.

See the [backend matrix](../backend_matrix.md) for feature-by-feature coverage.

## DSL building blocks

The DSL is designed to resemble algebraic equations while supporting tensor-specific operations:

* Equations with projections (`=`, `+=`, `max=`, `avg=`) define computation graphs.
* Sources map filenames to tensors; sinks export results.
* Boolean tensors use parentheses (`Fact(i, j)`), and dotted indices indicate reduction axes.
* Builtins cover activation functions, attention, concatenation, and reduction helpers.

Consult the [DSL reference](../dsl_reference.md) for the full grammar and examples.

## Migration to structured syntax (v2)

The legacy parser remains the default to keep existing programs working. You can opt into a structured grammar that adds expressions and blocks by constructing `Program` with `parser='v2'`. See the [migration guide](migration.md) for an incremental, no‑rewrite path.
