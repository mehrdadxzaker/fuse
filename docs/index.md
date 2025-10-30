# Fuse

> A focused tensor-equation DSL that keeps compilation, execution, and tooling within reach of a single team.

Fuse helps you describe tensor programs with clear algebraic notation, compile them against NumPy, and optionally lower to Torch FX or JAX. The runtime embraces readability—every stage from parsing to execution lives in Python so you can inspect, extend, and ship quickly.

<div class="hero-grid" markdown>

-   :material-lightning-bolt: **Batteries-included runtime**  
    Parser, IR, schedulers, and evaluators share a compact code base with first-class NumPy support and optional Torch/JAX paths.
-   :material-compass: **Policy-driven execution**  
    Cache helpers, weight stores, sharding, quantisation, and LoRA adapters are surfaced through friendly Python APIs.
-   :material-chart-timeline: **Explainable by default**  
    The evaluator ships with `explain()` traces, shape checks, and gradient builders so you can reason about every equation.

</div>

## Quick start checklist

1. Install Fuse and (optionally) your preferred backends.
2. Compile a program from the example gallery.
3. Inspect outputs and iteration traces to validate behaviour.

<div class="grid cards" markdown>

-   :material-rocket-launch: __Get started__  
    Step-by-step environment setup and first execution.  
    [:octicons-arrow-right-16: Follow the guide](get-started.md)
-   :material-teach: __Run the gallery__  
    Hands-on walkthroughs for the shipped `.fuse` examples.  
    [:octicons-arrow-right-16: Tutorials](tutorials/index.md)
-   :material-cog: __Build integrations__  
    Export, cache, and configure runtime policies for production.  
    [:octicons-arrow-right-16: How-to guides](how-to/index.md)
-   :material-book-open-page-variant: __Understand the DSL__  
    Grammar reference, backend matrix, and API docs.  
    [:octicons-arrow-right-16: Concepts & Reference](concepts/index.md)

</div>

!!! tip "Prefer examples you can copy and paste"
    All command and code snippets on this site are copy-ready—no shell prompts—so you can drop them straight into a terminal or notebook.

## When to reach for Fuse

* You want an expressive DSL that still compiles back to friendly Python objects.
* You need deterministic execution with the option to toggle demand/fixpoint modes per backend.
* You care about explainability and introspection as much as peak throughput.

## Related pages

* [Get Started](get-started.md)
* [How-to Guides](how-to/index.md)
* [Backend Support Matrix](backend_matrix.md)
* [API Reference](reference/index.md)
