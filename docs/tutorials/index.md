# Tutorials

[[toc]]

These walkthroughs mirror the scripts in `examples/`. Each one demonstrates a single concept and can be executed without extra scaffolding.

## Gallery overview

<div class="grid cards" markdown>

-   :material-function-variant: __Attention block__  
    Self-attention expressed in Fuse with projection helpers.  
    ```bash
    python examples/01_attention_block.py
    ```
-   :material-transit-connection: __Routing MLP__  
    Shows conditional execution across experts.  
    ```bash
    python -m fuse run examples/04_mlp.fuse --backend numpy
    ```
-   :material-account-tree: __Tree programs__  
    Compose hierarchical programs via `tree_program`.  
    ```bash
    python examples/08_tree_program.py
    ```
-   :material-alpha-d-box: __Custom operators__  
    Extend the DSL by injecting Python callables at runtime.  
    ```bash
    python examples/11_custom_operator.py
    ```

</div>

!!! info "Need optional backends?"
    Install `fuse[torch]` or `fuse[jax]` to enable the Torch FX and JAX runners. Examples automatically fall back to NumPy when a backend is unavailable.

## Run a single tutorial

The CLI exposes the same ergonomics as the Python API. Use `--backend` to switch engines.

```bash
python -m fuse run examples/05_transformer_block.fuse --backend numpy
```

Add `--explain` to emit execution traces:

```bash
python -m fuse run examples/05_transformer_block.fuse --backend numpy --explain runs/transformer_trace.json
```

## Batch execution helper

The gallery includes a batch runner that executes multiple programs and collects metrics.

```bash
python examples/run_new_architectures.py --backend numpy --out runs/gallery.jsonl
```

!!! tip
    Pass `--backend torch` or `--backend jax` once you have the extras installed. Failed runs raise immediately so CI can catch regressions.

## Explore the code

Each tutorial script highlights a distinct capability:

| Script | Highlights |
| --- | --- |
| `01_attention_block.py` | Attention kernel expressed in the DSL with gradients and explainers. |
| `03_runtime_policies.py` | Demonstrates manifest-backed weights and caching policies. |
| `05_transformer_block.fuse` | Raw DSL program executed through the CLI. |
| `08_tree_program.py` | Builds a tree of programs for recursive inference. |
| `11_custom_operator.py` | Registers a custom Python callable and wires it into the graph. |

Browse the [examples directory](https://github.com/mehrdadxzaker/fuse/tree/main/examples) for the full list.
