# Get Started

[[toc]]

Fuse targets Python 3.9+ and ships optional extras for Torch and JAX backends. The steps below keep environments reproducible and ensure every snippet stays copy/pasteable.

## 1. Choose your installation path

!!! note "Use a clean environment"
    We recommend creating a fresh virtual environment per project—either `venv`, Conda, or a tool like `uv`. This isolates dependencies and mirrors the CI setup.

=== "From PyPI"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install fuse
    ```

    Install optional backends as needed:

    ```bash
    pip install "fuse[torch]"
    pip install "fuse[jax]"
    ```

=== "Editable checkout"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    git clone https://github.com/mehrdadxzaker/fuse.git
    cd fuse
    pip install -e ".[dev]"
    ```

    The `dev` extras pull in MkDocs tooling, Ruff, MyPy, and pytest so you can run the full CI suite locally.

## 2. Verify the installation

Use the shipped examples to confirm the runtime works end-to-end.

```bash
python examples/01_attention_block.py
```

Expect logs describing compilation and execution along with saved artifacts under `examples/runs/`.

!!! tip "Prefer the CLI for quick experiments"
    ```bash
    python -m fuse run examples/05_transformer_block.fuse --backend numpy
    ```
    Swap `--backend` to `torch` or `jax` once the optional extras are installed. The runner prints the exported tensors or writes them to disk with `--out`.

## 3. Explore next steps

<div class="grid cards" markdown>

-   __Inspect execution__  
    Call [`Program.explain()`](reference/fuse/core/program/index.md) to capture intermediate tensors and fixpoint iterations.
-   __Enable caching__  
    Reuse compiled graphs by passing `cache_dir=".cache"` to `Program.compile`.
-   __Extend policies__  
    Configure `RuntimePolicies` for sharding, quantisation, LoRA adapters, and manifest-backed weight stores.

</div>

## Troubleshooting checklist

??? info "Common fixes"
    - Ensure the virtual environment is activated when installing extras.
    - Re-run `pip install -e ".[dev]"` after pulling new dependencies.
    - Use `pip install --upgrade pip` if installers complain about old wheels.
    - Optional backends are lazy-imported; missing Torch/JAX simply fall back to NumPy.

Need help? Swing by the [Community page](community.md) for contribution guidelines and support channels.
