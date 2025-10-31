# Fuse CLI

The CLI wraps the Python API for quick experiments and CI smoke tests.

## Usage

```text
python -m fuse run PROGRAM [--backend numpy|torch|jax] [--out PATH] [--cache PATH]
```

* `PROGRAM` — path to the `.fuse` file containing your equations.
* `--backend` — execution backend (`numpy`, `torch`, or `jax`). Defaults to `numpy`.
* `--out` — optional output file. Supports `.npy`, `.npz`, `.json`, and `.jsonl`.
* `--cache` — reuse compiled artifacts by pointing to a directory.

!!! example "Run a program"
    ```bash
    python -m fuse run examples/05_transformer_block.fuse --backend numpy --out runs/transformer.json
    ```

## Output behaviour

* Programs must declare at least one `export` statement.
* When `--out` is omitted, Fuse prints the first exported tensor to stdout.
* Multiple exports without `--out` emit the full tensor map as pretty-printed JSON.

!!! info "Backend availability"
    Torch and JAX backends are optional. If the frameworks are not installed, Fuse falls back to NumPy automatically.

!!! note "Python API default"
    When using the Python API, `Program.compile()` now defaults to `backend="auto"`, which picks a backend based on hardware and workload (e.g., attention/MLP on CUDA/MPS tends to select Torch; streaming, demand mode, or Monte Carlo projection select NumPy). The CLI keeps an explicit `--backend` flag with a default of `numpy`.

This utility is intended for CI smoke tests and ad-hoc experimentation; for integrated applications prefer the Python API.
