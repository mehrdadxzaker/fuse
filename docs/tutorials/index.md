# Tutorials

This section walks through running the example programs under `examples/`.

## Run the gallery (NumPy backend)

```bash
python examples/run_new_architectures.py --backend numpy
```

Artifacts are written under `examples/runs/`.

## Run a single program

```bash
python -m fuse run examples/04_mlp.fuse --backend numpy
```

Use `--backend torch` or `--backend jax` if the optional backends are installed.

