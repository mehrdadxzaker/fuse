# Get Started

## Install

- Python 3.9+
- Optional: Torch (`pip install fuse-ai[torch]`) and/or JAX (`pip install fuse-ai[jax]`)

Using a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or from PyPI:

```bash
pip install fuse-ai
# Optional backends
pip install fuse-ai[torch]
pip install fuse-ai[jax]
```

## First run

- Run an example:

```bash
python examples/01_attention_block.py
```

- Or execute a `.fuse` program via the CLI:

```bash
python -m fuse run examples/05_transformer_block.fuse --backend numpy
```

## Next steps

- Browse the Tutorials to run the full example gallery
- See How‑to guides for exporting (TorchScript/ONNX), caching, and policies
- Explore Concepts for the DSL reference and backend capabilities

