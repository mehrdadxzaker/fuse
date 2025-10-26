# Repository Guidelines

## Project Structure & Modules
The Python package lives under `src/fuse/`. `core/` houses the parser/IR plus the shared runtime (`evaluator_numpy.py`), cache helpers, and policy abstractions (weight stores, sharding, quant, LoRA). Backend integrations live in `torch_backend/compile.py` (FX lowering + runner) and `jax_backend/compile.py` (jax.numpy executor + optional XLA callable). Examples in `examples/*.py` pair with `.fuse` programs and read assets from `examples/data` and `examples/ckpt`. Example runs emit artifacts to per-example `runs/` directories; keep large outputs out of version control.

## Setup, Build, and Run
Create a virtual environment and install the project in editable mode:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
Run any example with `python examples/01_attention_block.py`. Use the `.fuse` files as fixtures when writing additional programs; they can be executed via the Python harness or imported by tests. Optional extras: `pip install torch` enables the FX backend, `pip install jax` enables the XLA path.

## Coding Style & Naming
Target Python 3.9+, follow PEP 8, and use 4 spaces per indent—mirror the updated runtime modules (`evaluator_numpy.py`, backend compilers). Functions and modules stay snake_case, classes use PascalCase, and constants are UPPER_SNAKE. Prefer explicit imports from sibling modules (`from fuse.core.ir import ...`) to keep dependency flow obvious. Add concise docstrings where behavior is non-trivial and keep pure parsing logic separate from backend code. When touching execution paths, thread the new `ExecutionConfig`, `RuntimePolicies`, and caching hooks through call sites so NumPy/Torch/JAX stay aligned.

## Testing Guidelines
The repo ships without a bundled suite; place new coverage under `tests/` using pytest-style names (`test_parser_roundtrip.py`). Use fakes from `examples/` to build fixtures and assert on the produced IR or backend outputs. Prefer parametrising over backends (`numpy`, `torch`, `jax`) and guard optional dependencies with `pytest.importorskip`. Run tests with:
```bash
pytest -q
```
When touching execution paths, also rerun the relevant example scripts and confirm `runs/` artifacts update as expected. Cache-heavy changes should be validated by compiling with `Program.compile(..., cache_dir=".cache")` and ensuring replays reuse artifacts.

## Commit & PR Workflow
The distributed snapshot lacks git history, so adopt Conventional Commit style (`feat: add numpy evaluator guard`). Limit commit scope to one logical change, include reproduction or verification steps in the body, and reference issues with `Fixes #123` where applicable. PRs should describe motivation, summarize behavior changes, call out testing performed (commands above), and attach sample outputs when altering generated artifacts.
