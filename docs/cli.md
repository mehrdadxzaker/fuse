# Fuse CLI Cheatsheet

Fuse now includes a minimal command-line runner for quick experiments:

```bash
python -m fuse run program.fuse --backend numpy --out out.npy
```

## Usage

```text
usage: python -m fuse run PROGRAM [--backend numpy|torch|jax] [--out PATH]
```

* `PROGRAM` — path to the `.fuse` file containing your equations.
* `--backend` — execution backend (`numpy`, `torch`, or `jax`). Defaults to
  `numpy`.
* `--out` — optional output file. Supports `.npy`, `.npz`, `.json`, and
  `.jsonl`. If omitted, the runner prints the first exported tensor to stdout.

The runner expects the program to declare at least one `export` statement. When
multiple exports are present and `--out` is omitted, the entire output map is
rendered as indented JSON.

Backend-specific notes:

* Torch and JAX require the respective frameworks installed and fall back to
  the NumPy engine when unavailable.
* `.npy/.npz` outputs honour `RuntimePolicies` (e.g., memory mapped sources) via
  the standard program compilation path.

This utility is intended for CI smoke tests and ad-hoc experimentation; for
integrated applications prefer the Python API.
