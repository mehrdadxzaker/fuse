# How-to Guides

[[toc]]

These recipes cover day-to-day tasks: compiling programs, exporting artifacts, caching, and configuring runtime policies.

## Compile across backends

=== "Python API"

    ```python
    from pathlib import Path

    from fuse import Program

    source = Path("examples/04_mlp.fuse").read_text()
    program = Program(source)

    numpy_runner = program.compile(backend="numpy")
    numpy_runner()
    ```

=== "Torch FX"

    ```python
    from pathlib import Path

    from fuse import Program
    from fuse import torch as fuse_torch

    program = Program(Path("examples/04_mlp.fuse").read_text())
    torch_runner = fuse_torch.compile(program, device="auto")
    torch_runner()
    ```

=== "CLI"

    ```bash
    python -m fuse run examples/04_mlp.fuse --backend numpy --cache .cache
    ```

!!! tip
    Switching between backends is a compile-time choice. The DSL stays identical across engines, so you can validate behaviour under NumPy before deploying with Torch or JAX.

## Export to TorchScript / ONNX

```python
from pathlib import Path

from fuse import Program, to_torchscript, to_onnx

program = Program(Path("examples/04_mlp.fuse").read_text())
script_module = to_torchscript(program)
onxx_model = to_onnx(program)
```

!!! note
    TorchScript export returns a `torch.jit.ScriptModule`. The ONNX helper writes to disk and returns the file path.

## Enable caching

```python
from pathlib import Path

from fuse import Program

program = Program(Path("examples/05_transformer_block.fuse").read_text())
runner = program.compile(backend="numpy", cache_dir=".cache")
runner()
```

Cache directories persist compiled IR and backend artifacts. Reuse the same path across runs to skip recompilation.

## Configure runtime policies

Policies drive weight loading, quantisation, sharding, and LoRA adapters. Mix and match to match your deployment constraints.

```python
from pathlib import Path

from fuse import Program, RuntimePolicies, ManifestWeightStore

program = Program(Path("examples/05_transformer_block.fuse").read_text())
policies = RuntimePolicies(
    weight_store=ManifestWeightStore("examples/ckpt/manifest.json"),
    strict_weights=True,
)
runner = program.compile(backend="numpy", policies=policies)
```

!!! info "Policy debugging"
    Call `policies.describe()` to inspect active rules. During execution, `Program.explain()` captures policy-driven caching events alongside tensor traces.
