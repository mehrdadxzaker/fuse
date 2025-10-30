# How‑to Guides

- Compile across backends (NumPy/Torch/JAX)
- Export to TorchScript / ONNX
- Use caching and artifacts
- Configure runtime policies (weights, sharding, quant, LoRA)

## Compile across backends

```python
from fuse import Program
from fuse import torch as fuse_torch

prog = Program(open("examples/04_mlp.fuse").read())

# NumPy
runner = prog.compile(backend="numpy")
runner()

# Torch FX
t_runner = fuse_torch.compile(prog, device="auto")
t_runner()
```

## Export to TorchScript / ONNX

```python
from fuse import to_torchscript, to_onnx

ts = to_torchscript(prog)  # torch.jit.ScriptModule
onnx = to_onnx(prog)       # writes a basic ONNX graph
```

## Caching

```python
runner = prog.compile(backend="numpy", cache_dir=".cache")
runner()
```

## Runtime policies (weights, quant, LoRA)

```python
from fuse import RuntimePolicies, ManifestWeightStore

pol = RuntimePolicies(weight_store=ManifestWeightStore("examples/ckpt/manifest.json"))
runner = prog.compile(backend="numpy", policies=pol)
```

