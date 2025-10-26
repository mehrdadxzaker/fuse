from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from .core.policies import RuntimePolicies
from .core.program import Program

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.copy()
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        try:
            return tensor.numpy().copy()
        except RuntimeError:
            return np.asarray(tensor.tolist())
    return np.asarray(value).copy()


def _load_spec_tensor(
    data: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    strict: bool,
) -> np.ndarray:
    key = spec.get("key")
    if key is None:
        raise ValueError("Mapping specification must provide 'key'")
    if key not in data:
        if strict:
            raise KeyError(f"Key '{key}' not found in source mapping")
        return None  # type: ignore[return-value]
    tensor = data[key]
    arr = _to_numpy(tensor)
    if spec.get("transpose", False):
        if arr.ndim != 2:
            raise ValueError("transpose=True expects a 2-D tensor")
        arr = arr.T
    source_axes = spec.get("source_axes")
    target_axes = spec.get("target_axes")
    perm = spec.get("perm")
    order = spec.get("order")
    if perm is not None:
        arr = np.transpose(arr, axes=list(perm))
    elif order is not None:
        if source_axes is None:
            raise ValueError("Spec with 'order' must define 'source_axes'")
        if len(order) != len(source_axes):
            raise ValueError("'order' must list the same number of axes as source")
        arr = np.transpose(arr, axes=[source_axes.index(axis) for axis in order])
    elif source_axes is not None and target_axes is not None:
        if len(source_axes) != arr.ndim:
            raise ValueError(f"Source axes {source_axes} do not match tensor rank {arr.ndim}")
        if len(target_axes) != len(source_axes):
            raise ValueError("Target axes must match source axes length")
        if set(source_axes) == set(target_axes):
            permuted = [source_axes.index(axis) for axis in target_axes]
            arr = np.transpose(arr, axes=permuted)
        # If the sets differ we treat this as a pure rename without reordering.
    dtype = spec.get("dtype")
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def from_pytorch(
    state_dict: Mapping[str, Any],
    mapping: Mapping[str, Mapping[str, Any]],
    *,
    strict: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert a PyTorch state dict into Fuse weight tensors with named-axis remapping.

    Parameters
    ----------
    state_dict:
        Mapping of parameter names to tensors (typically ``torch.Tensor``).
    mapping:
        Mapping from desired Fuse tensor names to a specification dictionary.
        Each specification supports:

        - ``key`` (str): source key in the state dict (required)
        - ``source_axes`` (Sequence[str]): axis names describing the source order
        - ``target_axes`` (Sequence[str]): desired axis order for Fuse tensors
        - ``transpose`` (bool): optional convenience flag for 2-D transpose
        - ``dtype`` (np.dtype or str): optional dtype cast
    strict:
        If True, missing keys raise an error. Otherwise they are skipped.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of Fuse tensor names to NumPy arrays in the requested axis order.
    """

    weights: Dict[str, np.ndarray] = {}
    for target_name, spec in mapping.items():
        arr = _load_spec_tensor(state_dict, spec, strict=strict)
        if arr is None:
            continue
        weights[target_name] = arr
    return weights


def from_safetensors(
    path: Union[str, Path],
    mapping: Mapping[str, Mapping[str, Any]],
    *,
    strict: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load weights from a ``.safetensors`` file with named-axis remapping.
    """
    try:
        from safetensors.numpy import load_file as load_safetensors  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("safetensors is required for from_safetensors") from exc

    data = load_safetensors(str(path))
    return from_pytorch(data, mapping, strict=strict)


def _ensure_torch_available():
    if torch is None:  # pragma: no cover - torch optional
        raise RuntimeError("PyTorch is required for export operations")


def _ensure_runtime_policies(policies: Optional[RuntimePolicies]) -> RuntimePolicies:
    return policies or RuntimePolicies()


if torch is not None:  # pragma: no cover - defined only when torch is available

    class _FuseTorchModule(torch.nn.Module):  # type: ignore[misc]
        def __init__(self, runner, input_names: Sequence[str]):
            super().__init__()
            self.runner = runner
            self.input_names = list(input_names)

        def forward(self, *args):
            tensor_inputs: Dict[str, Any] = {}
            for name, value in zip(self.input_names, args):
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value, device=self.runner.device)
                else:
                    value = value.to(self.runner.device)
                tensor_inputs[name] = value
            cfg = replace(self.runner.config, mode="single")
            outputs = self.runner.run(inputs=tensor_inputs, config=cfg, skip_sinks=True)
            return tuple(outputs[name] for name in self.runner.ir.exports)

else:  # pragma: no cover - fallback when torch is absent
    _FuseTorchModule = None  # type: ignore[assignment]


def _prepare_example_tensors(
    example_inputs: Mapping[str, Any],
    device: Any,
) -> Tuple[List[str], List["torch.Tensor"]]:
    input_names: List[str] = []
    tensors: List["torch.Tensor"] = []
    for name, value in example_inputs.items():
        input_names.append(name)
        if isinstance(value, torch.Tensor):
            tensors.append(value.to(device))
        else:
            tensors.append(torch.as_tensor(value, device=device))
    return input_names, tensors


def to_torchscript(
    program: Program,
    example_inputs: Mapping[str, Any],
    *,
    policies: Optional[RuntimePolicies] = None,
    device: str = "auto",
    config: Optional[Any] = None,
    file_path: Optional[Union[str, Path]] = None,
) -> "torch.jit.ScriptModule":
    """
    Trace a Fuse program into a TorchScript module using example inputs.
    """
    _ensure_torch_available()
    runtime_policies = _ensure_runtime_policies(policies)
    runner = program.compile(
        backend="torch",
        device=device,
        config=config,
        policies=runtime_policies,
    )
    input_names, tensors = _prepare_example_tensors(example_inputs, runner.device)
    module = _FuseTorchModule(runner, input_names)
    module.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(module, tuple(tensors), strict=False)
    if file_path is not None:
        scripted.save(str(file_path))
    return scripted


def to_onnx(
    program: Program,
    example_inputs: Mapping[str, Any],
    *,
    policies: Optional[RuntimePolicies] = None,
    device: str = "auto",
    config: Optional[Any] = None,
    file_path: Optional[Union[str, Path]] = None,
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> Union[bytes, Path]:
    """
    Export a Fuse program to ONNX using PyTorch tracing.
    """
    _ensure_torch_available()
    runtime_policies = _ensure_runtime_policies(policies)
    runner = program.compile(
        backend="torch",
        device=device,
        config=config,
        policies=runtime_policies,
    )
    input_names, tensors = _prepare_example_tensors(example_inputs, runner.device)
    module = _FuseTorchModule(runner, input_names)
    module.eval()

    output_names = list(program.ir.exports)

    kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset_version,
        "dynamic_axes": dynamic_axes or {},
    }

    if file_path is None:
        buffer = io.BytesIO()
        torch.onnx.export(
            module,
            tuple(tensors),
            buffer,
            **kwargs,
        )
        buffer.seek(0)
        return buffer.getvalue()

    torch.onnx.export(
        module,
        tuple(tensors),
        str(file_path),
        **kwargs,
    )
    return Path(file_path)


__all__ = [
    "from_pytorch",
    "from_safetensors",
    "to_torchscript",
    "to_onnx",
]
