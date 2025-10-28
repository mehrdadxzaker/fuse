from pathlib import Path
from typing import Any, Mapping, Sequence

from numpy.typing import NDArray
import torch
from _typeshed import Incomplete

from .core.policies import RuntimePolicies as RuntimePolicies
from .core.program import Program as Program

def from_pytorch(state_dict: Mapping[str, Any], mapping: Mapping[str, Mapping[str, Any]], *, strict: bool = True) -> dict[str, NDArray[Any]]: ...
def from_safetensors(path: str | Path, mapping: Mapping[str, Mapping[str, Any]], *, strict: bool = True) -> dict[str, NDArray[Any]]: ...

class _FuseTorchModule:
    runner: Incomplete
    input_names: Incomplete
    def __init__(self, runner: Any, input_names: Sequence[str]) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

def to_torchscript(program: Program, example_inputs: Mapping[str, Any], *, policies: RuntimePolicies | None = None, device: str = 'auto', config: Any | None = None, file_path: str | Path | None = None) -> torch.jit.ScriptModule: ...
def to_onnx(program: Program, example_inputs: Mapping[str, Any], *, policies: RuntimePolicies | None = None, device: str = 'auto', config: Any | None = None, file_path: str | Path | None = None, opset_version: int = 17, dynamic_axes: dict[str, dict[int, str]] | None = None) -> bytes | Path: ...
