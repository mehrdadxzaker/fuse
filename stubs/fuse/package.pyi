from pathlib import Path
from typing import Any, Mapping

from .core.ir import FuncCall as FuncCall
from .core.ir import IndexFunction as IndexFunction
from .core.ir import ProgramIR as ProgramIR
from .core.ir import TensorRef as TensorRef
from .core.ir import Term as Term
from .core.policies import ManifestWeightStore as ManifestWeightStore
from .core.policies import RuntimePolicies as RuntimePolicies
from .core.program import Program as Program

def serialize_program_ir(ir: ProgramIR) -> dict[str, Any]: ...
def build_package(program: Program, *, package_path: str | Path, backend: str = 'numpy', device: str = 'auto', inputs: Mapping[str, Any] | None = None, config: Any | None = None, policies: RuntimePolicies | None = None, cache_dir: str | Path | None = None, warm_run: bool = True) -> Path: ...
