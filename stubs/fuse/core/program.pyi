from typing import Any

from _typeshed import Incomplete

from .cache import CacheManager as CacheManager
from .cache import build_cache_key as build_cache_key
from .cache import compute_program_hash as compute_program_hash
from .evaluator_numpy import DemandNumpyRunner as DemandNumpyRunner
from .evaluator_numpy import ExecutionConfig as ExecutionConfig
from .evaluator_numpy import NumpyRunner as NumpyRunner
from .exceptions import BackendError as BackendError
from .ir import ProgramIR as ProgramIR
from .ir import TensorRef as TensorRef
from .ir import equation_index_summary as equation_index_summary
from .ir import format_index_summary as format_index_summary
from .ir import json_ready as json_ready
from .ir import lhs_indices as lhs_indices
from .ir import rhs_indices as rhs_indices
from .parser import parse as parse
from .policies import RuntimePolicies as RuntimePolicies
from .shape_checker import validate_program_shapes as validate_program_shapes

class Program:
    src: Incomplete
    ir: ProgramIR
    digest: Incomplete
    def __init__(self, eqs: str) -> None: ...
    def compile(
        self,
        backend: str = ...,
        device: str = ...,
        cache_dir: str | None = None,
        config: ExecutionConfig | None = None,
        execution: str | None = None,
        policies: RuntimePolicies | None = None,
        **backend_kwargs: Any,
    ) -> NumpyRunner: ...
    def explain(self, *, json: bool = False) -> Any: ...
