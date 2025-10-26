from dataclasses import dataclass
from typing import Any

import numpy as np
from _typeshed import Incomplete

from .builtins import SparseBoolTensor as SparseBoolTensor
from .builtins import attention as attention
from .builtins import causal_mask as causal_mask
from .builtins import concat as concat
from .builtins import const as const
from .builtins import gelu as gelu
from .builtins import gelu_grad as gelu_grad
from .builtins import layernorm as layernorm
from .builtins import lnorm as lnorm
from .builtins import masked_softmax as masked_softmax
from .builtins import read_tensor_from_file as read_tensor_from_file
from .builtins import reduce_max as reduce_max
from .builtins import reduce_mean as reduce_mean
from .builtins import relu as relu
from .builtins import rope as rope
from .builtins import sig as sig
from .builtins import softmax as softmax
from .builtins import softmax_grad as softmax_grad
from .builtins import step as step
from .builtins import topk as topk
from .builtins import tucker_dense as tucker_dense
from .builtins import write_tensor_to_file as write_tensor_to_file
from .ir import Equation as Equation
from .ir import FuncCall as FuncCall
from .ir import IndexFunction as IndexFunction
from .ir import ProgramIR as ProgramIR
from .ir import TensorRef as TensorRef
from .ir import Term as Term
from .ir import equation_index_summary as equation_index_summary
from .ir import format_index_summary as format_index_summary
from .policies import RuntimePolicies as RuntimePolicies
from .stats import compute_einsum_stats as compute_einsum_stats
from .temperature import TemperatureSchedule as TemperatureSchedule
from .temperature import coerce_temperature_value as coerce_temperature_value
from .temperature import normalize_temperature_map as normalize_temperature_map

EINSUM_LABELS: Incomplete

class _StreamRingStore:
    axes: Incomplete
    window_sizes: Incomplete
    window_arr: Incomplete
    initialized: bool
    origin: Incomplete
    max_pos: Incomplete
    ring_to_value: dict[tuple[int, ...], np.ndarray]
    ring_to_global: dict[tuple[int, ...], tuple[int, ...]]
    global_to_ring: dict[tuple[int, ...], tuple[int, ...]]
    def __init__(self, axes: tuple[str, ...], window_sizes: tuple[int, ...]) -> None: ...
    def store(self, key: tuple[int, ...], value: np.ndarray, *, update_max: bool) -> None: ...
    def get(self, key: tuple[int, ...]) -> np.ndarray | None: ...
    def latest(self, positions: dict[str, int]) -> np.ndarray | None: ...
    def prune_before(self, minimums: tuple[int, ...]) -> None: ...
    def max_position_map(self) -> dict[str, int]: ...

@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = ...
    fixpoint_strategy: str = ...
    max_iters: int = ...
    tol: float = ...
    chaining: str = ...
    explain_timings: bool = ...
    projection_strategy: str = ...
    projection_samples: int | None = ...
    projection_seed: int | None = ...
    temperatures: dict[str, TemperatureSchedule] | None = ...
    precision: str = ...
    device: str = ...
    zero_copy: bool = ...
    jax_enable_xla_cache: bool = ...
    jax_cache_dir: str | None = ...
    validate_device_transfers: bool = ...
    block_size: int | None = ...
    def normalized(self) -> ExecutionConfig: ...

class NumpyRunner:
    ir: Incomplete
    config: Incomplete
    policies: Incomplete
    tensors: dict[str, Any]
    index_domains: dict[str, int]
    logs: list[dict[str, Any]]
    boolean_tensors: set[str]
    stream_enabled: Incomplete
    stream_axes: set[str]
    stream_axis_min_offset: dict[str, int]
    tensor_stream_axes: dict[str, tuple[str, ...]]
    tensor_static_indices: dict[str, tuple[str, ...]]
    stream_storage: dict[str, _StreamRingStore]
    stream_positions: dict[str, int]
    def __init__(self, ir: ProgramIR, config: ExecutionConfig | None = None, policies: RuntimePolicies | None = None) -> None: ...
    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        config: ExecutionConfig | None = None,
        policies: RuntimePolicies | None = None,
    ) -> dict[str, Any]: ...
    def run(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        config: ExecutionConfig | None = None,
        policies: RuntimePolicies | None = None,
    ) -> dict[str, Any]: ...
    def explain(self, *, json: bool = False) -> dict[str, Any] | str: ...
    def temperature_manifest(self) -> dict[str, Any]: ...

class DemandNumpyRunner(NumpyRunner):
    def __init__(self, ir: ProgramIR, config: ExecutionConfig | None = None, policies: RuntimePolicies | None = None) -> None: ...
    def run(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        config: ExecutionConfig | None = None,
        policies: RuntimePolicies | None = None,
    ) -> dict[str, Any]: ...
    def query(
        self,
        name: str,
        selectors: dict[str, Any] | None = None,
        *,
        inputs: dict[str, Any] | None = None,
        config: ExecutionConfig | None = None,
        policies: RuntimePolicies | None = None,
    ) -> np.ndarray: ...
    def reset(self) -> None: ...
