from __future__ import annotations

import string
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .builtins import (
    SparseBoolTensor,
    attention,
    causal_mask,
    concat,
    const,
    gelu,
    gelu_grad,
    layernorm,
    lnorm,
    masked_softmax,
    read_tensor_from_file,
    reduce_max,
    reduce_mean,
    relu,
    rope,
    sig,
    softmax,
    softmax_grad,
    step,
    topk,
    tucker_dense,
    write_tensor_to_file,
)
from .ir import (
    Equation,
    FuncCall,
    IndexFunction,
    ProgramIR,
    TensorRef,
    Term,
    equation_index_summary,
    format_index_summary,
)
from .policies import RuntimePolicies
from .stats import compute_einsum_stats
from .temperature import (
    TemperatureSchedule,
    coerce_temperature_value,
    normalize_temperature_map,
)

# Map index tokens to letters for einsum (limited alphabet for demo)
EINSUM_LABELS = list(string.ascii_letters)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    return str(value)


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, SparseBoolTensor):
        return value.to_dense()
    return np.asarray(value)


class _StreamRingStore:
    def __init__(self, axes: Tuple[str, ...], window_sizes: Tuple[int, ...]):
        self.axes = axes
        self.window_sizes = tuple(max(1, int(w)) for w in window_sizes)
        self.window_arr = (
            np.array(self.window_sizes, dtype=np.int64)
            if self.axes
            else np.array([], dtype=np.int64)
        )
        self.initialized = False
        self.origin = np.zeros(len(self.axes), dtype=np.int64)
        self.max_pos = np.zeros(len(self.axes), dtype=np.int64)
        self.ring_to_value: Dict[Tuple[int, ...], np.ndarray] = {}
        self.ring_to_global: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        self.global_to_ring: Dict[Tuple[int, ...], Tuple[int, ...]] = {}

    def _tuple_key(self, key: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(int(k) for k in key)

    def store(self, key: Tuple[int, ...], value: np.ndarray, *, update_max: bool) -> None:
        key = self._tuple_key(key)
        if not self.axes:
            ring_key: Tuple[int, ...] = ()
            existing = self.ring_to_value.get(ring_key)
            if existing is None or existing.shape != value.shape or existing.dtype != value.dtype:
                self.ring_to_value[ring_key] = np.array(value, copy=True)
            else:
                np.copyto(existing, value, casting="unsafe")
            self.ring_to_global[ring_key] = ()
            self.global_to_ring[()] = ring_key
            self.initialized = True
            return

        key_arr = np.array(key, dtype=np.int64)
        if not self.initialized:
            self.origin = key_arr.copy()
            self.max_pos = key_arr.copy()
            self.initialized = True
        else:
            self.max_pos = np.maximum(self.max_pos, key_arr)

        desired_origin = self.max_pos - self.window_arr + 1
        if desired_origin.size:
            self.origin = np.maximum(self.origin, desired_origin)
        self.origin = np.minimum(self.origin, key_arr)

        offset = key_arr - self.origin
        shift = np.maximum(offset - self.window_arr + 1, 0)
        if shift.size and np.any(shift > 0):
            self.origin += shift
            offset = key_arr - self.origin

        min_allowed = tuple(int(o) for o in self.origin)
        for global_key in list(self.global_to_ring.keys()):
            if any(g < min_allowed[idx] for idx, g in enumerate(global_key)):
                ring = self.global_to_ring.pop(global_key)
                self.ring_to_global.pop(ring, None)
                self.ring_to_value.pop(ring, None)

        ring_key = tuple(
            int((offset[idx]) % self.window_sizes[idx]) for idx in range(len(self.axes))
        )
        existing_global = self.ring_to_global.get(ring_key)
        if existing_global is not None and existing_global != key:
            self.global_to_ring.pop(existing_global, None)

        existing_value = self.ring_to_value.get(ring_key)
        if (
            existing_value is None
            or existing_value.shape != value.shape
            or existing_value.dtype != value.dtype
        ):
            self.ring_to_value[ring_key] = np.array(value, copy=True)
        else:
            np.copyto(existing_value, value, casting="unsafe")
        self.ring_to_global[ring_key] = key
        self.global_to_ring[key] = ring_key

    def get(self, key: Tuple[int, ...]) -> Optional[np.ndarray]:
        key = self._tuple_key(key)
        ring = self.global_to_ring.get(key)
        if ring is None:
            return None
        return self.ring_to_value.get(ring)

    def latest(self, positions: Dict[str, int]) -> Optional[np.ndarray]:
        if not self.axes:
            return self.ring_to_value.get(())
        if not self.initialized:
            return None
        key = tuple(
            int(positions.get(axis, self.max_pos[idx])) for idx, axis in enumerate(self.axes)
        )
        return self.get(key)

    def prune_before(self, minimums: Tuple[int, ...]) -> None:
        if not self.axes:
            return
        mins = np.array([int(m) for m in minimums], dtype=np.int64)
        self.origin = np.maximum(self.origin, mins)
        for global_key in list(self.global_to_ring.keys()):
            if any(g < self.origin[idx] for idx, g in enumerate(global_key)):
                ring = self.global_to_ring.pop(global_key)
                self.ring_to_global.pop(ring, None)
                self.ring_to_value.pop(ring, None)

    def max_position_map(self) -> Dict[str, int]:
        if not self.axes or not self.initialized:
            return {}
        return {axis: int(self.max_pos[idx]) for idx, axis in enumerate(self.axes)}


def _normalize_device_spec(spec: str) -> str:
    device = (spec or "").strip()
    if not device:
        return "auto"
    lowered = device.lower()
    if lowered == "gpu":
        return "cuda"
    if lowered.startswith("gpu:"):
        return "cuda:" + lowered.split(":", 1)[1]
    if lowered in {"auto", "cpu", "mps"}:
        return lowered
    if lowered.startswith("cuda"):
        return lowered
    if lowered.startswith("mps"):
        return "mps"
    return lowered


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Common execution switches shared across the NumPy/Torch/JAX runtimes.

    Key behaviors:
    * ``precision`` defaults to ``"fp32"`` and can be set to ``"bf16"``, ``"fp16"``,
      or ``"auto"`` for backends that support mixed-precision lowering.
    * ``device`` tracks the desired target (``"cpu"``, ``"cuda"``, ``"mps"`` or ``"auto"``)
      so Torch FX and JAX lowerings can keep device placement aligned with NumPy fallbacks.
    * ``zero_copy`` enables zero-copy host handoffs whenever possible to avoid
      unnecessary host ↔ device transfers in hybrid execution paths.
    """

    mode: str = "single"  # "single" | "fixpoint" | "demand"
    fixpoint_strategy: str = "synchronous"  # "synchronous" | "semi_naive"
    max_iters: int = 32
    tol: float = 1e-6
    chaining: str = "forward"  # "forward" | "backward"
    explain_timings: bool = True
    projection_strategy: str = "exact"  # "exact" | "monte_carlo"
    projection_samples: Optional[int] = None
    projection_seed: Optional[int] = None
    temperatures: Optional[Dict[str, TemperatureSchedule]] = None
    precision: str = "fp32"  # "fp32" | "bf16" | "fp16" | "auto"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps" | "cuda:N"
    zero_copy: bool = True
    jax_enable_xla_cache: bool = False
    jax_cache_dir: Optional[str] = None
    validate_device_transfers: bool = False
    block_size: Optional[int] = None

    def normalized(self) -> "ExecutionConfig":
        mode = self.mode.lower()
        if mode not in {"single", "fixpoint", "demand"}:
            raise ValueError(f"Unsupported execution mode: {self.mode}")
        fixpoint = (self.fixpoint_strategy or "synchronous").lower()
        if fixpoint not in {"synchronous", "semi_naive"}:
            raise ValueError(f"Unsupported fixpoint strategy: {self.fixpoint_strategy}")
        chaining = self.chaining.lower()
        if chaining not in {"forward", "backward"}:
            raise ValueError(f"Unsupported chaining mode: {self.chaining}")
        strategy = self.projection_strategy.lower()
        if strategy not in {"exact", "monte_carlo"}:
            raise ValueError(f"Unsupported projection strategy: {self.projection_strategy}")
        samples = self.projection_samples
        if samples is not None:
            samples = int(samples)
            if samples <= 0:
                raise ValueError("projection_samples must be positive when provided")
        seed = self.projection_seed
        if seed is not None:
            seed = int(seed)
        temperatures = normalize_temperature_map(self.temperatures)
        precision = (self.precision or "fp32").lower()
        if precision not in {"fp32", "bf16", "fp16", "auto"}:
            raise ValueError(f"Unsupported precision setting: {self.precision}")
        device = _normalize_device_spec(self.device)
        zero_copy = bool(self.zero_copy)
        jax_enable_cache = bool(self.jax_enable_xla_cache)
        cache_dir = self.jax_cache_dir
        if cache_dir is not None:
            cache_dir = str(Path(cache_dir).expanduser())
        validate_transfers = bool(self.validate_device_transfers)
        block_size = self.block_size
        if block_size is not None:
            block_size = int(block_size)
            if block_size <= 0:
                raise ValueError("block_size must be positive when provided")
        return replace(
            self,
            mode=mode,
            fixpoint_strategy=fixpoint,
            chaining=chaining,
            projection_strategy=strategy,
            projection_samples=samples,
            projection_seed=seed,
            temperatures=temperatures,
            precision=precision,
            device=device,
            zero_copy=zero_copy,
            jax_enable_xla_cache=jax_enable_cache,
            jax_cache_dir=cache_dir,
            validate_device_transfers=validate_transfers,
            block_size=block_size,
        )


class NumpyRunner:
    def __init__(
        self,
        ir: ProgramIR,
        config: Optional[ExecutionConfig] = None,
        policies: Optional[RuntimePolicies] = None,
    ):
        self.ir = ir
        self.config = (config or ExecutionConfig()).normalized()
        self.policies = policies or RuntimePolicies()
        self.tensors: Dict[str, Any] = {}
        self.index_domains: Dict[str, int] = {}
        self.logs: List[Dict[str, Any]] = []
        self.boolean_tensors: Set[str] = self.ir.boolean_tensors()
        self._active_lhs: Optional[str] = None
        self._active_prev_value: Optional[np.ndarray] = None
        self._temperature_schedules = self.config.temperatures or {}
        self._last_temperatures: Dict[str, float] = {}
        self._active_equation_temperature: Optional[float] = None
        self._sig_temperatures: List[float] = []
        self._temperature_zero_tol = 1e-9

        self.stream_enabled = self.ir.has_streaming()
        self.stream_axes: Set[str] = set()
        self.stream_axis_min_offset: Dict[str, int] = {}
        self.tensor_stream_axes: Dict[str, Tuple[str, ...]] = {}
        self.tensor_static_indices: Dict[str, Tuple[str, ...]] = {}
        self.stream_storage: Dict[str, _StreamRingStore] = {}
        self.stream_positions: Dict[str, int] = {}
        if self.stream_enabled:
            self._analyze_streaming()
        else:
            self.stream_positions = {}

        self._sources: List[Equation] = []
        self._sinks: List[Equation] = []
        self._groups: List[Tuple[str, List[Equation]]] = []
        self._group_dependencies: Dict[str, Set[str]] = {}
        self._group_dependents: Dict[str, Set[str]] = {}
        self._group_equation_dependencies: Dict[str, List[Set[str]]] = {}
        self._group_contrib_cache: Dict[str, List[Optional[np.ndarray]]] = {}
        self._group_meta_cache: Dict[str, List[Optional[Dict[str, Any]]]] = {}
        self._prepare()
        self._group_dependents = self._build_group_dependents()
        self._reset_rng(self.config)

    # Public API ----------------------------------------------------------------
    def __call__(
        self,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[ExecutionConfig] = None,
        policies: Optional[RuntimePolicies] = None,
    ):
        return self.run(inputs=inputs, config=config, policies=policies)

    def run(
        self,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[ExecutionConfig] = None,
        policies: Optional[RuntimePolicies] = None,
    ):
        cfg = (config or self.config).normalized()
        if policies is not None:
            self.policies = policies
        self.config = cfg
        self._temperature_schedules = self.config.temperatures or {}
        self._reset_rng(cfg)
        self._reset_state()
        if self.stream_enabled:
            self._stream_base = {
                axis: self.stream_positions.get(axis, 0) for axis in self.stream_axes
            }
            self._stream_max_pos = dict(self._stream_base)
        else:
            self._stream_base = {}
            self._stream_max_pos = {}
        if inputs:
            for name, value in inputs.items():
                if self.stream_enabled and name in self.tensor_stream_axes:
                    normalized = self._normalize_boolean(name, value)
                    self._store_stream_input(name, normalized)
                else:
                    normalized = self._normalize_boolean(name, value)
                    self.tensors[name] = normalized
        self.logs.clear()

        self._run_sources()
        if cfg.mode == "fixpoint":
            self._run_fixpoint(cfg)
        else:
            self._run_single_pass(cfg)
        self._run_sinks()
        if self.stream_enabled:
            self.stream_positions.update(self._stream_max_pos)

        exports: Dict[str, Any] = {}
        for name in self.ir.exports:
            if self.stream_enabled and name in self.tensor_stream_axes:
                exports[name] = self._latest_stream_value(name)
            else:
                exports[name] = self.tensors.get(name)
        return exports

    def explain(self, *, json: bool = False):
        eq_metrics: List[Dict[str, Any]] = []
        total_time_ms = 0.0
        total_flops = 0.0
        total_bytes = 0.0

        def _format_metric(value: Optional[float], unit: str) -> Optional[str]:
            if value is None:
                return None
            magnitude = float(value)
            if magnitude == 0:
                return f"{unit}=0"
            suffixes = [
                (1e12, "T"),
                (1e9, "G"),
                (1e6, "M"),
                (1e3, "K"),
            ]
            for threshold, label in suffixes:
                if magnitude >= threshold:
                    return f"{unit}={magnitude / threshold:.2f}{label}"
            return f"{unit}={magnitude:.2f}"

        lines: List[str] = []
        for entry in self.logs:
            kind = entry.get("kind")
            if kind == "source":
                src = entry["source"]
                lines.append(f"[src] {src['name']} <- {src['path']} shape={src['shape']}")
            elif kind == "equation":
                eq = entry["equation"]
                details: List[str] = []
                einsum = eq.get("einsum")
                projected = eq.get("projected") or []
                if einsum:
                    proj_suffix = f" projected={','.join(projected)}" if projected else ""
                    details.append(f"einsum={einsum}{proj_suffix}")
                elif projected:
                    details.append(f"projected={','.join(projected)}")
                op = eq.get("op")
                if op:
                    details.append(f"op={op}")
                temp = eq.get("temperature")
                if temp is not None:
                    if isinstance(temp, list):
                        temp_str = ",".join(f"{float(t):g}" for t in temp)
                    else:
                        temp_str = f"{float(temp):g}"
                    details.append(f"T={temp_str}")
                flops = eq.get("flops")
                bytes_total = eq.get("bytes_total")
                flops_str = _format_metric(flops, "flops") if flops is not None else None
                bytes_str = (
                    _format_metric(bytes_total, "bytes") if bytes_total is not None else None
                )
                if flops_str:
                    details.append(flops_str)
                if bytes_str:
                    details.append(bytes_str)
                table = eq.get("index_table")
                if table:
                    details.append(f"idx[{table}]")
                status = eq["status"]
                iteration = eq["iteration"]
                duration = eq.get("duration_ms")
                if duration is not None:
                    total_time_ms += float(duration)
                if flops is not None:
                    total_flops += float(flops)
                if bytes_total is not None:
                    total_bytes += float(bytes_total)
                eq_metrics.append(
                    {
                        "name": eq.get("name"),
                        "iteration": iteration,
                        "status": status,
                        "duration_ms": duration,
                        "flops": flops,
                        "bytes_total": bytes_total,
                    }
                )
                timing = f" {duration:.3f}ms" if duration is not None else ""
                note = f" {' '.join(details)}" if details else ""
                lines.append(f"[iter {iteration:02d}] {eq['name']} {status}{timing}{note}")
            elif kind == "sink":
                sk = entry["sink"]
                lines.append(f"[sink] {sk['path']} <- {sk['name']} ({sk['mode']})")

        perf_summary: Optional[Dict[str, Any]] = None
        if eq_metrics:
            max_entry = max(
                eq_metrics,
                key=lambda item: item["duration_ms"] if item["duration_ms"] is not None else -1.0,
            )
            perf_summary = {
                "equations": len(eq_metrics),
                "total_ms": total_time_ms,
                "total_flops": total_flops or None,
                "total_bytes": total_bytes or None,
                "max_equation": {
                    "name": max_entry.get("name"),
                    "duration_ms": max_entry.get("duration_ms"),
                    "iteration": max_entry.get("iteration"),
                },
            }
            summary_parts: List[str] = [
                f"total={total_time_ms:.3f}ms",
                f"equations={len(eq_metrics)}",
            ]
            flops_summary = _format_metric(total_flops if total_flops else None, "flops")
            bytes_summary = _format_metric(total_bytes if total_bytes else None, "bytes")
            if flops_summary:
                summary_parts.append(flops_summary)
            if bytes_summary:
                summary_parts.append(bytes_summary)
            max_duration = max_entry.get("duration_ms")
            if max_duration is not None:
                summary_parts.append(f"max={max_entry.get('name')}({float(max_duration):.3f}ms)")
            lines.append(f"[perf] {' '.join(filter(None, summary_parts))}")

        if json:
            payload: Dict[str, Any] = {"logs": [_json_ready(entry) for entry in self.logs]}
            if perf_summary is not None:
                payload["summary"] = perf_summary
            return payload

        return "\n".join(lines)

    # Internal helpers ----------------------------------------------------------
    def _prepare(self):
        group_map: Dict[str, List[Equation]] = {}
        for eq in self.ir.equations:
            if eq.is_source:
                self._sources.append(eq)
            elif eq.is_sink:
                self._sinks.append(eq)
            else:
                bucket = group_map.setdefault(eq.lhs.name, [])
                bucket.append(eq)
        # Maintain authoring order
        seen = set()
        for eq in self.ir.equations:
            if eq.is_source or eq.is_sink:
                continue
            name = eq.lhs.name
            if name in seen:
                continue
            seen.add(name)
            eqs = group_map[name]
            self._groups.append((name, eqs))
            deps: Set[str] = set()
            per_eq: List[Set[str]] = []
            for item in eqs:
                eq_deps = _collect_tensor_names(item.rhs)
                eq_deps.discard(name)
                per_eq.append(eq_deps)
                deps.update(eq_deps)
            self._group_dependencies[name] = deps
            self._group_equation_dependencies[name] = per_eq

    def _build_group_dependents(self) -> Dict[str, Set[str]]:
        dependents: Dict[str, Set[str]] = {name: set() for name, _ in self._groups}
        for name, deps in self._group_dependencies.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].add(name)
        return dependents

    def _reset_state(self):
        self.tensors = {}
        self.index_domains = {}
        self._last_temperatures.clear()
        self._active_equation_temperature = None
        self._sig_temperatures.clear()
        self._group_contrib_cache = {name: [None] * len(eqs) for name, eqs in self._groups}
        self._group_meta_cache = {name: [None] * len(eqs) for name, eqs in self._groups}

    def _reset_rng(self, cfg: ExecutionConfig):
        seed = getattr(cfg, "projection_seed", None)
        if seed is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = np.random.default_rng(seed)

    def _is_boolean_tensor(self, name: str) -> bool:
        return name in self.boolean_tensors

    def _normalize_boolean(self, name: str, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        arr = _to_numpy_array(value)
        if not self._is_boolean_tensor(name):
            return arr
        return step(arr)

    # Streaming helpers ---------------------------------------------------------
    def _analyze_streaming(self):
        axis_min: Dict[str, int] = {}

        for eq in self.ir.equations:
            self._register_stream_ref(eq.lhs, axis_min)
            self._register_stream_expr(eq.rhs, axis_min)

        self.stream_axes = set(axis_min.keys())
        for axis in self.stream_axes:
            self.stream_axis_min_offset[axis] = axis_min.get(axis, 0)
            self.stream_positions.setdefault(axis, 0)

    def _register_stream_ref(
        self,
        ref: TensorRef,
        axis_min: Dict[str, int],
    ):
        if not isinstance(ref, TensorRef) or not ref.rolling:
            return
        axes_in_order: List[str] = []
        for idx in ref.indices:
            if idx not in ref.rolling:
                continue
            offset = ref.rolling[idx]
            axis_min[idx] = min(axis_min.get(idx, 0), offset)
            axes_in_order.append(idx)
        if axes_in_order:
            canonical = self.tensor_stream_axes.setdefault(ref.name, tuple(axes_in_order))
            if canonical != tuple(axes_in_order):
                raise ValueError(
                    f"Tensor '{ref.name}' uses streaming axes with inconsistent ordering"
                )
        static_in_order = tuple(idx for idx in ref.indices if idx not in ref.rolling)
        if static_in_order:
            if (
                ref.name not in self.tensor_static_indices
                or not self.tensor_static_indices[ref.name]
            ):
                self.tensor_static_indices[ref.name] = static_in_order
        else:
            self.tensor_static_indices.setdefault(ref.name, tuple())

    def _register_stream_expr(
        self,
        expr: Any,
        axis_min: Dict[str, int],
    ):
        if isinstance(expr, TensorRef):
            self._register_stream_ref(expr, axis_min)
            return
        if isinstance(expr, Term):
            for factor in expr.factors:
                self._register_stream_expr(factor, axis_min)
            return
        if isinstance(expr, FuncCall):
            arg = expr.arg
            if isinstance(arg, tuple):
                for item in arg:
                    self._register_stream_expr(item, axis_min)
            elif arg is not None:
                self._register_stream_expr(arg, axis_min)

    def _stream_axes_for_ref(self, ref: TensorRef) -> Tuple[str, ...]:
        if ref.name in self.tensor_stream_axes:
            return self.tensor_stream_axes[ref.name]
        axes = tuple(idx for idx in ref.indices if idx in ref.rolling)
        self.tensor_stream_axes[ref.name] = axes
        return axes

    def _stream_key_for_ref(
        self, ref: TensorRef, base_positions: Dict[str, int]
    ) -> Tuple[int, ...]:
        axes = self._stream_axes_for_ref(ref)
        key: List[int] = []
        for axis in axes:
            if axis not in ref.rolling:
                raise ValueError(
                    f"Streaming axis '{axis}' missing in reference to tensor '{ref.name}'"
                )
            offset = ref.rolling[axis]
            base = base_positions.get(axis, 0)
            key.append(base + offset)
        return tuple(key)

    def _window_sizes_for_tensor(self, name: str) -> Tuple[int, ...]:
        axes = self.tensor_stream_axes.get(name, ())
        return tuple(max(1, 1 - self.stream_axis_min_offset.get(axis, 0)) for axis in axes)

    def _ensure_stream_store(self, name: str) -> _StreamRingStore:
        store = self.stream_storage.get(name)
        if store is None:
            axes = self.tensor_stream_axes.get(name, ())
            window_sizes = self._window_sizes_for_tensor(name)
            store = _StreamRingStore(axes, window_sizes)
            self.stream_storage[name] = store
        return store

    def _store_stream_input(self, name: str, value: np.ndarray):
        stream_axes = self.tensor_stream_axes.get(name)
        if not stream_axes:
            self.tensors[name] = self._normalize_boolean(name, value)
            return
        static_axes = self.tensor_static_indices.get(name, tuple())
        indices = list(static_axes) + list(stream_axes)
        rolling = {axis: 0 for axis in stream_axes}
        ref = TensorRef(name=name, indices=indices, rolling=rolling)
        normalized = self._normalize_boolean(name, value)
        self._store_stream_value(
            ref, normalized, base_positions=self._stream_base, update_max=False
        )

    def _store_stream_value(
        self,
        ref: TensorRef,
        value: np.ndarray,
        *,
        base_positions: Dict[str, int],
        update_max: bool,
    ):
        store = self._ensure_stream_store(ref.name)
        key = self._stream_key_for_ref(ref, base_positions)
        arr = _to_numpy_array(self._normalize_boolean(ref.name, value))
        store.store(key, arr, update_max=update_max)
        self._update_stream_index_domains_stream(ref, arr)
        if update_max:
            for axis, max_pos in store.max_position_map().items():
                current = self._stream_max_pos.get(axis, self.stream_positions.get(axis, 0))
                if max_pos > current:
                    self._stream_max_pos[axis] = max_pos
            self._prune_stream_tensor(ref.name)

    def _get_stream_value(self, ref: TensorRef) -> np.ndarray:
        store = self.stream_storage.get(ref.name)
        if store is None:
            raise KeyError(f"Streaming tensor '{ref.name}' has no stored values")
        key = self._stream_key_for_ref(ref, self._stream_base)
        arr = store.get(key)
        if arr is None:
            raise KeyError(f"Streaming tensor '{ref.name}' missing value for indices {key}")
        return arr

    def _assign_stream(self, ref: TensorRef, value: np.ndarray, *, tol: float) -> bool:
        store = self._ensure_stream_store(ref.name)
        key = self._stream_key_for_ref(ref, self._stream_base)
        arr = _to_numpy_array(value)
        prev = store.get(key)
        changed = True
        if prev is not None and self._values_close(prev, arr, tol=tol):
            changed = False
        store.store(key, arr, update_max=True)
        self._update_stream_index_domains_stream(ref, arr)
        for axis, max_pos in store.max_position_map().items():
            current = self._stream_max_pos.get(axis, self.stream_positions.get(axis, 0))
            if max_pos > current:
                self._stream_max_pos[axis] = max_pos
        self._prune_stream_tensor(ref.name)
        return changed

    def _update_stream_index_domains_stream(self, ref: TensorRef, value: np.ndarray):
        static_indices = [idx for idx in ref.indices if idx not in ref.rolling]
        if not static_indices:
            return
        arr = _to_numpy_array(value)
        for i, idx in enumerate(static_indices):
            if i < arr.ndim:
                self.index_domains[idx] = max(self.index_domains.get(idx, 0), arr.shape[i])

    def _prune_stream_tensor(self, name: str):
        store = self.stream_storage.get(name)
        if not store:
            return
        axes = store.axes
        if not axes:
            return
        minimums: List[int] = []
        position_map = store.max_position_map()
        for axis in axes:
            max_pos = position_map.get(axis, self.stream_positions.get(axis, 0))
            min_offset = self.stream_axis_min_offset.get(axis, 0)
            minimums.append(max_pos + min_offset)
        store.prune_before(tuple(minimums))

    def _latest_stream_value(self, name: str) -> Optional[np.ndarray]:
        store = self.stream_storage.get(name)
        if not store:
            return None
        return store.latest(self.stream_positions)

    # Source / sink handling ----------------------------------------------------
    def _run_sources(self):
        for eq in self._sources:
            name = eq.lhs.name
            if self.stream_enabled and eq.lhs.rolling:
                storage = self.stream_storage.get(name)
                arr = None
                if storage is not None:
                    try:
                        key = self._stream_key_for_ref(eq.lhs, self.stream_positions)
                        arr = storage.get(key)
                    except ValueError:
                        arr = None
                if arr is not None:
                    src_mode = "provided"
                else:
                    arr = self._normalize_boolean(name, read_tensor_from_file(eq.src_file))
                    self._store_stream_value(
                        eq.lhs,
                        arr,
                        base_positions=self.stream_positions,
                        update_max=False,
                    )
                    src_mode = "loaded"
            else:
                if name in self.tensors:
                    arr = self.tensors[name]
                    src_mode = "provided"
                else:
                    arr = self._normalize_boolean(name, read_tensor_from_file(eq.src_file))
                    self.tensors[name] = arr
                    src_mode = "loaded"
                self._update_index_domains(eq.lhs, _to_numpy_array(arr))
            self.logs.append(
                {
                    "kind": "source",
                    "source": {
                        "name": name,
                        "path": eq.src_file,
                        "shape": tuple(_to_numpy_array(arr).shape),
                        "mode": src_mode,
                    },
                }
            )

    def _run_sinks(self):
        for eq in self._sinks:
            val = self._eval(eq.rhs, lhs=None)
            mode = "tensor"
            if isinstance(val, tuple) and val and val[0] == "__topk__":
                _, array, k = val
                out = topk(array, k=k, rng=self._rng)
                mode = f"topk(k={k})"
            else:
                out = val
            if eq.sink_file is None:
                raise ValueError("Sink equation missing target file path")
            target_path = self.policies.resolve_output_path(eq.sink_file)
            write_tensor_to_file(target_path, out)
            sink_path = str(target_path)
            self.logs.append(
                {
                    "kind": "sink",
                    "sink": {
                        "path": sink_path,
                        "name": eq.rhs.name if isinstance(eq.rhs, TensorRef) else None,
                        "mode": mode,
                    },
                }
            )

    # Equation execution -------------------------------------------------------
    def _ordered_groups(
        self,
        subset: Optional[Set[str]] = None,
    ) -> List[Tuple[str, List[Equation]]]:
        if subset is None:
            sequence = self._groups
        else:
            wanted = subset
            sequence = [(name, eqs) for name, eqs in self._groups if name in wanted]
        if self.config.chaining == "forward":
            return list(sequence)
        return list(reversed(sequence))

    def _run_single_pass(self, cfg: ExecutionConfig):
        for name, eqs in self._ordered_groups():
            self._evaluate_group(
                name,
                eqs,
                iteration=0,
                tol=cfg.tol,
                capture_timing=cfg.explain_timings,
                changed_inputs=None,
            )

    def _run_fixpoint(self, cfg: ExecutionConfig):
        strategy = getattr(cfg, "fixpoint_strategy", "synchronous")
        if strategy == "semi_naive":
            self._run_fixpoint_semi_naive(cfg)
        else:
            self._run_fixpoint_sync(cfg)

    def _run_fixpoint_sync(self, cfg: ExecutionConfig):
        groups = self._ordered_groups()
        for iteration in range(cfg.max_iters):
            any_change = False
            for name, eqs in groups:
                changed = self._evaluate_group(
                    name,
                    eqs,
                    iteration=iteration,
                    tol=cfg.tol,
                    capture_timing=cfg.explain_timings,
                    changed_inputs=None,
                )
                any_change = any_change or changed
            if not any_change:
                break

    def _run_fixpoint_semi_naive(self, cfg: ExecutionConfig):
        pending: Set[str] = {name for name, _ in self._groups}
        changed_last_iter: Set[str] = set()
        for iteration in range(cfg.max_iters):
            if not pending:
                break
            groups = self._ordered_groups(pending)
            next_pending: Set[str] = set()
            changed_this_iter: Set[str] = set()
            changed_inputs = None if iteration == 0 else changed_last_iter
            for name, eqs in groups:
                changed = self._evaluate_group(
                    name,
                    eqs,
                    iteration=iteration,
                    tol=cfg.tol,
                    capture_timing=cfg.explain_timings,
                    changed_inputs=changed_inputs,
                )
                if changed:
                    changed_this_iter.add(name)
                    next_pending.update(self._group_dependents.get(name, set()))
                    next_pending.add(name)
            if not changed_this_iter:
                break
            changed_last_iter = changed_this_iter
            pending = next_pending

    def _evaluate_group(
        self,
        name: str,
        equations: Sequence[Equation],
        *,
        iteration: int,
        tol: float,
        capture_timing: bool,
        changed_inputs: Optional[Set[str]],
    ) -> bool:
        if not equations:
            return False

        contributions: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        durations: List[Optional[float]] = []
        lhs_ref = equations[0].lhs
        lhs_name = lhs_ref.name
        group_temperature = self._resolve_group_temperature(lhs_name, iteration)
        prev_value = self.tensors.get(lhs_name)
        had_prev = lhs_name in self.tensors
        running_total: Optional[np.ndarray] = None
        self._active_lhs = lhs_name
        self._active_prev_value = (
            None if (lhs_ref.rolling or prev_value is None) else _to_numpy_array(prev_value)
        )
        self._active_equation_temperature = group_temperature

        eq_dependencies = self._group_equation_dependencies.get(name, [])
        cache_values = self._group_contrib_cache.setdefault(name, [None] * len(equations))
        cache_metas = self._group_meta_cache.setdefault(name, [None] * len(equations))
        if len(cache_values) < len(equations):
            cache_values.extend([None] * (len(equations) - len(cache_values)))
        if len(cache_metas) < len(equations):
            cache_metas.extend([None] * (len(equations) - len(cache_metas)))

        try:
            for idx, eq in enumerate(equations):
                start = time.perf_counter() if capture_timing else None
                self._sig_temperatures.clear()
                deps = eq_dependencies[idx] if idx < len(eq_dependencies) else set()
                can_reuse = (
                    changed_inputs is not None
                    and cache_values[idx] is not None
                    and (not deps or not (deps & changed_inputs))
                )
                if can_reuse and cache_metas[idx] is not None:
                    value_arr = cache_values[idx]  # type: ignore[assignment]
                    meta = dict(cache_metas[idx])  # type: ignore[arg-type]
                    meta["cached"] = True
                    durations.append(0.0 if capture_timing else None)
                else:
                    value, meta = self._eval_equation(eq)
                    value_arr = _to_numpy_array(value)
                    cache_values[idx] = value_arr
                    stored_meta = dict(meta)
                    stored_meta.pop("cached", None)
                    cache_metas[idx] = stored_meta
                    if start is not None:
                        durations.append((time.perf_counter() - start) * 1000.0)
                    else:
                        durations.append(None)
                temps_used = list(self._sig_temperatures)
                effective_temp = None
                if temps_used:
                    effective_temp = temps_used[0] if len(temps_used) == 1 else temps_used
                elif group_temperature is not None:
                    effective_temp = group_temperature
                if effective_temp is not None:
                    meta = dict(meta)
                    meta["temperature"] = effective_temp
                value_arr = _to_numpy_array(value_arr)
                contributions.append(value_arr)
                metas.append(meta)
                temps_used = list(self._sig_temperatures)
                if self._is_boolean_tensor(lhs_name):
                    running_total = (
                        value_arr if running_total is None else np.maximum(running_total, value_arr)
                    )
                    self.tensors[lhs_name] = running_total
                else:
                    running_total = (
                        value_arr if running_total is None else running_total + value_arr
                    )
                    self.tensors[lhs_name] = running_total
        finally:
            self._active_lhs = None
            self._active_prev_value = None
            self._active_equation_temperature = None
            self._sig_temperatures.clear()

        total = running_total if running_total is not None else contributions[0]
        if self._is_boolean_tensor(lhs_name):
            total = step(total)

        if had_prev:
            self.tensors[lhs_name] = prev_value  # type: ignore[assignment]
        else:
            self.tensors.pop(lhs_name, None)

        lhs_ref = equations[0].lhs
        changed = self._assign(lhs_ref, total, tol=tol)

        for idx, (eq, meta) in enumerate(zip(equations, metas)):
            cached = bool(meta.get("cached"))
            if cached:
                status = "cached"
            else:
                status = (
                    "update"
                    if (changed and idx == len(equations) - 1)
                    else ("unchanged" if (not changed and idx == len(equations) - 1) else "contrib")
                )
            self._log_equation(
                eq,
                meta,
                iteration=iteration,
                status=status,
                duration_ms=durations[idx],
            )
        return changed

    def _assign(self, ref: TensorRef, value: np.ndarray, *, tol: float) -> bool:
        if ref.rolling:
            return self._assign_stream(ref, value, tol=tol)
        value = self._normalize_boolean(ref.name, value)
        value = _to_numpy_array(value)
        prev = self.tensors.get(ref.name)
        if prev is None:
            self.tensors[ref.name] = value
            self._update_index_domains(ref, value)
            return True
        if not self._values_close(prev, value, tol=tol):
            self.tensors[ref.name] = value
            self._update_index_domains(ref, value)
            return True
        self._update_index_domains(ref, prev)
        return False

    def _update_index_domains(self, ref: TensorRef, arr: np.ndarray):
        arr = _to_numpy_array(arr)
        static_indices = [idx for idx in ref.indices if idx not in ref.rolling]
        for i, idx in enumerate(static_indices):
            if i < arr.ndim:
                self.index_domains[idx] = max(self.index_domains.get(idx, 0), arr.shape[i])

    def _log_equation(
        self,
        eq: Equation,
        meta: Dict[str, Any],
        *,
        iteration: int,
        status: str,
        duration_ms: Optional[float],
    ):
        projected = meta.get("projected", []) or []
        index_summary = equation_index_summary(eq, projected)
        entry = {
            "kind": "equation",
            "equation": {
                "name": eq.lhs.name,
                "status": status,
                "iteration": iteration,
                "duration_ms": duration_ms,
                "einsum": meta.get("einsum"),
                "projected": projected,
                "op": meta.get("op"),
                "temperature": meta.get("temperature"),
                "projection": meta.get("projection"),
                "strategy": meta.get("strategy"),
                "flops": meta.get("flops"),
                "bytes_total": meta.get("bytes_total"),
                "bytes_in": meta.get("bytes_in"),
                "bytes_out": meta.get("bytes_out"),
                "contracted": meta.get("contracted", []),
                "output_indices": meta.get("output_indices", []),
                "reductions": meta.get("reductions"),
                "index_summary": index_summary,
                "index_table": format_index_summary(index_summary),
            },
        }
        self.logs.append(entry)

    # Temperature helpers ------------------------------------------------------
    def temperature_manifest(self) -> Dict[str, Any]:
        if not self._temperature_schedules:
            return {}
        return {name: schedule.manifest() for name, schedule in self._temperature_schedules.items()}

    def _resolve_group_temperature(self, name: str, iteration: int) -> Optional[float]:
        if not self._temperature_schedules:
            return None
        schedule = self._temperature_schedules.get(name) or self._temperature_schedules.get("*")
        if schedule is None:
            return None
        value = schedule(iteration)
        if value is None:
            return None
        temperature = self._sanitize_temperature(value)
        self._last_temperatures[name] = temperature
        return temperature

    def _sanitize_temperature(self, value: Any) -> float:
        temp = coerce_temperature_value(value)
        if abs(temp) <= self._temperature_zero_tol:
            return 0.0
        return temp

    def _evaluate_temperature_operand(
        self,
        operand: Any,
        lhs: Optional[TensorRef],
    ) -> Optional[float]:
        if operand is None:
            return None
        if isinstance(operand, (int, float)):
            return self._sanitize_temperature(operand)
        value = self._eval(operand, lhs=lhs)
        if value is None:
            return None
        return self._sanitize_temperature(value)

    # Evaluation primitives -----------------------------------------------------
    def _eval_equation(self, eq: Equation) -> Tuple[np.ndarray, Dict[str, Any]]:
        rhs = eq.rhs
        lhs_name = eq.lhs.name
        if isinstance(rhs, Term):
            value, meta = self._eval_term(
                rhs,
                lhs=eq.lhs,
                capture_meta=True,
                projection=eq.projection,
            )
            value = self._normalize_boolean(lhs_name, value)
            return value, meta
        if isinstance(rhs, FuncCall):
            value = self._eval_fn(rhs, lhs=eq.lhs)
            meta = {"op": rhs.name}
            value = self._normalize_boolean(lhs_name, value)
            return value, meta
        value = self._eval(rhs, lhs=eq.lhs)
        value = self._normalize_boolean(lhs_name, value)
        return value, {}

    def _eval(self, expr: Any, lhs: Optional[TensorRef] = None):
        if isinstance(expr, TensorRef):
            if expr.rolling:
                return self._get_stream_value(expr)
            use_prev = (
                self._active_lhs == expr.name
                and self._active_prev_value is not None
                and not expr.rolling
            )
            if use_prev:
                base_value = self._active_prev_value
            else:
                resolved = None
                if expr.name not in self.tensors:
                    if self.policies.weight_store is not None:
                        try:
                            resolved = self.policies.weight_store.resolve(expr.name)
                        except KeyError:
                            resolved = None
                if resolved is not None:
                    arr_np = self.policies.materialize_weight(
                        expr.name,
                        resolved,
                        backend="numpy",
                        device=None,
                    )
                    arr_np = self._normalize_boolean(expr.name, arr_np)
                    self.tensors[expr.name] = arr_np
                    self._update_index_domains(expr, arr_np)
                elif expr.name not in self.tensors:
                    raise KeyError(f"Tensor '{expr.name}' not yet defined")
                base_value = self.tensors[expr.name]
            materialized = self._apply_index_specs(_to_numpy_array(base_value), expr)
            self._update_index_domains(expr, materialized)
            return materialized
        if isinstance(expr, Term):
            value, _ = self._eval_term(expr, lhs=lhs, capture_meta=False, projection="sum")
            return value
        if isinstance(expr, FuncCall):
            return self._eval_fn(expr, lhs=lhs)
        if isinstance(expr, IndexFunction):
            axis_lengths = self._collect_axis_lengths(lhs, [], [])
            dtype = np.dtype(np.float32)
            return self._eval_index_function(expr, axis_lengths, dtype)
        if isinstance(expr, (int, float, bool, np.ndarray)):
            return expr
        if isinstance(expr, str):
            return expr
        if expr is None:
            return None
        raise ValueError(f"Unknown expr type: {type(expr)}")

    def _apply_index_specs(self, array: np.ndarray, ref: TensorRef) -> np.ndarray:
        specs = getattr(ref, "index_specs", None)
        if not specs:
            return array
        result = _to_numpy_array(array)
        if result.ndim == 0:
            return result
        static_specs = [spec for spec in specs if spec.axis not in ref.rolling]
        if not static_specs:
            return result
        if result.ndim < len(static_specs):
            raise ValueError(
                f"Tensor '{ref.name}' rank {result.ndim} smaller than indices {ref.indices}"
            )
        indexer = [slice(None)] * result.ndim
        has_slice = False
        for axis_idx, spec in enumerate(static_specs[: result.ndim]):
            if spec.slice is not None:
                sl = spec.slice
                indexer[axis_idx] = slice(sl.start, sl.stop, sl.step)
                has_slice = True
        if has_slice:
            result = result[tuple(indexer)]
        for axis_idx, spec in enumerate(static_specs[: result.ndim]):
            if spec.slice is not None:
                continue
            if spec.offset:
                result = _shift_axis(result, axis_idx, spec.offset)
        return result

    def _collect_axis_lengths(
        self,
        lhs: Optional[TensorRef],
        factors: Sequence[Any],
        evaluated: Sequence[Any],
    ) -> Dict[str, int]:
        lengths: Dict[str, int] = dict(self.index_domains)
        if lhs is not None and lhs.name in self.tensors:
            lhs_value = self.tensors[lhs.name]
            lhs_indices = [idx for idx in lhs.indices if idx not in lhs.rolling]
            for axis_idx, axis_name in enumerate(lhs_indices):
                if axis_idx < lhs_value.ndim:
                    lengths.setdefault(axis_name, lhs_value.shape[axis_idx])
        for factor, value in zip(factors, evaluated):
            if value is None:
                continue
            indices = _factor_indices(factor)
            if not indices:
                continue
            arr = _to_numpy_array(value)
            for axis_idx, axis_name in enumerate(indices):
                if axis_idx < arr.ndim:
                    lengths.setdefault(axis_name, arr.shape[axis_idx])
        return lengths

    def _resolve_term_dtype(self, evaluated: Sequence[Any]) -> np.dtype:
        for value in evaluated:
            if value is None:
                continue
            arr = _to_numpy_array(value)
            if arr.dtype != object:
                return arr.dtype
        return np.dtype(np.float32)

    def _eval_index_function(
        self,
        fn: IndexFunction,
        axis_lengths: Dict[str, int],
        dtype: np.dtype,
    ) -> np.ndarray:
        length = axis_lengths.get(fn.axis)
        if length is None:
            raise ValueError(f"Axis '{fn.axis}' length unknown for index function '{fn.name}'")
        indices = np.arange(length, dtype=np.int64)
        if fn.name == "even":
            mask = (indices % 2) == 0
        elif fn.name == "odd":
            mask = (indices % 2) == 1
        else:
            raise ValueError(f"Unsupported index function '{fn.name}'")
        return mask.astype(dtype)

    def _eval_term(
        self,
        term: Term,
        *,
        lhs: Optional[TensorRef],
        capture_meta: bool,
        projection: str = "sum",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if projection not in {"sum", "max", "mean"}:
            raise ValueError(f"Unsupported projection op '{projection}'")

        (
            base_equation,
            projected,
            factor_order,
            base_index_map,
        ) = _normalized_einsum(term, lhs)
        evaluated: List[Any] = []
        pending: List[Tuple[int, IndexFunction]] = []
        for idx, factor in enumerate(term.factors):
            if isinstance(factor, IndexFunction):
                evaluated.append(None)
                pending.append((idx, factor))
            else:
                evaluated.append(self._eval(factor, lhs=lhs))
        axis_lengths = self._collect_axis_lengths(lhs, term.factors, evaluated)
        dtype = self._resolve_term_dtype(evaluated)
        for pos, index_fn in pending:
            evaluated[pos] = self._eval_index_function(index_fn, axis_lengths, dtype)

        def _ordered_arrays(order: Sequence[int]) -> List[np.ndarray]:
            return [_to_numpy_array(evaluated[idx]) for idx in order]

        base_arrays = _ordered_arrays(factor_order)

        lhs_output_indices: List[str] = []
        if lhs is not None:
            lhs_output_indices = [idx for idx in lhs.indices if idx not in lhs.rolling]

        strategy = getattr(self.config, "projection_strategy", "exact")
        use_mc = strategy == "monte_carlo" and bool(projected)
        need_extended = bool(projected) and (projection != "sum" or use_mc)

        extended_equation: Optional[str] = None
        extended_order: Optional[List[int]] = None
        extended_index_map: Optional[Dict[str, str]] = None
        if need_extended:
            extended_lhs = self._build_extended_lhs(lhs, projected)
            (
                extended_equation,
                _,
                extended_order,
                extended_index_map,
            ) = _normalized_einsum(term, extended_lhs)

        def _array_shapes_and_sizes(
            arrays: Sequence[np.ndarray],
        ) -> Tuple[List[Tuple[int, ...]], List[int]]:
            shapes: List[Tuple[int, ...]] = []
            itemsizes: List[int] = []
            for arr in arrays:
                arr_np = _to_numpy_array(arr)
                shapes.append(tuple(int(dim) for dim in arr_np.shape))
                itemsizes.append(int(arr_np.dtype.itemsize))
            return shapes, itemsizes

        if use_mc:
            if extended_equation is None or extended_index_map is None:
                extended_lhs = self._build_extended_lhs(lhs, projected)
                (
                    extended_equation,
                    _,
                    extended_order,
                    extended_index_map,
                ) = _normalized_einsum(term, extended_lhs)
            extended_arrays = _ordered_arrays(extended_order or factor_order)
            path = np.einsum_path(extended_equation, *extended_arrays, optimize="optimal")[0]
            raw = np.einsum(extended_equation, *extended_arrays, optimize=path)
            shapes, itemsizes = _array_shapes_and_sizes(extended_arrays)
            stats = compute_einsum_stats(
                extended_equation,
                shapes,
                itemsizes,
                tuple(int(dim) for dim in raw.shape),
                int(raw.dtype.itemsize),
            )
            result, meta = self._apply_monte_carlo_projection(
                raw=raw,
                axes_start=len(lhs_output_indices),
                projection=projection,
                projected=projected,
                equation=extended_equation,
            )
            meta.update(stats)
        elif projection != "sum" and projected:
            if extended_equation is None or extended_index_map is None:
                extended_lhs = self._build_extended_lhs(lhs, projected)
                (
                    extended_equation,
                    _,
                    extended_order,
                    extended_index_map,
                ) = _normalized_einsum(term, extended_lhs)
            extended_arrays = _ordered_arrays(extended_order or factor_order)
            path = np.einsum_path(extended_equation, *extended_arrays, optimize="optimal")[0]
            block_size = getattr(self.config, "block_size", None)
            use_blocking = (
                block_size is not None
                and projected
                and axis_lengths.get(projected[0], 0) > block_size
            )
            if use_blocking:
                result, block_meta = self._einsum_projected_chunked(
                    equation=extended_equation,
                    arrays=extended_arrays,
                    projected=projected,
                    axis_lengths=axis_lengths,
                    axes_start=len(lhs_output_indices),
                    projection=projection,
                    index_map=extended_index_map,
                    block_size=int(block_size),
                    path=path,
                )
                meta = {
                    "einsum": extended_equation,
                    "projected": projected,
                    "projection": projection,
                }
                meta.update(block_meta)
            else:
                raw = np.einsum(extended_equation, *extended_arrays, optimize=path)
                result = self._reduce_extended_raw(
                    raw,
                    axes_start=len(lhs_output_indices),
                    projection=projection,
                )
                meta = {
                    "einsum": extended_equation,
                    "projected": projected,
                    "projection": projection,
                }
            shapes, itemsizes = _array_shapes_and_sizes(extended_arrays)
            stats = compute_einsum_stats(
                extended_equation,
                shapes,
                itemsizes,
                tuple(int(dim) for dim in result.shape),
                int(result.dtype.itemsize),
            )
            meta.update(stats)
        else:
            base_path = np.einsum_path(base_equation, *base_arrays, optimize="optimal")[0]
            result = np.einsum(base_equation, *base_arrays, optimize=base_path)
            meta = {"einsum": base_equation, "projected": projected}
            if projection != "sum":
                meta["projection"] = projection
            shapes, itemsizes = _array_shapes_and_sizes(base_arrays)
            stats = compute_einsum_stats(
                base_equation,
                shapes,
                itemsizes,
                tuple(int(dim) for dim in result.shape),
                int(result.dtype.itemsize),
            )
            meta.update(stats)

        result = _to_numpy_array(result)
        return (result, meta) if capture_meta else (result, {})

    def _build_extended_lhs(
        self,
        lhs: Optional[TensorRef],
        projected: Sequence[str],
    ) -> TensorRef:
        if lhs is not None:
            extended_indices = list(lhs.indices)
            dotted_axes = list(lhs.dotted_axes)
            rolling = dict(lhs.rolling)
            name = lhs.name
        else:
            extended_indices = []
            dotted_axes: List[str] = []
            rolling: Dict[str, int] = {}
            name = "__tmp__"
        for idx in projected:
            if idx not in extended_indices:
                extended_indices.append(idx)
        return TensorRef(
            name=name,
            indices=extended_indices,
            dotted_axes=dotted_axes,
            rolling=rolling,
        )

    def _einsum_projected_chunked(
        self,
        *,
        equation: str,
        arrays: Sequence[np.ndarray],
        projected: Sequence[str],
        axis_lengths: Dict[str, int],
        axes_start: int,
        projection: str,
        index_map: Dict[str, str],
        block_size: int,
        path,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not projected:
            raise ValueError("Chunked einsum requires at least one projected index")
        block_index = projected[0]
        block_letter = index_map.get(block_index)
        if block_letter is None:
            raise ValueError(f"Missing einsum index mapping for '{block_index}'")
        block_dim = axis_lengths.get(block_index)
        if block_dim is None:
            raise ValueError(f"Unknown axis length for projected index '{block_index}'")
        if block_dim <= 0:
            zero_shape = arrays[0].shape[:axes_start] if arrays else ()
            return np.zeros(zero_shape, dtype=np.float32), {"blocked": {"axis": block_index}}
        block = max(1, min(int(block_size), int(block_dim)))
        inputs, output = equation.split("->")
        input_specs = inputs.split(",")
        other_volume = 1
        for idx_name in projected[1:]:
            other_dim = axis_lengths.get(idx_name, 1)
            other_volume *= max(1, int(other_dim))

        total_sum: Optional[np.ndarray] = None
        total_result: Optional[np.ndarray] = None
        total_count = 0
        chunk_count = 0
        for start in range(0, int(block_dim), block):
            stop = min(start + block, int(block_dim))
            chunk_count += 1
            sliced_arrays: List[np.ndarray] = []
            for arr, spec in zip(arrays, input_specs):
                if block_letter in spec:
                    axis = spec.index(block_letter)
                    slicer = [slice(None)] * arr.ndim
                    slicer[axis] = slice(start, stop)
                    sliced_arrays.append(arr[tuple(slicer)])
                else:
                    sliced_arrays.append(arr)
            chunk_raw = np.einsum(equation, *sliced_arrays, optimize=path)
            if projection == "mean":
                chunk_sum = self._reduce_extended_raw(chunk_raw, axes_start, projection="sum")
                total_sum = chunk_sum if total_sum is None else total_sum + chunk_sum
                total_count += (stop - start) * other_volume
            elif projection == "max":
                chunk_reduced = self._reduce_extended_raw(chunk_raw, axes_start, projection="max")
                total_result = (
                    chunk_reduced
                    if total_result is None
                    else np.maximum(total_result, chunk_reduced)
                )
            else:
                chunk_reduced = self._reduce_extended_raw(chunk_raw, axes_start, projection="sum")
                total_result = (
                    chunk_reduced if total_result is None else total_result + chunk_reduced
                )
        if projection == "mean":
            if total_sum is None or total_count == 0:
                raise ValueError("Unable to compute mean for empty projection")
            result = total_sum / float(total_count)
        else:
            if total_result is None:
                raise ValueError("Chunked einsum produced no result")
            result = total_result
        meta = {
            "blocked": {
                "axis": block_index,
                "chunks": int(chunk_count),
                "block_size": int(block),
            }
        }
        if projection == "mean":
            meta["blocked"]["elements"] = int(total_count)
        return result, meta

    def _reduce_extended_raw(
        self,
        raw: np.ndarray,
        axes_start: int,
        *,
        projection: str,
    ) -> np.ndarray:
        if raw.ndim == axes_start:
            return raw
        reduce_axes = tuple(range(axes_start, raw.ndim))
        if projection == "sum":
            return np.sum(raw, axis=reduce_axes)
        if projection == "max":
            return np.amax(raw, axis=reduce_axes)
        if projection == "mean":
            return np.mean(raw, axis=reduce_axes)
        raise ValueError(f"Unsupported projection '{projection}'")

    def _apply_monte_carlo_projection(
        self,
        *,
        raw: np.ndarray,
        axes_start: int,
        projection: str,
        projected: Sequence[str],
        equation: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        proj_shape = raw.shape[axes_start:]
        proj_size = int(np.prod(proj_shape, dtype=np.int64)) if proj_shape else 1
        meta_base: Dict[str, Any] = {
            "einsum": equation,
            "projected": list(projected),
            "projection": projection,
            "strategy": "monte_carlo",
        }
        if proj_size == 0:
            zeros_shape = raw.shape[:axes_start]
            result = np.zeros(zeros_shape, dtype=raw.dtype)
            meta = dict(meta_base)
            meta.update({"samples": 0, "total": 0})
            return result, meta

        samples_cfg = getattr(self.config, "projection_samples", None)
        if samples_cfg is None:
            samples = min(64, proj_size)
        else:
            samples = int(samples_cfg)
        if samples <= 0:
            samples = 1

        if samples >= proj_size:
            reduced = self._reduce_extended_raw(raw, axes_start, projection=projection)
            meta = dict(meta_base)
            meta.update({"samples": int(proj_size), "total": int(proj_size)})
            return _to_numpy_array(reduced), meta

        samples = max(1, samples)
        raw_flat = raw.reshape(raw.shape[:axes_start] + (proj_size,))
        if not hasattr(self, "_rng"):
            self._reset_rng(self.config)
        idx = self._rng.integers(0, proj_size, size=samples)
        sampled = np.take(raw_flat, idx, axis=-1)
        if projection == "sum":
            approx = sampled.mean(axis=-1) * float(proj_size)
        elif projection == "mean":
            approx = sampled.mean(axis=-1)
        elif projection == "max":
            approx = sampled.max(axis=-1)
        else:
            raise ValueError(f"Unsupported projection '{projection}' for Monte Carlo")
        meta = dict(meta_base)
        meta.update({"samples": int(samples), "total": int(proj_size)})
        return _to_numpy_array(approx), meta

    def _eval_fn(self, fn: FuncCall, lhs: Optional[TensorRef] = None):
        name = fn.name.lower()
        args_expr = ()
        if isinstance(fn.arg, tuple):
            args_expr = fn.arg
        elif fn.arg is not None:
            args_expr = (fn.arg,)

        def eval_arg(expr):
            return self._eval(expr, lhs=lhs)

        if name == "step":
            return step(eval_arg(args_expr[0]))
        if name == "relu":
            return relu(eval_arg(args_expr[0]))
        if name == "sig":
            if not args_expr:
                raise ValueError("sig requires at least one argument")
            base = eval_arg(args_expr[0])
            temperature = None
            if len(args_expr) > 1:
                temperature = self._evaluate_temperature_operand(args_expr[1], lhs)
            elif "T" in fn.kwargs:
                temperature = self._evaluate_temperature_operand(fn.kwargs.get("T"), lhs)
            else:
                temperature = self._active_equation_temperature
                if temperature is not None:
                    temperature = self._sanitize_temperature(temperature)
            if temperature is None:
                temperature = 0.0
            self._sig_temperatures.append(temperature)
            return sig(base, temperature)
        if name == "gelu":
            return gelu(eval_arg(args_expr[0]))
        if name == "gelu_grad":
            return gelu_grad(eval_arg(args_expr[0]))
        if name == "lnorm":
            arr = eval_arg(args_expr[0])
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs)
            )
            return lnorm(arr, axis=axis)
        if name in {"layernorm", "layer_norm"}:
            arr = eval_arg(args_expr[0])
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs)
            )
            eps = float(fn.kwargs.get("eps", 1e-5))
            return layernorm(arr, axis=axis, eps=eps)
        if name == "softmax":
            arr = eval_arg(args_expr[0])
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs)
            )
            return softmax(arr, axis=axis)
        if name == "masked_softmax":
            arr = eval_arg(args_expr[0])
            mask_expr: Optional[Any] = None
            if len(args_expr) > 1:
                mask_expr = args_expr[1]
            elif "mask" in fn.kwargs:
                mask_expr = fn.kwargs.get("mask")
            mask_value = self._eval(mask_expr, lhs=lhs) if mask_expr is not None else None
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs)
            )
            fill_value = fn.kwargs.get("fill") if "fill" in fn.kwargs else None
            return masked_softmax(arr, mask=mask_value, axis=axis, fill_value=fill_value)
        if name == "softmax_grad":
            if len(args_expr) < 2:
                raise ValueError("softmax_grad requires probabilities and gradient arguments")
            probs = eval_arg(args_expr[0])
            grad = eval_arg(args_expr[1])
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs)
            )
            return softmax_grad(probs, grad, axis=axis)
        if name == "sin":
            if not args_expr:
                raise ValueError("sin requires an argument")
            return np.sin(eval_arg(args_expr[0]))
        if name == "cos":
            if not args_expr:
                raise ValueError("cos requires an argument")
            return np.cos(eval_arg(args_expr[0]))
        if name == "rope":
            if len(args_expr) == 2:
                arr = eval_arg(args_expr[0])
                return rope(arr, args_expr[1])
            return rope(eval_arg(args_expr[0]), None)
        if name == "concat":
            if not args_expr:
                raise ValueError("concat requires at least one argument")
            arrays = [eval_arg(arg) for arg in args_expr]
            axis_spec = fn.kwargs.get("axis")
            if len(args_expr) == 1:
                axis = None
            else:
                axis = self._axis_from_spec(axis_spec, args_expr[0], lhs, default=-1)
            return concat(arrays, axis=axis)
        if name in {"max", "amax"}:
            if not args_expr:
                raise ValueError("max/amax requires an argument")
            arr = eval_arg(args_expr[0])
            axis = self._axis_from_spec(fn.kwargs.get("axis"), args_expr[0], lhs, default=-1)
            keepdims = bool(fn.kwargs.get("keepdims", False))
            return reduce_max(arr, axis=axis, keepdims=keepdims)
        if name in {"avg", "mean"}:
            if not args_expr:
                raise ValueError("avg/mean requires an argument")
            arr = eval_arg(args_expr[0])
            axis = self._axis_from_spec(fn.kwargs.get("axis"), args_expr[0], lhs, default=-1)
            keepdims = bool(fn.kwargs.get("keepdims", False))
            return reduce_mean(arr, axis=axis, keepdims=keepdims)
        if name == "causal_mask":
            axis_expr = args_expr[0] if args_expr else fn.arg
            return causal_mask(L=int(axis_expr))
        if name == "const":
            value = args_expr[0] if args_expr else fn.arg
            return const(value)
        if name == "attention":
            if len(args_expr) < 3:
                raise ValueError("attention requires query, key, and value arguments")
            query = eval_arg(args_expr[0])
            key = eval_arg(args_expr[1])
            value = eval_arg(args_expr[2])
            mask_expr: Optional[Any] = None
            if len(args_expr) > 3:
                mask_expr = args_expr[3]
            elif "mask" in fn.kwargs:
                mask_expr = fn.kwargs.get("mask")
            mask_value = self._eval(mask_expr, lhs=lhs) if mask_expr is not None else None
            scale_expr = fn.kwargs.get("scale")
            scale_value = self._eval(scale_expr, lhs=lhs) if scale_expr is not None else None
            if isinstance(scale_value, str):
                scale_value = self.tensors.get(scale_value)
            causal = bool(fn.kwargs.get("causal", False))
            return attention(query, key, value, mask=mask_value, scale=scale_value, causal=causal)
        if name == "tucker_dense":
            if not args_expr:
                raise ValueError("tucker_dense requires a tensor argument")
            base = eval_arg(args_expr[0])
            rank_value = None
            threshold_value: Any = 0.5
            if len(args_expr) > 1:
                rank_value = eval_arg(args_expr[1])
            elif "rank" in fn.kwargs:
                rank_value = fn.kwargs.get("rank")
            if len(args_expr) > 2:
                threshold_value = eval_arg(args_expr[2])
            elif "threshold" in fn.kwargs:
                threshold_value = fn.kwargs.get("threshold")
            if isinstance(rank_value, str):
                rank_value = self.tensors.get(rank_value)
            rng = getattr(self, "_rng", None)
            return tucker_dense(base, rank=rank_value, threshold=float(threshold_value), rng=rng)
        if name == "topk":
            arr = eval_arg(args_expr[0])
            k = int(fn.kwargs.get("k", 5))
            return ("__topk__", _to_numpy_array(arr), k)
        if name == "case":
            if not args_expr:
                raise ValueError("case requires condition/value pairs")
            items = list(args_expr)
            default_expr: Optional[Any] = None
            if len(items) % 2 == 1:
                default_expr = items.pop()
            pairs = list(zip(items[0::2], items[1::2]))
            if not pairs and default_expr is None:
                raise ValueError("case requires at least one condition/value pair or default")
            result = None
            if default_expr is not None:
                result = _to_numpy_array(self._eval(default_expr, lhs=lhs))
            for cond_expr, value_expr in pairs:
                value = _to_numpy_array(self._eval(value_expr, lhs=lhs))
                cond = _to_numpy_array(self._eval(cond_expr, lhs=lhs)).astype(bool)
                if result is None:
                    result = np.zeros_like(value)
                result = np.where(cond, value, result)
            if result is None:
                raise ValueError("case evaluation produced no result")
            return result
        raise ValueError(f"Unknown function: {fn.name}")

    # Misc utilities ------------------------------------------------------------
    def _values_close(self, a: np.ndarray, b: np.ndarray, *, tol: float) -> bool:
        if a.shape != b.shape:
            return False
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
            return np.allclose(a, b, atol=tol, rtol=1e-5)
        return np.array_equal(a, b)

    def _dotted_axis(self, lhs: Optional[TensorRef]) -> Optional[int]:
        if lhs and lhs.dotted_axes:
            dotted = lhs.dotted_axes[0]
            if dotted in lhs.indices:
                return lhs.indices.index(dotted)
        return None

    def _axis_from_spec(
        self,
        spec: Any,
        arg_expr: Any,
        lhs: Optional[TensorRef],
        *,
        default: Optional[int],
    ) -> int:
        if spec is None:
            if default is not None:
                return default
            return -1
        if isinstance(spec, (int, float)):
            return int(spec)
        if isinstance(spec, str):
            if lhs and spec in lhs.indices:
                return lhs.indices.index(spec)
            tref = _first_tensor_ref(arg_expr)
            if tref and spec in tref.indices:
                return tref.indices.index(spec)
        raise ValueError(f"Cannot resolve axis specification: {spec}")


class DemandNumpyRunner(NumpyRunner):
    def __init__(
        self,
        ir: ProgramIR,
        config: Optional[ExecutionConfig] = None,
        policies: Optional[RuntimePolicies] = None,
    ):
        cfg = (config or ExecutionConfig(mode="demand")).normalized()
        if cfg.mode != "demand":
            cfg = replace(cfg, mode="demand")
        super().__init__(ir, config=cfg, policies=policies)
        if self.stream_enabled:
            raise NotImplementedError(
                "Demand-driven execution does not support streaming programs yet"
            )
        self._group_lookup: Dict[str, List[Equation]] = {name: eqs for name, eqs in self._groups}
        self._sources_lookup: Dict[str, Equation] = {src.lhs.name: src for src in self._sources}
        self._lhs_lookup: Dict[str, TensorRef] = {}
        for name, eqs in self._groups:
            if eqs:
                self._lhs_lookup[name] = eqs[0].lhs
        for src in self._sources:
            self._lhs_lookup[src.lhs.name] = src.lhs
        self._dependencies: Dict[str, Set[str]] = self._build_group_dependencies()
        cache_targets = set(self._lhs_lookup.keys())
        self._slice_cache: Dict[
            str, Dict[Tuple[Tuple[Optional[int], Optional[int], Optional[int]], ...], np.ndarray]
        ] = {name: {} for name in cache_targets}
        self._computed_groups: Set[str] = set()
        self._materialized_sources: Set[str] = set()
        self._configure_for_call(inputs=None, config=cfg, policies=policies, force_reset=True)

    def run(
        self,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[ExecutionConfig] = None,
        policies: Optional[RuntimePolicies] = None,
    ):
        self.logs.clear()
        self._configure_for_call(inputs=inputs, config=config, policies=policies, force_reset=True)
        exports: Dict[str, Any] = {}
        for name in self.ir.exports:
            exports[name] = self.query(name)
        return exports

    def query(
        self,
        name: str,
        selectors: Optional[Dict[str, Any]] = None,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[ExecutionConfig] = None,
        policies: Optional[RuntimePolicies] = None,
    ):
        self._configure_for_call(inputs=inputs, config=config, policies=policies, force_reset=False)
        selector = self._normalize_selector(name, selectors)
        cache_key = _selector_key(selector)
        cache = self._slice_cache.setdefault(name, {})
        if cache_key in cache:
            return np.array(cache[cache_key], copy=True)
        tensor = self._ensure_tensor_ready(name)
        result = tensor[selector]
        squeeze_axes = _axes_to_squeeze(selector)
        if squeeze_axes:
            result = np.squeeze(result, axis=squeeze_axes)
        cached = np.array(result, copy=True)
        cache[cache_key] = cached
        return np.array(cached, copy=True)

    def reset(self):
        self._reset_caches()

    def _configure_for_call(
        self,
        *,
        inputs: Optional[Dict[str, Any]],
        config: Optional[ExecutionConfig],
        policies: Optional[RuntimePolicies],
        force_reset: bool,
    ):
        reset_needed = force_reset
        if config is not None:
            cfg = config.normalized()
            if cfg.mode != "demand":
                raise ValueError("DemandNumpyRunner requires ExecutionConfig.mode='demand'")
            self.config = cfg
            self._temperature_schedules = self.config.temperatures or {}
            reset_needed = True
        if policies is not None:
            self.policies = policies
            reset_needed = True
        if inputs is not None:
            reset_needed = True
        if reset_needed:
            self._reset_caches()
        if inputs:
            self._apply_inputs(inputs)

    def _reset_caches(self):
        super()._reset_state()
        for cache in self._slice_cache.values():
            cache.clear()
        self._computed_groups.clear()
        self._materialized_sources.clear()
        self._reset_rng(self.config)

    def _apply_inputs(self, inputs: Dict[str, Any]):
        for name, value in inputs.items():
            normalized = self._normalize_boolean(name, value)
            arr = _to_numpy_array(normalized)
            self.tensors[name] = arr
            lhs = self._lhs_lookup.get(name)
            if lhs is not None:
                self._update_index_domains(lhs, arr)
            self._materialized_sources.add(name)
            self.logs.append(
                {
                    "kind": "source",
                    "source": {
                        "name": name,
                        "path": None,
                        "shape": tuple(arr.shape),
                        "mode": "provided",
                    },
                }
            )

    def _ensure_tensor_ready(self, name: str) -> np.ndarray:
        if name in self.tensors:
            return _to_numpy_array(self.tensors[name])
        if name in self._sources_lookup:
            return self._ensure_source(name)
        if name in self._group_lookup:
            self._materialize_groups_for(name)
            if name in self.tensors:
                return _to_numpy_array(self.tensors[name])
        if name in self.tensors:
            return _to_numpy_array(self.tensors[name])
        raise KeyError(f"Tensor '{name}' not defined")

    def _ensure_source(self, name: str) -> np.ndarray:
        if name in self.tensors:
            return _to_numpy_array(self.tensors[name])
        eq = self._sources_lookup.get(name)
        if eq is None:
            raise KeyError(f"Source '{name}' not found")
        arr = self._normalize_boolean(name, read_tensor_from_file(eq.src_file))
        arr_np = _to_numpy_array(arr)
        self.tensors[name] = arr_np
        self._materialized_sources.add(name)
        self._update_index_domains(eq.lhs, arr_np)
        self.logs.append(
            {
                "kind": "source",
                "source": {
                    "name": name,
                    "path": eq.src_file,
                    "shape": tuple(arr_np.shape),
                    "mode": "loaded",
                },
            }
        )
        return arr_np

    def _materialize_groups_for(self, target: str):
        needed_groups, needed_sources = self._collect_requirements(target)
        if needed_sources:
            for src_name in needed_sources:
                self._ensure_source(src_name)
        ordered = self._ordered_subset(needed_groups)
        pending = [(name, eqs) for name, eqs in ordered if name not in self._computed_groups]
        if not pending:
            return
        for iteration in range(self.config.max_iters):
            any_change = False
            for name, eqs in pending:
                changed = self._evaluate_group(
                    name,
                    eqs,
                    iteration=iteration,
                    tol=self.config.tol,
                    capture_timing=self.config.explain_timings,
                    changed_inputs=None,
                )
                any_change = any_change or changed
            if not any_change:
                break
        self._computed_groups.update(name for name, _ in pending)

    def _collect_requirements(self, target: str) -> Tuple[Set[str], Set[str]]:
        needed_groups: Set[str] = set()
        needed_sources: Set[str] = set()
        stack: List[str] = [target]
        visited: Set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in self._group_lookup:
                needed_groups.add(current)
                for dep in self._dependencies.get(current, set()):
                    if dep in self._group_lookup:
                        stack.append(dep)
                    elif dep in self._sources_lookup:
                        needed_sources.add(dep)
                    else:
                        # weight store or provided tensor
                        continue
            elif current in self._sources_lookup:
                needed_sources.add(current)
        return needed_groups, needed_sources

    def _ordered_subset(self, group_names: Set[str]) -> List[Tuple[str, List[Equation]]]:
        items = [(name, eqs) for name, eqs in self._groups if name in group_names]
        if self.config.chaining == "forward":
            return items
        return list(reversed(items))

    def _build_group_dependencies(self) -> Dict[str, Set[str]]:
        dependencies: Dict[str, Set[str]] = {}
        for name, eqs in self._groups:
            refs: Set[str] = set()
            for eq in eqs:
                refs.update(_collect_tensor_names(eq.rhs))
            refs.discard(name)
            dependencies[name] = refs
        return dependencies

    def _normalize_selector(
        self,
        name: str,
        selectors: Optional[Dict[str, Any]],
    ) -> Tuple[slice, ...]:
        lhs = self._lhs_lookup.get(name)
        if lhs is None:
            raise KeyError(f"Unknown tensor '{name}'")
        indices = [idx for idx in lhs.indices if idx not in lhs.rolling]
        normalized: List[slice] = []
        selectors = selectors or {}
        for _axis, idx_name in enumerate(indices):
            if idx_name in selectors:
                value = selectors[idx_name]
                if isinstance(value, slice):
                    normalized.append(value)
                else:
                    offset = int(value)
                    normalized.append(slice(offset, offset + 1))
            else:
                normalized.append(slice(None))
        return tuple(normalized)


# Standalone helpers -----------------------------------------------------------
def _first_tensor_ref(expr: Any) -> Optional[TensorRef]:
    if isinstance(expr, TensorRef):
        return expr
    if isinstance(expr, Term):
        for factor in expr.factors:
            found = _first_tensor_ref(factor)
            if found:
                return found
    if isinstance(expr, FuncCall):
        arg = expr.arg
        if isinstance(arg, tuple):
            for item in arg:
                found = _first_tensor_ref(item)
                if found:
                    return found
        elif arg is not None:
            return _first_tensor_ref(arg)
    return None


def _collect_tensor_names(expr: Any) -> Set[str]:
    names: Set[str] = set()
    if isinstance(expr, TensorRef):
        names.add(expr.name)
        return names
    if isinstance(expr, Term):
        for factor in expr.factors:
            names.update(_collect_tensor_names(factor))
        return names
    if isinstance(expr, FuncCall):
        arg = expr.arg
        if isinstance(arg, tuple):
            for item in arg:
                names.update(_collect_tensor_names(item))
        elif arg is not None:
            names.update(_collect_tensor_names(arg))
        for value in expr.kwargs.values():
            if isinstance(value, (TensorRef, Term, FuncCall)):
                names.update(_collect_tensor_names(value))
        return names
    return names


def _selector_key(
    selector: Tuple[slice, ...],
) -> Tuple[Tuple[Optional[int], Optional[int], Optional[int]], ...]:
    return tuple((sl.start, sl.stop, sl.step) for sl in selector)


def _axes_to_squeeze(selector: Tuple[slice, ...]) -> Tuple[int, ...]:
    axes: List[int] = []
    for axis, sl in enumerate(selector):
        if not isinstance(sl, slice):
            continue
        step = sl.step or 1
        if step != 1:
            continue
        start = sl.start
        stop = sl.stop
        if start is None or stop is None:
            continue
        if stop - start == 1:
            axes.append(axis)
    return tuple(axes)


def _shift_axis(array: np.ndarray, axis: int, offset: int) -> np.ndarray:
    if offset == 0:
        return array
    result = np.zeros_like(array)
    src = [slice(None)] * array.ndim
    dst = [slice(None)] * array.ndim
    if offset > 0:
        src[axis] = slice(offset, None)
        dst[axis] = slice(None, -offset)
    else:
        src[axis] = slice(None, offset)
        dst[axis] = slice(-offset, None)
    result[tuple(dst)] = array[tuple(src)]
    return result


def _factor_indices(factor: Any) -> List[str]:
    if isinstance(factor, TensorRef):
        return [idx for idx in factor.indices if idx not in factor.rolling]
    if isinstance(factor, FuncCall):
        tref = _first_tensor_ref(factor)
        if tref is None:
            return []
        return [idx for idx in tref.indices if idx not in tref.rolling]
    if isinstance(factor, IndexFunction):
        return [factor.axis]
    return []


def _factor_sort_rank(factor: Any) -> int:
    if isinstance(factor, TensorRef):
        return 0
    if isinstance(factor, FuncCall):
        return 1
    if isinstance(factor, IndexFunction):
        return 2
    return 3


def _canonical_factor_order(term: Term, mapping: Dict[str, str]) -> List[int]:
    keyed: List[Tuple[int, str, int, str, int]] = []
    for position, factor in enumerate(term.factors):
        indices = _factor_indices(factor)
        label = "".join(mapping[idx] for idx in indices)
        keyed.append(
            (
                len(indices),
                label,
                _factor_sort_rank(factor),
                repr(factor),
                position,
            )
        )
    keyed.sort()
    return [pos for (_, _, _, _, pos) in keyed]


def _normalized_einsum(
    term: Term, lhs: Optional[TensorRef]
) -> Tuple[str, List[str], List[int], Dict[str, str]]:
    lhs_indices = [idx for idx in lhs.indices if idx not in lhs.rolling] if lhs else []
    all_indices: List[str] = []
    for factor in term.factors:
        all_indices.extend(_factor_indices(factor))
    all_indices.extend(idx for idx in lhs_indices if idx not in all_indices)

    unique_sorted = sorted(set(all_indices))
    mapping: Dict[str, str] = {}
    for idx, sym in zip(unique_sorted, EINSUM_LABELS):
        mapping[idx] = sym

    factor_order = _canonical_factor_order(term, mapping)

    inputs: List[str] = []
    for position in factor_order:
        factor = term.factors[position]
        indices = _factor_indices(factor)
        inputs.append("".join(mapping[idx] for idx in indices))

    output = "".join(mapping[idx] for idx in lhs_indices if idx in mapping)
    projected = [idx for idx in unique_sorted if idx not in lhs_indices]
    equation = ",".join(inputs)
    if output:
        equation += f"->{output}"
    return equation, projected, factor_order, mapping
