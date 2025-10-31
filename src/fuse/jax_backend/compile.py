from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.builtins import read_tensor_from_file, write_tensor_to_file
from ..core.cache import CacheManager
from ..core.evaluator_numpy import (
    DemandNumpyRunner,
    ExecutionConfig,
    NumpyRunner,
    _factor_indices,
    _first_tensor_ref,
    _normalized_einsum,
)
from ..core.ir import (
    Equation,
    FuncCall,
    IndexFunction,
    ProgramIR,
    TensorRef,
    Term,
    equation_index_summary,
    format_index_summary,
)
from ..core.policies import RuntimePolicies
from ..core.stats import compute_einsum_stats
from ..core.temperature import coerce_temperature_value

try:
    import jax
    import jax.nn as jnn
    import jax.numpy as jnp
except Exception:  # pragma: no cover - jax optional
    jax = None
    jnp = None
    jnn = None

try:  # pragma: no cover - optional cache
    from jax.experimental import compilation_cache as _jax_compilation_cache
except Exception:  # pragma: no cover - cache optional
    _jax_compilation_cache = None

_COMPILATION_CACHE_ENABLED: Optional[str] = None


def _maybe_enable_compilation_cache(cfg: ExecutionConfig) -> None:
    global _COMPILATION_CACHE_ENABLED
    if not cfg.jax_enable_xla_cache or _COMPILATION_CACHE_ENABLED is not None:
        return
    if _jax_compilation_cache is None:
        return
    cache_dir = cfg.jax_cache_dir or str(Path.home() / ".cache" / "fuse" / "jax")
    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    try:
        _jax_compilation_cache.set_cache_dir(str(cache_path))
        _COMPILATION_CACHE_ENABLED = str(cache_path)
    except Exception:
        _COMPILATION_CACHE_ENABLED = ""


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if jnp is not None and isinstance(value, jnp.ndarray):
        material = jax.device_get(value) if jax is not None else value
        return material.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    return str(value)


def _resolve_device(device_spec: str):
    if jax is None:
        raise RuntimeError("JAX is not available")
    spec = (device_spec or "auto").strip().lower()
    if not spec or spec == "auto":
        return None
    index: Optional[int] = None
    if ":" in spec:
        base, index_str = spec.split(":", 1)
        spec = base
        if index_str:
            try:
                index = int(index_str)
            except ValueError as exc:  # pragma: no cover - invalid user input
                raise ValueError(f"Invalid device index in '{device_spec}'") from exc
    platform = spec
    if spec in {"cuda", "gpu", "mps"}:
        platform = "gpu"
    elif spec in {"cpu", "tpu"}:
        platform = spec
    else:
        raise ValueError(f"Unsupported JAX device spec '{device_spec}'")
    devices = jax.devices(platform)
    if not devices:
        raise ValueError(f"No JAX devices available for platform '{platform}'")
    if index is not None:
        for dev in devices:
            if getattr(dev, "id", None) == index:
                return dev
        raise ValueError(f"JAX device index {index} not found for platform '{platform}'")
    return devices[0]


def _device_platform(device) -> str:
    if device is not None:
        return getattr(device, "platform", "cpu")
    if jax is None:
        return "cpu"
    default_backend = getattr(jax, "default_backend", lambda: "cpu")
    return default_backend() or "cpu"


def _resolve_precision_dtype(precision: str, device) -> Any:
    if jnp is None:
        raise RuntimeError("JAX is not available")
    platform = _device_platform(device)
    prec = precision.lower()
    if prec == "fp32":
        return jnp.float32
    if prec == "bf16":
        if platform in {"cpu", "gpu", "tpu"}:
            return jnp.bfloat16
        raise ValueError(f"bf16 precision is not supported on JAX platform '{platform}'")
    if prec == "fp16":
        if platform in {"gpu"}:
            return jnp.float16
        raise ValueError(f"fp16 precision is not supported on JAX platform '{platform}'")
    if prec == "auto":
        if platform in {"gpu"}:
            return jnp.bfloat16
        if platform == "tpu":
            return jnp.bfloat16
        return jnp.float32
    raise ValueError(f"Unsupported precision '{precision}' for JAX backend")


def _jax_gelu_grad(x):
    x64 = jnp.asarray(x, dtype=jnp.float64)
    c = jnp.sqrt(2.0 / jnp.pi)
    inner = c * (x64 + 0.044715 * jnp.power(x64, 3))
    tanh_inner = jnp.tanh(inner)
    sech2 = 1.0 - jnp.power(tanh_inner, 2)
    grad64 = 0.5 * (1.0 + tanh_inner) + 0.5 * x64 * sech2 * (
        c * (1.0 + 0.134145 * jnp.power(x64, 2))
    )
    return grad64.astype(jnp.asarray(x).dtype)


def _jax_softmax_grad(y, grad, axis: int):
    dot = jnp.sum(grad * y, axis=axis, keepdims=True)
    return (grad - dot) * y


def _jax_layer_norm(x, axis: int, eps: float = 1e-5):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=axis, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _jax_masked_softmax(x, mask=None, axis: int = -1, fill_value=None):
    arr = jnp.asarray(x)
    mask_arr = None
    if mask is not None:
        mask_arr = jnp.asarray(mask).astype(bool)
        if fill_value is None:
            fill_value = jnp.finfo(arr.dtype).min
        else:
            fill_value = jnp.asarray(fill_value).reshape(())
        arr = jnp.where(mask_arr, arr, fill_value)
    if jnn is not None:
        result = jnn.softmax(arr, axis=axis)
    else:
        shifted = arr - jnp.max(arr, axis=axis, keepdims=True)
        exp_vals = jnp.exp(shifted)
        result = exp_vals / jnp.sum(exp_vals, axis=axis, keepdims=True)
    if mask_arr is not None:
        result = jnp.where(mask_arr, result, 0.0)
    return result


def _jax_attention(query, key, value, mask=None, scale=None, causal=False):
    q = jnp.asarray(query)
    k = jnp.asarray(key)
    v = jnp.asarray(value)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("attention requires query/key to share last dimension")
    if scale is None:
        scale_val = 1.0 / jnp.sqrt(q.shape[-1])
    else:
        scale_val = jnp.asarray(scale).reshape(())
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale_val
    mask_arr = None
    if mask is not None:
        mask_arr = jnp.asarray(mask).astype(bool)
    if causal:
        seq_len = scores.shape[-2]
        mem_len = scores.shape[-1]
        causal_mask = jnp.tril(jnp.ones((seq_len, mem_len), dtype=bool))
        mask_arr = causal_mask if mask_arr is None else jnp.logical_and(mask_arr, causal_mask)
    fill_value = jnp.finfo(scores.dtype).min
    weights = _jax_masked_softmax(scores, mask=mask_arr, axis=-1, fill_value=fill_value)
    return jnp.matmul(weights, v)


def _coerce_scalar(value) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if hasattr(value, "shape") and value.shape == () and hasattr(value, "reshape"):
        reshaped = value.reshape(())
        try:
            return float(reshaped)
        except TypeError:
            pass
    arr = jnp.asarray(value) if jnp is not None else np.asarray(value)
    if arr.size != 1:
        raise ValueError("Expected scalar value for threshold")
    return float(arr.reshape(()))


def _coerce_rank_spec(rank):
    if rank is None:
        return None
    if isinstance(rank, (int, float, np.integer, np.floating)):
        return int(rank)
    arr = jnp.asarray(rank) if jnp is not None else np.asarray(rank)
    if arr.size == 1:
        return int(arr.reshape(()))
    return [int(v) for v in arr.reshape(-1).tolist()]


def _normalize_tucker_ranks(rank, shape):
    if rank is None:
        return [max(1, min(dim, int(math.ceil(math.sqrt(dim))))) for dim in shape]
    if isinstance(rank, int):
        return [max(1, min(dim, rank)) for dim in shape]
    rank_list = list(rank)
    if len(rank_list) != len(shape):
        raise ValueError(
            f"rank specification must match tensor order (got {len(rank_list)} for {len(shape)})"
        )
    normalized = []
    for dim, item in zip(shape, rank_list):
        val = int(item)
        if val <= 0:
            raise ValueError("tucker_dense ranks must be positive")
        normalized.append(min(dim, val))
    return normalized


def _jax_mode_product(tensor, matrix, mode):
    moved = jnp.moveaxis(tensor, mode, 0)
    leading = moved.shape[0]
    rest = moved.shape[1:]
    flat = moved.reshape(leading, -1)
    result = jnp.matmul(matrix, flat)
    new_shape = (matrix.shape[0],) + rest
    return jnp.moveaxis(result.reshape(new_shape), 0, mode)


def _jax_tucker_dense(value, rank=None, threshold: float = 0.5, target_dtype=None):
    if jnp is None:
        raise RuntimeError("JAX is not available")
    dtype = target_dtype or jnp.float32
    x = jnp.asarray(value, dtype=jnp.float32)
    if x.ndim == 0:
        return jnp.asarray(x > threshold, dtype=jnp.float32).astype(dtype)
    ranks = _normalize_tucker_ranks(rank, x.shape)
    if all(r == dim for r, dim in zip(ranks, x.shape)):
        approx = x
    else:
        factors = []
        for mode, dim in enumerate(x.shape):
            unfolded = jnp.moveaxis(x, mode, 0).reshape(dim, -1)
            u, s, vh = jnp.linalg.svd(unfolded, full_matrices=False)
            r = min(ranks[mode], u.shape[1])
            factors.append(u[:, :r])
        core = x
        for mode, factor in enumerate(factors):
            core = _jax_mode_product(core, jnp.transpose(factor), mode)
        approx = core
        for mode, factor in enumerate(factors):
            approx = _jax_mode_product(approx, factor, mode)
    adjusted = approx - threshold
    return jnp.asarray(adjusted > 0, dtype=jnp.float32).astype(dtype)


def compile(
    program,
    device: str = "auto",
    mesh=None,
    cache_manager: Optional[CacheManager] = None,
    execution_config: Optional[ExecutionConfig] = None,
    policies: Optional[RuntimePolicies] = None,
    **_,
):
    cfg = (execution_config or ExecutionConfig()).normalized()
    device_spec = device if device != "auto" else cfg.device
    if cfg.device != device_spec:
        cfg = replace(cfg, device=device_spec).normalized()
    policy_obj = policies or RuntimePolicies()
    if cfg.mode == "demand":
        return DemandNumpyRunner(program.ir, config=cfg, policies=policy_obj)
    if cfg.projection_strategy == "monte_carlo":
        return NumpyRunner(program.ir, config=cfg, policies=policy_obj)
    _maybe_enable_compilation_cache(cfg)

    if jax is None:
        return NumpyRunner(program.ir, config=cfg, policies=policy_obj)
    if program.ir.has_streaming():
        return NumpyRunner(program.ir, config=cfg, policies=policy_obj)

    return JaxRunner(
        program=program,
        config=cfg,
        policies=policy_obj,
        cache_manager=cache_manager,
        mesh=mesh,
        device=cfg.device,
    )


class JaxRunner:
    def __init__(
        self,
        program,
        config: ExecutionConfig,
        policies: RuntimePolicies,
        cache_manager: Optional[CacheManager],
        mesh=None,
        device: str = "auto",
    ):
        self.program = program
        self.ir: ProgramIR = program.ir
        self.config = config.normalized()
        self.policies = policies
        self.cache_manager = cache_manager
        self.mesh = mesh
        self._device_spec = device
        self.device = None if jax is None else _resolve_device(device)
        self.zero_copy = self.config.zero_copy
        self.default_dtype = (
            jnp.float32
            if jnp is None
            else _resolve_precision_dtype(self.config.precision, self.device)
        )
        self.validate_transfers = self.config.validate_device_transfers
        self._xla_callable: Optional[Any] = None

        self.tensors: Dict[str, jnp.ndarray] = {}
        self.index_domains: Dict[str, int] = {}
        self.logs: List[Dict[str, Any]] = []
        self.boolean_tensors: Set[str] = self.ir.boolean_tensors()
        self._active_lhs: Optional[str] = None
        self._active_prev_value: Optional[jnp.ndarray] = None
        self._temperature_schedules = self.config.temperatures or {}
        self._last_temperatures: Dict[str, float] = {}
        self._active_equation_temperature: Optional[float] = None
        self._sig_temperatures: List[float] = []
        self._temperature_zero_tol = 1e-9

        self._sources: List[Equation] = []
        self._sinks: List[Equation] = []
        self._groups: List[Tuple[str, List[Equation]]] = []
        self._prepare()
        self.input_names: List[str] = [src.lhs.name for src in self._sources]

    # Public API -------------------------------------------------------------
    def __call__(
        self,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[ExecutionConfig] = None,
        policies: Optional[RuntimePolicies] = None,
    ):
        return self.run(inputs=inputs, config=config, policies=policies)

    @property
    def xla_callable(self):
        if self._xla_callable is None:
            self._xla_callable = self._build_xla_callable()
        return self._xla_callable

    def run(
        self,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[ExecutionConfig] = None,
        policies: Optional[RuntimePolicies] = None,
        skip_sinks: bool = False,
    ):
        prev_config = self.config
        cfg = (config or prev_config).normalized()
        if cfg.device != prev_config.device:
            raise ValueError(
                f"JaxRunner cannot switch devices at runtime (current={prev_config.device}, requested={cfg.device})"
            )
        policies_changed = False
        if policies is not None:
            self.policies = policies
            policies_changed = True

        config_changed = cfg != prev_config
        self.config = cfg
        self.zero_copy = cfg.zero_copy
        if jnp is not None:
            self.default_dtype = _resolve_precision_dtype(cfg.precision, self.device)
        self.validate_transfers = cfg.validate_device_transfers
        if config_changed or policies_changed:
            self._xla_callable = None
        self._temperature_schedules = self.config.temperatures or {}
        self._reset_state()
        if inputs:
            for name, value in inputs.items():
                self.tensors[name] = self._ensure_boolean_tensor(name, value)

        self.logs.clear()
        self._run_sources()

        if cfg.mode == "fixpoint":
            self._run_fixpoint(cfg)
        else:
            self._run_single_pass(cfg)

        if not skip_sinks:
            self._run_sinks()

        return {name: self.tensors.get(name) for name in self.ir.exports}

    def explain(self, *, json: bool = False):
        if json:
            return {"logs": [_json_ready(entry) for entry in self.logs]}
        lines = []
        for entry in self.logs:
            kind = entry.get("kind")
            if kind == "source":
                src = entry["source"]
                lines.append(f"[src] {src['name']} <- {src['path']} shape={tuple(src['shape'])}")
            elif kind == "equation":
                eq = entry["equation"]
                projected = eq.get("projected") or []
                details: List[str] = []
                einsum = eq.get("einsum")
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
                table = eq.get("index_table")
                if table:
                    details.append(f"idx[{table}]")
                note = f" {' '.join(details)}" if details else ""
                lines.append(f"[iter {eq['iteration']:02d}] {eq['name']} {eq['status']}{note}")
            elif kind == "sink":
                sk = entry["sink"]
                lines.append(f"[sink] {sk['path']} <- {sk['name']} ({sk['mode']})")
        return "\n".join(lines)

    # Preparation ------------------------------------------------------------
    def _prepare(self):
        group_map: Dict[str, List[Equation]] = {}
        for eq in self.ir.equations:
            if eq.is_source:
                self._sources.append(eq)
            elif eq.is_sink:
                self._sinks.append(eq)
            else:
                group_map.setdefault(eq.lhs.name, []).append(eq)

        seen = set()
        for eq in self.ir.equations:
            if eq.is_source or eq.is_sink:
                continue
            name = eq.lhs.name
            if name in seen:
                continue
            seen.add(name)
            self._groups.append((name, group_map[name]))

    def _reset_state(self):
        self.tensors = {}
        self.index_domains = {}
        self._last_temperatures.clear()
        self._active_equation_temperature = None
        self._sig_temperatures.clear()

    def _is_boolean_tensor(self, name: str) -> bool:
        return name in self.boolean_tensors

    def _as_array(self, value: Any, *, tensor_name: Optional[str] = None) -> jnp.ndarray:
        if jnp is None:
            raise RuntimeError("JAX is not available")
        if isinstance(value, jnp.ndarray):
            arr = value
            if arr.dtype != self.default_dtype:
                arr = arr.astype(self.default_dtype)
        else:
            material = value
            if isinstance(value, np.ndarray) and not self.zero_copy:
                material = np.array(value, copy=True)
            arr = jnp.asarray(material, dtype=self.default_dtype)
        arr_device = None
        if self.device is not None:
            dev_attr = getattr(arr, "device", None)
            arr_device = dev_attr() if callable(dev_attr) else dev_attr
            device_platform = getattr(self.device, "platform", None)
            if arr_device != self.device:
                if (
                    self.validate_transfers
                    and device_platform not in (None, "cpu")
                    and (arr_device is None or getattr(arr_device, "platform", "cpu") == "cpu")
                ):
                    name = tensor_name or "<unnamed>"
                    raise ValueError(
                        f"Tensor '{name}' provided on host for device '{self.device}'; "
                        "upload explicitly with jax.device_put or disable transfer validation."
                    )
                arr = jax.device_put(arr, self.device)
        return arr

    def _ensure_boolean_tensor(self, name: str, value: Any) -> jnp.ndarray:
        arr = self._as_array(value, tensor_name=name)
        if not self._is_boolean_tensor(name):
            return arr
        on = jnp.array(1, dtype=self.default_dtype)
        off = jnp.array(0, dtype=self.default_dtype)
        return jnp.where(arr > 0, on, off)

    # Sources / sinks --------------------------------------------------------
    def _run_sources(self):
        for eq in self._sources:
            name = eq.lhs.name
            mode = "loaded"
            if name in self.tensors:
                arr = self._ensure_boolean_tensor(name, self.tensors[name])
                mode = "provided"
            else:
                value = read_tensor_from_file(eq.src_file)
                arr = self._materialize_weight(name, value)
                arr = self._ensure_boolean_tensor(name, arr)
                self.tensors[name] = arr
            self._update_index_domains(eq.lhs, arr)
            self.logs.append(
                {
                    "kind": "source",
                    "source": {
                        "name": name,
                        "path": eq.src_file,
                        "shape": tuple(int(dim) for dim in getattr(arr, "shape", ())),
                        "mode": mode,
                    },
                }
            )

    def _run_sinks(self):
        for eq in self._sinks:
            val = self._eval(eq.rhs, lhs=None)
            mode = "tensor"
            if isinstance(val, tuple) and val and val[0] == "__topk__":
                _, array, k = val
                out = _jax_topk(array, k=k)
                mode = f"topk(k={k})"
            else:
                out = val
            material = jax.device_get(out) if jax is not None else out
            array = np.array(material)
            if eq.sink_file is None:
                raise ValueError("Sink equation missing target file path")
            target_path = self.policies.resolve_output_path(eq.sink_file)
            write_tensor_to_file(target_path, array)
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

    # Execution --------------------------------------------------------------
    def _run_single_pass(self, cfg: ExecutionConfig):
        groups = self._groups if cfg.chaining == "forward" else list(reversed(self._groups))
        for name, eqs in groups:
            self._evaluate_group(name, eqs, iteration=0, tol=cfg.tol)

    def _run_fixpoint(self, cfg: ExecutionConfig):
        groups = self._groups if cfg.chaining == "forward" else list(reversed(self._groups))
        for iteration in range(cfg.max_iters):
            changed = False
            for name, eqs in groups:
                step_changed = self._evaluate_group(name, eqs, iteration=iteration, tol=cfg.tol)
                changed = changed or step_changed
            if not changed:
                break

    def _evaluate_group(
        self,
        name: str,
        equations: Sequence[Equation],
        *,
        iteration: int,
        tol: float,
    ) -> bool:
        if not equations:
            return False

        contributions: List[jnp.ndarray] = []
        metas: List[Dict[str, Any]] = []
        lhs_ref = equations[0].lhs
        lhs_name = lhs_ref.name
        group_temperature = self._resolve_group_temperature(lhs_name, iteration)
        prev_value = self.tensors.get(lhs_name)
        had_prev = lhs_name in self.tensors
        running_total: Optional[jnp.ndarray] = None

        self._active_lhs = lhs_name
        self._active_prev_value = None
        if prev_value is not None and not lhs_ref.rolling:
            self._active_prev_value = jnp.asarray(prev_value)
        self._active_equation_temperature = group_temperature

        try:
            for eq in equations:
                self._sig_temperatures.clear()
                value, meta = self._eval_equation(eq)
                temps_used = list(self._sig_temperatures)
                effective_temp: Optional[Any]
                if temps_used:
                    effective_temp = temps_used[0] if len(temps_used) == 1 else temps_used
                elif group_temperature is not None:
                    effective_temp = group_temperature
                else:
                    effective_temp = None
                if effective_temp is not None:
                    meta = dict(meta)
                    meta["temperature"] = effective_temp
                value = self._ensure_boolean_tensor(lhs_name, value)
                contributions.append(value)
                metas.append(meta)
                if self._is_boolean_tensor(lhs_name):
                    running_total = (
                        value if running_total is None else jnp.maximum(running_total, value)
                    )
                else:
                    running_total = value if running_total is None else running_total + value
                if running_total is not None:
                    self.tensors[lhs_name] = running_total
        finally:
            self._active_lhs = None
            self._active_prev_value = None
            self._active_equation_temperature = None
            self._sig_temperatures.clear()

        total = running_total if running_total is not None else contributions[0]
        if self._is_boolean_tensor(lhs_name):
            total = self._ensure_boolean_tensor(lhs_name, total)

        if had_prev:
            self.tensors[lhs_name] = prev_value  # type: ignore[assignment]
        else:
            self.tensors.pop(lhs_name, None)

        lhs = equations[0].lhs
        changed = self._assign(lhs, total, tol)

        for idx, eq in enumerate(equations):
            status = (
                "update"
                if (changed and idx == len(equations) - 1)
                else ("unchanged" if (not changed and idx == len(equations) - 1) else "contrib")
            )
            self._log_equation(eq, metas[idx], iteration=iteration, status=status)
        return changed

    # Evaluation helpers -----------------------------------------------------
    def _eval_equation(self, eq: Equation) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        rhs = eq.rhs
        lhs_name = eq.lhs.name
        if isinstance(rhs, Term):
            value, meta = self._eval_term(
                rhs,
                lhs=eq.lhs,
                capture_meta=True,
                projection=eq.projection,
            )
            value = self._ensure_boolean_tensor(lhs_name, value)
            return value, meta
        if isinstance(rhs, FuncCall):
            value = self._eval_fn(rhs, lhs=eq.lhs)
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                value = self._ensure_boolean_tensor(lhs_name, value)
            return value, {"op": rhs.name}
        value = self._eval(rhs, lhs=eq.lhs)
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            value = self._ensure_boolean_tensor(lhs_name, value)
        return value, {}

    def _eval(self, expr: Any, lhs: Optional[TensorRef] = None):
        if isinstance(expr, TensorRef):
            use_prev = (
                self._active_lhs == expr.name
                and self._active_prev_value is not None
                and not expr.rolling
            )
            if use_prev:
                arr = self._active_prev_value
            else:
                if expr.name not in self.tensors:
                    resolved = None
                    if self.policies.weight_store is not None:
                        try:
                            resolved = self.policies.weight_store.resolve(expr.name)
                        except KeyError:
                            resolved = None
                    if resolved is not None:
                        arr = self._materialize_weight(expr.name, resolved)
                        arr = self._ensure_boolean_tensor(expr.name, arr)
                        self.tensors[expr.name] = arr
                        self._update_index_domains(expr, arr)
                    else:
                        raise KeyError(f"Tensor '{expr.name}' not yet defined")
                arr = self.tensors[expr.name]
            arr = self._ensure_boolean_tensor(expr.name, arr)
            arr = self._apply_index_specs(arr, expr)
            self._update_index_domains(expr, arr)
            return arr
        if isinstance(expr, Term):
            value, _ = self._eval_term(expr, lhs=lhs, capture_meta=False, projection="sum")
            return value
        if isinstance(expr, FuncCall):
            return self._eval_fn(expr, lhs=lhs)
        if isinstance(expr, IndexFunction):
            axis_lengths = self._collect_axis_lengths(lhs, [], [])
            dtype = self._resolve_term_dtype([])
            return self._eval_index_function(expr, axis_lengths, dtype)
        if isinstance(expr, (int, float, bool)):
            return jnp.asarray(expr)
        if isinstance(expr, np.ndarray):
            return jnp.asarray(expr)
        if isinstance(expr, str):
            return expr
        if expr is None:
            return None
        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _apply_index_specs(self, array: Any, ref: TensorRef):
        specs = getattr(ref, "index_specs", None)
        if not specs:
            return jnp.asarray(array)
        result = jnp.asarray(array)
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
                result = _shift_axis_jnp(result, axis_idx, spec.offset)
        return result

    def _collect_axis_lengths(
        self,
        lhs: Optional[TensorRef],
        factors: Sequence[Any],
        evaluated: Sequence[Any],
    ) -> Dict[str, int]:
        lengths: Dict[str, int] = dict(self.index_domains)
        if lhs is not None and lhs.name in self.tensors:
            lhs_arr = self.tensors[lhs.name]
            lhs_indices = [idx for idx in lhs.indices if idx not in lhs.rolling]
            for axis_idx, axis_name in enumerate(lhs_indices):
                if axis_idx < lhs_arr.ndim:
                    lengths.setdefault(axis_name, int(lhs_arr.shape[axis_idx]))
        for factor, value in zip(factors, evaluated):
            if value is None:
                continue
            indices = _factor_indices(factor)
            if not indices:
                continue
            arr = self._as_array(value)
            for axis_idx, axis_name in enumerate(indices):
                if axis_idx < arr.ndim:
                    lengths.setdefault(axis_name, int(arr.shape[axis_idx]))
        return lengths

    def _resolve_term_dtype(self, evaluated: Sequence[Any]):
        for value in evaluated:
            if value is None:
                continue
            arr = self._as_array(value)
            return arr.dtype
        return self.default_dtype

    def _eval_index_function(
        self,
        fn: IndexFunction,
        axis_lengths: Dict[str, int],
        dtype,
    ):
        length = axis_lengths.get(fn.axis)
        if length is None:
            raise ValueError(f"Axis '{fn.axis}' length unknown for index function '{fn.name}'")
        indices = jnp.arange(length, dtype=jnp.int32)
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
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        if projection not in {"sum", "max", "mean"}:
            raise ValueError(f"Unsupported projection op '{projection}'")

        base_equation, projected, factor_order, _ = _normalized_einsum(term, lhs)
        evaluated: List[Optional[Any]] = []
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

        def _ordered_arrays(order: Sequence[int]) -> List[jnp.ndarray]:
            return [self._as_array(evaluated[idx]) for idx in order]

        base_arrays = _ordered_arrays(factor_order)

        lhs_output_indices: List[str] = []
        if lhs is not None:
            lhs_output_indices = [idx for idx in lhs.indices if idx not in lhs.rolling]

        def _shapes_and_sizes(arrays: Sequence[Any]) -> Tuple[List[Tuple[int, ...]], List[int]]:
            shapes: List[Tuple[int, ...]] = []
            sizes: List[int] = []
            for arr in arrays:
                arr_shape = getattr(arr, "shape", ())
                shapes.append(tuple(int(dim) for dim in arr_shape))
                dtype = getattr(arr, "dtype", None)
                if dtype is None:
                    probe = jnp.asarray(arr) if jnp is not None else np.asarray(arr)
                    dtype = probe.dtype
                itemsize = getattr(dtype, "itemsize", None)
                if itemsize is None:
                    itemsize = int(np.dtype(dtype).itemsize)
                sizes.append(int(itemsize))
            return shapes, sizes

        if projection != "sum" and projected:
            if lhs is not None:
                extended_indices = list(lhs.indices)
                dotted_axes = list(lhs.dotted_axes)
                rolling = dict(lhs.rolling)
                name = lhs.name
            else:
                extended_indices = []
                dotted_axes = []
                rolling = {}
                name = "__tmp__"
            for idx in projected:
                if idx not in extended_indices:
                    extended_indices.append(idx)
            extended_lhs = TensorRef(
                name=name,
                indices=extended_indices,
                dotted_axes=dotted_axes,
                rolling=rolling,
            )
            extended_equation, _, extended_order, _ = _normalized_einsum(term, extended_lhs)
            extended_arrays = _ordered_arrays(extended_order or factor_order)
            raw = jnp.einsum(extended_equation, *extended_arrays)
            shapes, sizes = _shapes_and_sizes(extended_arrays)
            stats = compute_einsum_stats(
                extended_equation,
                shapes,
                sizes,
                tuple(int(dim) for dim in raw.shape),
                int(np.dtype(raw.dtype).itemsize),
            )
            axes_start = len(lhs_output_indices)
            reduce_axes = tuple(range(axes_start, axes_start + len(projected)))
            if not reduce_axes:
                result = raw
            elif projection == "max":
                result = jnp.max(raw, axis=reduce_axes)
            else:
                result = jnp.mean(raw, axis=reduce_axes)
            meta = {
                "einsum": extended_equation,
                "projected": projected,
                "projection": projection,
            }
            meta.update(stats)
        else:
            result = jnp.einsum(base_equation, *base_arrays)
            meta = {"einsum": base_equation, "projected": projected}
            if projection != "sum":
                meta["projection"] = projection
            shapes, sizes = _shapes_and_sizes(base_arrays)
            stats = compute_einsum_stats(
                base_equation,
                shapes,
                sizes,
                tuple(int(dim) for dim in result.shape),
                int(np.dtype(result.dtype).itemsize),
            )
            meta.update(stats)

        return (result, meta) if capture_meta else (result, {})

    def _eval_fn(self, fn: FuncCall, lhs: Optional[TensorRef] = None):
        name = fn.name.lower()
        args_expr = ()
        if isinstance(fn.arg, tuple):
            args_expr = fn.arg
        elif fn.arg is not None:
            args_expr = (fn.arg,)

        def eval_arg(expr):
            return self._eval(expr, lhs=lhs)

        def eval_array(expr):
            return self._as_array(eval_arg(expr))

        if name == "step":
            base = eval_array(args_expr[0])
            return (base > 0).astype(self.default_dtype)
        if name == "relu":
            arr = eval_array(args_expr[0])
            zero = jnp.array(0, dtype=self.default_dtype)
            return jnp.maximum(arr, zero)
        if name == "sig":
            if not args_expr:
                raise ValueError("sig requires at least one argument")
            arr = eval_array(args_expr[0])
            temperature: Optional[float] = None
            if len(args_expr) > 1:
                temperature = self._evaluate_temperature_operand(args_expr[1], lhs)
            elif "T" in fn.kwargs:
                temperature = self._evaluate_temperature_operand(fn.kwargs.get("T"), lhs)
            elif self._active_equation_temperature is not None:
                temperature = self._sanitize_temperature(self._active_equation_temperature)
            if temperature is None:
                temperature = 0.0
            self._sig_temperatures.append(temperature)
            if temperature == 0.0:
                return jnp.where(
                    arr > 0,
                    jnp.array(1, dtype=self.default_dtype),
                    jnp.array(0, dtype=self.default_dtype),
                )
            return jax.nn.sigmoid(arr / temperature).astype(self.default_dtype)
        if name == "gelu":
            return jax.nn.gelu(eval_array(args_expr[0])).astype(self.default_dtype)
        if name == "gelu_grad":
            return _jax_gelu_grad(eval_array(args_expr[0])).astype(self.default_dtype)
        if name == "lnorm":
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs) or -1
            )
            return _jax_layer_norm(eval_array(args_expr[0]), axis=axis).astype(self.default_dtype)
        if name in {"layernorm", "layer_norm"}:
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs) or -1
            )
            eps = float(fn.kwargs.get("eps", 1e-5))
            return _jax_layer_norm(eval_array(args_expr[0]), axis=axis, eps=eps).astype(
                self.default_dtype
            )
        if name == "softmax":
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs) or -1
            )
            mask_expr: Optional[Any] = None
            if len(args_expr) > 1:
                mask_expr = args_expr[1]
            elif "mask" in fn.kwargs:
                mask_expr = fn.kwargs.get("mask")
            logits = eval_array(args_expr[0])
            if mask_expr is not None:
                mask_value = self._as_array(self._eval(mask_expr, lhs=lhs)).astype(bool)
                fill_expr = fn.kwargs.get("fill")
                fill_value = self._eval(fill_expr, lhs=lhs) if fill_expr is not None else None
                return _jax_masked_softmax(
                    logits,
                    mask=mask_value,
                    axis=axis,
                    fill_value=fill_value,
                ).astype(self.default_dtype)
            if jnn is not None:
                return jnn.softmax(logits, axis=axis).astype(self.default_dtype)
            return _jax_masked_softmax(logits, mask=None, axis=axis, fill_value=None).astype(
                self.default_dtype
            )
        if name == "masked_softmax":
            if not args_expr:
                raise ValueError("masked_softmax requires logits argument")
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs) or -1
            )
            mask_expr: Optional[Any] = None
            if len(args_expr) > 1:
                mask_expr = args_expr[1]
            elif "mask" in fn.kwargs:
                mask_expr = fn.kwargs.get("mask")
            mask_value = (
                self._as_array(self._eval(mask_expr, lhs=lhs)).astype(bool)
                if mask_expr is not None
                else None
            )
            fill_expr = fn.kwargs.get("fill")
            fill_value = self._eval(fill_expr, lhs=lhs) if fill_expr is not None else None
            logits = eval_array(args_expr[0])
            return _jax_masked_softmax(
                logits,
                mask=mask_value,
                axis=axis,
                fill_value=fill_value,
            ).astype(self.default_dtype)
        if name == "softmax_grad":
            if len(args_expr) < 2:
                raise ValueError("softmax_grad requires probabilities and gradient arguments")
            probs = eval_array(args_expr[0])
            grad = eval_array(args_expr[1])
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs) or -1
            )
            return _jax_softmax_grad(probs, grad, axis).astype(self.default_dtype)
        if name == "sin":
            if not args_expr:
                raise ValueError("sin requires an argument")
            return jnp.sin(eval_array(args_expr[0])).astype(self.default_dtype)
        if name == "cos":
            if not args_expr:
                raise ValueError("cos requires an argument")
            return jnp.cos(eval_array(args_expr[0])).astype(self.default_dtype)
        if name == "rope":
            arr = eval_array(args_expr[0])
            d = arr.shape[-1]
            half = d // 2
            cos = jnp.cos(jnp.arange(half, dtype=arr.dtype))
            sin = jnp.sin(jnp.arange(half, dtype=arr.dtype))
            xr, xi = arr[..., :half], arr[..., half : half * 2]
            return jnp.concatenate([xr * cos - xi * sin, xr * sin + xi * cos], axis=-1)
        if name == "concat":
            if not args_expr:
                raise ValueError("concat requires an argument")
            arrays = [eval_array(arg) for arg in args_expr]
            axis_spec = fn.kwargs.get("axis")
            if len(arrays) == 1:
                arr = arrays[0]
                if arr.ndim < 2:
                    return arr
                shape = arr.shape[:-2] + (arr.shape[-2] * arr.shape[-1],)
                return jnp.reshape(arr, shape)
            axis = self._axis_from_spec(axis_spec, args_expr[0], lhs, default=-1)
            return jnp.concatenate(arrays, axis=axis)
        if name in {"max", "amax"}:
            arr = eval_array(args_expr[0])
            axis = self._axis_from_spec(fn.kwargs.get("axis"), args_expr[0], lhs, default=-1)
            keepdims = bool(fn.kwargs.get("keepdims", False))
            return jnp.max(arr, axis=axis, keepdims=keepdims)
        if name in {"avg", "mean"}:
            arr = eval_array(args_expr[0])
            axis = self._axis_from_spec(fn.kwargs.get("axis"), args_expr[0], lhs, default=-1)
            keepdims = bool(fn.kwargs.get("keepdims", False))
            return jnp.mean(arr, axis=axis, keepdims=keepdims)
        if name == "causal_mask":
            axis_expr = args_expr[0] if args_expr else fn.arg
            L = int(axis_expr)
            return jnp.tril(jnp.ones((L, L), dtype=self.default_dtype))
        if name == "const":
            value = args_expr[0] if args_expr else fn.arg
            return self._as_array(value)
        if name == "attention":
            if len(args_expr) < 3:
                raise ValueError("attention requires query, key, and value arguments")
            query = eval_array(args_expr[0])
            key = eval_array(args_expr[1])
            value = eval_array(args_expr[2])
            mask_expr: Optional[Any] = None
            if len(args_expr) > 3:
                mask_expr = args_expr[3]
            elif "mask" in fn.kwargs:
                mask_expr = fn.kwargs.get("mask")
            mask_value = (
                self._as_array(self._eval(mask_expr, lhs=lhs)).astype(bool)
                if mask_expr is not None
                else None
            )
            scale_expr = fn.kwargs.get("scale")
            scale_value = self._eval(scale_expr, lhs=lhs) if scale_expr is not None else None
            if isinstance(scale_value, str):
                scale_value = self.tensors.get(scale_value)
            causal = bool(fn.kwargs.get("causal", False))
            return _jax_attention(
                query, key, value, mask=mask_value, scale=scale_value, causal=causal
            ).astype(self.default_dtype)
        if name == "tucker_dense":
            if not args_expr:
                raise ValueError("tucker_dense requires a tensor argument")
            base = eval_array(args_expr[0])
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
            rank_spec = _coerce_rank_spec(rank_value)
            threshold = _coerce_scalar(threshold_value)
            return _jax_tucker_dense(
                base, rank=rank_spec, threshold=threshold, target_dtype=self.default_dtype
            )
        if name == "topk":
            arr = eval_array(args_expr[0])
            k = int(fn.kwargs.get("k", 5))
            return ("__topk__", arr, k)
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
                result = self._as_array(self._eval(default_expr, lhs=lhs))
            for cond_expr, value_expr in pairs:
                value = self._as_array(self._eval(value_expr, lhs=lhs))
                cond = self._as_array(self._eval(cond_expr, lhs=lhs)).astype(bool)
                if result is None:
                    result = jnp.zeros_like(value)
                result = jnp.where(cond, value, result)
            if result is None:
                raise ValueError("case evaluation produced no result")
            return result
        raise ValueError(f"Unknown jax builtin: {fn.name}")

    # Assignment / logging ---------------------------------------------------
    def _assign(self, ref: TensorRef, value: jnp.ndarray, tol: float) -> bool:
        normalized = self._ensure_boolean_tensor(ref.name, value)
        prev = self.tensors.get(ref.name)
        self.tensors[ref.name] = normalized
        self._update_index_domains(ref, normalized)
        if prev is None:
            return True
        prev_comp = self._ensure_boolean_tensor(ref.name, prev)
        if prev_comp.shape != normalized.shape:
            return True
        if prev_comp.dtype.kind in ("f", "c") or normalized.dtype.kind in ("f", "c"):
            return not bool(jnp.allclose(prev_comp, normalized, atol=tol, rtol=1e-5))
        return not jnp.array_equal(prev_comp, normalized)

    def _log_equation(
        self,
        eq: Equation,
        meta: Dict[str, Any],
        *,
        iteration: int,
        status: str,
    ):
        projected = meta.get("projected", []) or []
        index_summary = equation_index_summary(eq, projected)
        self.logs.append(
            {
                "kind": "equation",
                "equation": {
                    "name": eq.lhs.name,
                    "status": status,
                    "iteration": iteration,
                    "duration_ms": None,
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
        )

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

    # Utilities --------------------------------------------------------------
    def _update_index_domains(self, ref: TensorRef, arr):
        try:
            shape = tuple(getattr(arr, "shape", ()))
        except Exception:
            shape = ()
        if not shape:
            material = jnp.asarray(arr) if jnp is not None else np.asarray(arr)
            shape = tuple(int(dim) for dim in material.shape)
        for i, idx in enumerate(ref.indices):
            if i < len(shape):
                self.index_domains[idx] = max(self.index_domains.get(idx, 0), int(shape[i]))

    def _materialize_weight(self, name: str, payload: Any) -> jnp.ndarray:
        arr = self.policies.materialize_weight(
            name,
            payload,
            backend="jax",
            device=None,
        )
        return self._as_array(arr, tensor_name=name)

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
            return default if default is not None else -1
        if isinstance(spec, (int, float)):
            return int(spec)
        if isinstance(spec, str):
            if lhs and spec in lhs.indices:
                return lhs.indices.index(spec)
            tref = _first_tensor_ref(arg_expr)
            if tref and spec in tref.indices:
                return tref.indices.index(spec)
        raise ValueError(f"Cannot resolve axis specification: {spec}")

    # XLA callable -----------------------------------------------------------
    def _build_xla_callable(self):
        if jax is None:
            return None

        export_names = tuple(self.ir.exports)

        def _export(inputs):
            if inputs is None:
                runtime_inputs: Dict[str, Any] = {}
            else:
                runtime_inputs = dict(inputs)
            cfg = replace(self.config, mode="single")
            outputs = self.run(inputs=runtime_inputs, config=cfg, skip_sinks=True)
            return tuple(outputs[name] for name in export_names)

        try:
            return jax.jit(_export)
        except Exception:
            return None


def _shift_axis_jnp(array: jnp.ndarray, axis: int, offset: int) -> jnp.ndarray:
    if offset == 0:
        return array
    result = jnp.zeros_like(array)
    src = [slice(None)] * array.ndim
    dst = [slice(None)] * array.ndim
    if offset > 0:
        src[axis] = slice(offset, None)
        dst[axis] = slice(None, -offset)
    else:
        src[axis] = slice(None, offset)
        dst[axis] = slice(-offset, None)
    return result.at[tuple(dst)].set(array[tuple(src)])


def _jax_topk(array: jnp.ndarray, k: int):
    values, indices = jax.lax.top_k(array, k)
    values_host = jax.device_get(values) if jax is not None else np.array(values)
    indices_host = jax.device_get(indices) if jax is not None else np.array(indices)
    values_arr = np.asarray(values_host)
    indices_arr = np.asarray(indices_host)
    if values_arr.ndim == 1:
        return [(int(indices_arr[i]), float(values_arr[i])) for i in range(values_arr.shape[0])]
    results = []
    for row_idx in range(values_arr.shape[0]):
        row = []
        for col in range(values_arr.shape[1]):
            row.append((int(indices_arr[row_idx, col]), float(values_arr[row_idx, col])))
        results.append(row)
    return results
