from __future__ import annotations

import inspect
import io
import math
from contextlib import nullcontext as _nullcontext
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..core.builtins import read_tensor_from_file, write_tensor_to_file
from ..core.cache import CacheManager, cache_fingerprint, cache_key_from_fingerprint
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
    import torch
    import torch.nn.functional as F
    from torch.fx import GraphModule, symbolic_trace
    from torch.fx.passes.shape_prop import ShapeProp

    try:
        from torch.fx.proxy import Proxy as FxProxy  # type: ignore
    except Exception:  # pragma: no cover - optional import path differences
        FxProxy = None  # type: ignore
except Exception:  # pragma: no cover - torch optional
    torch = None
    F = None
    GraphModule = None
    ShapeProp = None
    FxProxy = None  # type: ignore


def compile(
    program,
    device: str = "auto",
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

    if torch is None:
        return NumpyRunner(program.ir, config=cfg, policies=policy_obj)
    if program.ir.has_streaming():
        return NumpyRunner(program.ir, config=cfg, policies=policy_obj)

    return TorchRunner(
        program=program,
        device=cfg.device,
        config=cfg,
        policies=policy_obj,
        cache_manager=cache_manager,
    )


# --------------------------------------------------------------------------- #
# Torch runtime
# --------------------------------------------------------------------------- #


def _resolve_device(device_spec: str) -> torch.device:
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    spec = device_spec or "auto"
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device but torch.cuda.is_available() is False")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("Requested MPS device but torch.backends.mps.is_available() is False")
    return device


def _cuda_bf16_supported() -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _resolve_precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    prec = precision.lower()
    if prec == "fp32":
        return torch.float32
    if prec == "bf16":
        if device.type == "cuda":
            if not _cuda_bf16_supported():
                raise ValueError("bf16 precision requires CUDA hardware with bf16 support")
            return torch.bfloat16
        if device.type == "cpu":
            return torch.bfloat16
        raise ValueError(f"bf16 precision is not supported on device type '{device.type}'")
    if prec == "fp16":
        if device.type in {"cuda", "mps"}:
            return torch.float16
        raise ValueError(f"fp16 precision is not supported on device type '{device.type}'")
    if prec == "auto":
        if device.type == "cuda":
            if _cuda_bf16_supported():
                return torch.bfloat16
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if device.type == "mps":
            return torch.float16
        return torch.float32
    raise ValueError(f"Unsupported precision '{precision}' for Torch backend")


def _torch_step(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(dtype=x.dtype)


def _torch_sig(x: torch.Tensor, T: Optional[Any], zero_tol: float = 1e-9) -> torch.Tensor:
    if T is None:
        return (x > 0).to(dtype=x.dtype)
    temp = float(T)
    if abs(temp) <= zero_tol:
        return (x > 0).to(dtype=x.dtype)
    return torch.sigmoid(x / temp)


SQRT1_2 = 1.0 / math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _torch_gelu(x: torch.Tensor) -> torch.Tensor:
    # Check if the device supports float64
    device = x.device
    supports_float64 = True
    if device.type == 'mps':
        # MPS doesn't support float64
        supports_float64 = False
    elif device.type == 'cuda':
        # Check if CUDA device supports float64
        try:
            torch.tensor([1.0], device=device, dtype=torch.float64)
            supports_float64 = True
        except (RuntimeError, TypeError):
            supports_float64 = False

    if supports_float64 and x.dtype != torch.float64:
        # Use float64 for higher precision if supported
        x64 = x.to(dtype=torch.float64)
        y64 = 0.5 * x64 * (1.0 + torch.erf(x64 * SQRT1_2))
        return y64.to(dtype=x.dtype)
    else:
        # Stay in current dtype (likely float32)
        return 0.5 * x * (1.0 + torch.erf(x * SQRT1_2))


def _coerce_scalar(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected scalar tensor")
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("Expected scalar ndarray")
        return float(value.reshape(()))
    return float(value)


def _coerce_rank_spec(rank: Any) -> Optional[Union[int, List[int]]]:
    if rank is None:
        return None
    if isinstance(rank, (int, float, np.integer, np.floating)):
        return int(rank)
    if torch is not None and isinstance(rank, torch.Tensor):
        if rank.numel() == 1:
            return int(rank.detach().cpu().item())
        return [int(v) for v in rank.detach().cpu().reshape(-1).tolist()]
    if isinstance(rank, np.ndarray):
        if rank.size == 1:
            return int(rank.reshape(()))
        return [int(v) for v in rank.reshape(-1).tolist()]
    if isinstance(rank, (list, tuple)):
        return [int(v) for v in rank]
    raise ValueError(f"Unsupported rank specification for tucker_dense: {rank!r}")


def _normalize_tucker_ranks(
    rank: Optional[Union[int, Sequence[int]]], shape: Sequence[int]
) -> List[int]:
    if rank is None:
        return [max(1, min(dim, int(math.ceil(math.sqrt(dim))))) for dim in shape]
    if isinstance(rank, int):
        return [max(1, min(dim, rank)) for dim in shape]
    rank_list = list(rank)
    if len(rank_list) != len(shape):
        raise ValueError(
            f"rank specification must match tensor order (got {len(rank_list)} for {len(shape)})"
        )
    normalized: List[int] = []
    for dim, item in zip(shape, rank_list):
        val = int(item)
        if val <= 0:
            raise ValueError("tucker_dense ranks must be positive")
        normalized.append(min(dim, val))
    return normalized


def _torch_mode_product(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    moved = torch.movedim(tensor, mode, 0)
    leading = moved.shape[0]
    rest = moved.shape[1:]
    flat = moved.reshape(leading, -1)
    if matrix.shape[1] != leading:
        raise ValueError(f"Matrix shape {matrix.shape} incompatible with axis size {leading}")
    result = matrix @ flat
    new_shape = (matrix.shape[0],) + rest
    return torch.movedim(result.reshape(new_shape), 0, mode)


def _torch_tucker_dense(
    tensor: torch.Tensor,
    *,
    rank: Optional[Union[int, Sequence[int]]] = None,
    threshold: float = 0.5,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    x = tensor.to(dtype=torch.float32)
    if x.ndim == 0:
        adjusted = x - torch.as_tensor(float(threshold), device=x.device, dtype=x.dtype)
        return _torch_step(adjusted).to(dtype=target_dtype)
    ranks = _normalize_tucker_ranks(rank, x.shape)
    if all(r == dim for r, dim in zip(ranks, x.shape)):
        approx = x
    else:
        factors: List[torch.Tensor] = []
        for mode, dim in enumerate(x.shape):
            unfolded = torch.movedim(x, mode, 0).reshape(dim, -1)
            try:
                u, _, _ = torch.linalg.svd(unfolded, full_matrices=False)
            except RuntimeError:
                cpu_unfolded = unfolded.detach().cpu()
                u_cpu, _, _ = torch.linalg.svd(cpu_unfolded, full_matrices=False)
                u = u_cpu.to(unfolded.device)
            r = min(ranks[mode], u.shape[1])
            factors.append(u[:, :r].contiguous())
        core = x
        for mode, factor in enumerate(factors):
            core = _torch_mode_product(core, factor.transpose(0, 1), mode)
        approx = core
        for mode, factor in enumerate(factors):
            approx = _torch_mode_product(approx, factor, mode)
    adjusted = approx - torch.as_tensor(float(threshold), device=x.device, dtype=approx.dtype)
    return _torch_step(adjusted).to(dtype=target_dtype)


# Torch Inductor/Triton helpers -------------------------------------------------
_TORCH_COMPILE = getattr(torch, "compile", None) if torch is not None else None
_MASKED_SOFTMAX_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    return str(value)


def _maybe_torch_compile(fn):
    if _TORCH_COMPILE is None:
        return fn
    try:
        return _TORCH_COMPILE(fn, dynamic=True)
    except Exception:
        return fn


def _ensure_scalar_tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected scalar tensor for fill value")
        return value.to(dtype=dtype, device=device).reshape(())
    return torch.tensor(float(value), dtype=dtype, device=device)


def _torch_layer_norm(x: torch.Tensor, axis: int, eps: float = 1e-5) -> torch.Tensor:
    if axis < 0:
        axis += x.dim()
    if axis < 0 or axis >= x.dim():
        raise ValueError(f"LayerNorm axis {axis} out of range for tensor with {x.dim()} dims")
    if axis == x.dim() - 1:
        normalized_shape = x.shape[-1:]
        return F.layer_norm(x, normalized_shape, eps=eps)
    perm = list(range(x.dim()))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    inv_perm = [0] * len(perm)
    for idx, val in enumerate(perm):
        inv_perm[val] = idx
    transposed = x.permute(*perm)
    normalized = F.layer_norm(transposed, transposed.shape[-1:], eps=eps)
    return normalized.permute(*inv_perm)


def _torch_masked_softmax(
    x: torch.Tensor, mask: Optional[torch.Tensor], *, dim: int, fill_value: Optional[Any]
) -> torch.Tensor:
    if not torch.is_floating_point(x):
        raise ValueError("masked_softmax requires floating point inputs")
    if mask is None:
        return F.softmax(x, dim=dim)
    if mask.dtype != torch.bool or mask.device != x.device:
        mask = mask.to(dtype=torch.bool, device=x.device)
    if mask.shape != x.shape:
        raise ValueError("masked_softmax mask must match input shape")
    if fill_value is None:
        finfo = torch.finfo(x.dtype)
        fill_value = finfo.min
    fill_tensor = _ensure_scalar_tensor(fill_value, dtype=x.dtype, device=x.device)
    cache_key = (tuple(x.shape), x.dtype, mask.dtype, dim)
    fn = _MASKED_SOFTMAX_CACHE.get(cache_key)
    if fn is None:

        def _masked_softmax_impl(
            scores: torch.Tensor, mask_tensor: torch.Tensor, fill_scalar: torch.Tensor
        ) -> torch.Tensor:
            logits = scores.masked_fill(~mask_tensor, fill_scalar)
            probs = F.softmax(logits, dim=dim)
            return probs.masked_fill(~mask_tensor, 0.0)

        fn = _maybe_torch_compile(_masked_softmax_impl)
        _MASKED_SOFTMAX_CACHE[cache_key] = fn
    return fn(x, mask, fill_tensor)


if F is not None and hasattr(F, "scaled_dot_product_attention"):
    try:
        _HAS_SDP_SCALE = "scale" in inspect.signature(F.scaled_dot_product_attention).parameters
    except (ValueError, TypeError):  # builtins without signature metadata
        _HAS_SDP_SCALE = False
else:
    _HAS_SDP_SCALE = False


def _promote_attention_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    original_dim = tensor.dim()
    if original_dim == 4:
        return tensor, original_dim
    if original_dim == 3:
        return tensor.unsqueeze(1), original_dim
    if original_dim == 2:
        return tensor.unsqueeze(0).unsqueeze(0), original_dim
    raise ValueError("attention expects tensors with rank >= 2")


def _restore_attention_tensor(tensor: torch.Tensor, original_dim: int) -> torch.Tensor:
    if original_dim == 4:
        return tensor
    if original_dim == 3:
        return tensor.squeeze(1)
    if original_dim == 2:
        return tensor.squeeze(0).squeeze(0)
    raise ValueError("Invalid original shape for attention restore")


def _scalar_from_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Attention scale must be a scalar tensor")
        return float(value.detach().cpu().reshape(()).item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("Attention scale must be a scalar array")
        return float(value.reshape(()))
    return float(value)


def _torch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: Optional[torch.Tensor],
    scale: Optional[Any],
    causal: bool,
) -> torch.Tensor:
    if F is None or not hasattr(F, "scaled_dot_product_attention"):
        raise RuntimeError("Scaled dot-product attention requires torch.nn.functional")
    q, q_dim = _promote_attention_tensor(query)
    k, _ = _promote_attention_tensor(key)
    v, _ = _promote_attention_tensor(value)
    attn_mask = None
    if mask is not None:
        attn_mask = mask
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask != 0
        while attn_mask.dim() < 4:
            attn_mask = attn_mask.unsqueeze(0)
        attn_mask = attn_mask.to(device=q.device)
    scale_value = _scalar_from_value(scale)
    scaled_query = q
    if scale_value is not None and not _HAS_SDP_SCALE:
        head_dim = q.shape[-1]
        scaled_query = q * (scale_value * math.sqrt(head_dim))
        scale_value = None
    use_cuda_sdpa_ctx = (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "sdp_kernel")
        and q.is_cuda
        and q.dtype in {torch.float16, torch.bfloat16, torch.float32}
    )
    try:
        if use_cuda_sdpa_ctx:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=True, enable_math=False
            ):
                try:
                    result = F.scaled_dot_product_attention(
                        scaled_query,
                        k,
                        v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=scale_value if _HAS_SDP_SCALE else None,
                    )
                except TypeError:
                    result = F.scaled_dot_product_attention(
                        scaled_query,
                        k,
                        v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                    )
        else:
            try:
                result = F.scaled_dot_product_attention(
                    scaled_query,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=causal,
                    scale=scale_value if _HAS_SDP_SCALE else None,
                )
            except TypeError:
                result = F.scaled_dot_product_attention(
                    scaled_query,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=causal,
                )
    except Exception:
        # Retry with math fallback if optimized kernels are unavailable
        math_ctx = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )  # type: ignore[attr-defined]
            if hasattr(torch, "backends")
            and hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "sdp_kernel")
            and q.is_cuda
            else _nullcontext()
        )
        with math_ctx:
            try:
                result = F.scaled_dot_product_attention(
                    scaled_query,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=causal,
                    scale=scale_value if _HAS_SDP_SCALE else None,
                )
            except TypeError:
                result = F.scaled_dot_product_attention(
                    scaled_query,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=causal,
                )
    return _restore_attention_tensor(result, q_dim)


def _torch_lnorm(x: torch.Tensor, axis: int, eps: float = 1e-5) -> torch.Tensor:
    return _torch_layer_norm(x, axis=axis, eps=eps)


def _torch_rope(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    cos = torch.cos(torch.arange(half, device=x.device, dtype=x.dtype))
    sin = torch.sin(torch.arange(half, device=x.device, dtype=x.dtype))
    xr, xi = x[..., :half], x[..., half : half * 2]
    return torch.cat([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1)


def _torch_concat(arrays: List[torch.Tensor], axis: Optional[int]) -> torch.Tensor:
    if len(arrays) == 1 and axis is None:
        arr = arrays[0]
        if arr.ndim < 2:
            return arr
        shape = arr.shape[:-2] + (arr.shape[-2] * arr.shape[-1],)
        return arr.reshape(shape)
    axis = -1 if axis is None else axis
    return torch.cat(arrays, dim=axis)


def _torch_reduce_max(x: torch.Tensor, axis: int, keepdims: bool) -> torch.Tensor:
    values, _ = torch.max(x, dim=axis, keepdim=keepdims)
    return values


def _torch_reduce_mean(x: torch.Tensor, axis: int, keepdims: bool) -> torch.Tensor:
    return torch.mean(x, dim=axis, keepdim=keepdims)


def _torch_causal_mask(
    L: int,
    device: torch.device,
    *,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.tril(torch.ones((L, L), dtype=torch.int8, device=device))
    return mask.to(dtype=target_dtype)


def _torch_const(
    val: Any,
    device: torch.device,
    *,
    target_dtype: torch.dtype,
    zero_copy: bool,
) -> torch.Tensor:
    return _tensor_from_numpy_safe(
        np.asarray(val),
        device=device,
        target_dtype=target_dtype,
        zero_copy=zero_copy,
    )


def _torch_gelu_grad(x: torch.Tensor) -> torch.Tensor:
    # Check if the device supports float64
    device = x.device
    supports_float64 = True
    if device.type == 'mps':
        # MPS doesn't support float64
        supports_float64 = False
    elif device.type == 'cuda':
        # Check if CUDA device supports float64
        try:
            torch.tensor([1.0], device=device, dtype=torch.float64)
            supports_float64 = True
        except (RuntimeError, TypeError):
            supports_float64 = False

    if supports_float64 and x.dtype != torch.float64:
        # Use float64 for higher precision if supported
        x64 = x.to(dtype=torch.float64)
        cdf64 = 0.5 * (1.0 + torch.erf(x64 * SQRT1_2))
        pdf64 = torch.exp(-0.5 * x64 * x64) * INV_SQRT_2PI
        grad64 = cdf64 + x64 * pdf64
        return grad64.to(dtype=x.dtype)
    else:
        # Stay in current dtype (likely float32)
        cdf = 0.5 * (1.0 + torch.erf(x * SQRT1_2))
        pdf = torch.exp(-0.5 * x * x) * INV_SQRT_2PI
        return cdf + x * pdf


def _torch_softmax_grad(y: torch.Tensor, grad: torch.Tensor, axis: int) -> torch.Tensor:
    dot = torch.sum(grad * y, dim=axis, keepdim=True)
    return (grad - dot) * y


def _tensor_from_numpy_safe(
    arr: np.ndarray,
    device: torch.device,
    *,
    target_dtype: torch.dtype,
    zero_copy: bool,
) -> torch.Tensor:
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype not in (np.float16, np.float32):
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if zero_copy and device.type == "cpu":
        base = torch.from_numpy(arr)
    else:
        base = torch.as_tensor(arr)
    return base.to(device=device, dtype=target_dtype)


def _annotate_fx_graph(module: Optional[GraphModule]) -> None:
    if module is None or GraphModule is None or torch is None:
        return
    if ShapeProp is not None:
        try:
            with torch.no_grad():
                ShapeProp(module).propagate()
        except Exception:
            pass
    seen_names: Dict[Tuple[str, str], int] = {}

    for node in module.graph.nodes:
        original_name = node.meta.setdefault("fuse_original_name", node.name)
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None:
            continue
        dtype = getattr(tensor_meta, "dtype", None)
        shape = getattr(tensor_meta, "shape", None)
        if dtype is None or shape is None:
            continue
        dtype_str = str(dtype).replace("torch.", "")
        try:
            shape_iter = tuple(int(dim) for dim in shape)
        except TypeError:
            shape_iter = tuple(int(dim) for dim in tuple(shape))
        shape_str = "scalar" if len(shape_iter) == 0 else "x".join(str(dim) for dim in shape_iter)
        suffix = f"{dtype_str}_{shape_str}"
        base_name = original_name
        count = seen_names.get((base_name, suffix), 0)
        if count == 0:
            new_name = f"{base_name}__{suffix}"
        else:
            new_name = f"{base_name}__{suffix}_{count}"
        seen_names[(base_name, suffix)] = count + 1
        node.name = new_name
    module.graph.lint()
    module.recompile()


def _shift_axis_tensor(tensor: torch.Tensor, axis: int, offset: int) -> torch.Tensor:
    if offset == 0:
        return tensor
    result = torch.zeros_like(tensor)
    src = [slice(None)] * tensor.dim()
    dst = [slice(None)] * tensor.dim()
    if offset > 0:
        src[axis] = slice(offset, None)
        dst[axis] = slice(None, -offset)
    else:
        src[axis] = slice(None, offset)
        dst[axis] = slice(-offset, None)
    result[tuple(dst)] = tensor[tuple(src)]
    return result


class TorchRunner:
    def __init__(
        self,
        program,
        device: str,
        config: ExecutionConfig,
        policies: RuntimePolicies,
        cache_manager: Optional[CacheManager],
    ):
        self.program = program
        self.ir: ProgramIR = program.ir
        self.config = config.normalized()
        self.policies = policies
        self.device = _resolve_device(device)
        self.zero_copy = self.config.zero_copy
        self.default_dtype = _resolve_precision_dtype(self.config.precision, self.device)
        self.cache_manager = cache_manager

        self.tensors: Dict[str, torch.Tensor] = {}
        self.index_domains: Dict[str, int] = {}
        self.logs: List[Dict[str, Any]] = []
        self.boolean_tensors: Set[str] = self.ir.boolean_tensors()
        self._active_lhs: Optional[str] = None
        self._active_prev_value: Optional[torch.Tensor] = None
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
        self.fx_input_names: List[str] = self._compute_fx_input_names()

        # Hint for faster fp32 matmul kernels on CUDA
        try:
            if torch is not None and self.device.type == "cuda":
                set_prec = getattr(torch, "set_float32_matmul_precision", None)
                if callable(set_prec):
                    set_prec("high")
        except Exception:
            pass

        # Build/load FX graph and optionally wrap with torch.compile
        self._fx_sig_key: Optional[Tuple[Any, ...]] = None
        self.fx_module: Optional[GraphModule] = self._load_or_build_fx()
        self._fx_callable = None
        if self.fx_module is not None:
            try:
                compiled = _maybe_torch_compile(self.fx_module)
                self._fx_callable = compiled
            except Exception:
                self._fx_callable = self.fx_module

    # Public API -------------------------------------------------------------
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
        skip_sinks: bool = False,
    ):
        prev_config = self.config
        cfg = (config or prev_config).normalized()
        if cfg.device != prev_config.device:
            raise ValueError(
                f"TorchRunner cannot switch devices at runtime (current={prev_config.device}, requested={cfg.device})"
            )
        if policies is not None:
            self.policies = policies
        self.config = cfg
        self.zero_copy = cfg.zero_copy
        self.default_dtype = _resolve_precision_dtype(cfg.precision, self.device)
        self._temperature_schedules = self.config.temperatures or {}
        self._reset_state()

        if inputs:
            for name, value in inputs.items():
                tensor = self._as_tensor(value)
                tensor = self._ensure_boolean_tensor(name, tensor)
                self.tensors[name] = tensor

        self.logs.clear()
        self._run_sources()

        # If runtime inputs are provided, rebuild FX for this signature when needed
        fx_sig: Optional[Tuple[Any, ...]] = None
        input_feed_for_trace: Optional[Dict[str, Any]] = None
        if inputs:
            sig_items: List[Tuple[str, Tuple[int, ...], str]] = []
            input_feed_for_trace = {}
            for name, val in inputs.items():
                if isinstance(val, torch.Tensor):
                    shape = tuple(int(s) for s in val.shape)
                    dtype_str = str(val.dtype)
                    tensor_val = val.to(device=self.device)
                    if tensor_val.dtype != self.default_dtype:
                        tensor_val = tensor_val.to(dtype=self.default_dtype)
                else:
                    arr = np.asarray(val)
                    shape = tuple(int(s) for s in arr.shape)
                    dtype_str = str(arr.dtype)
                    tensor_val = _tensor_from_numpy_safe(
                        arr,
                        device=self.device,
                        target_dtype=self.default_dtype,
                        zero_copy=self.zero_copy,
                    )
                sig_items.append((name, shape, dtype_str))
                input_feed_for_trace[name] = tensor_val
            fx_sig = tuple(sorted(sig_items, key=lambda x: x[0]))

            if getattr(self, "_fx_sig_key", None) != fx_sig:
                module = self._load_or_build_fx(
                    input_signature=fx_sig, input_feed=input_feed_for_trace
                )
                self.fx_module = module
                self._fx_sig_key = fx_sig
                if module is not None:
                    try:
                        self._fx_callable = _maybe_torch_compile(module)
                    except Exception:
                        self._fx_callable = module

        # Prefer executing the whole program via FX/Inductor when feasible
        can_use_fx = getattr(self, "_fx_callable", None) is not None and cfg.mode == "single"

        if can_use_fx:
            try:
                with torch.no_grad():
                    fx_out = self._fx_callable()  # type: ignore[operator]
                if isinstance(fx_out, tuple):
                    for name, value in zip(self.ir.exports, fx_out):
                        if isinstance(value, torch.Tensor):
                            value = value.to(device=self.device)
                            if value.dtype != self.default_dtype:
                                value = value.to(dtype=self.default_dtype)
                        self.tensors[name] = value
                elif isinstance(fx_out, torch.Tensor) and len(self.ir.exports) == 1:
                    name = next(iter(self.ir.exports))
                    self.tensors[name] = fx_out
                else:
                    can_use_fx = False
            except Exception:
                can_use_fx = False

        if not can_use_fx:
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
                timing = f" {eq['duration_ms']:.3f}ms" if eq["duration_ms"] is not None else ""
                note = f" {' '.join(details)}" if details else ""
                lines.append(
                    f"[iter {eq['iteration']:02d}] {eq['name']} {eq['status']}{timing}{note}"
                )
            elif kind == "sink":
                sk = entry["sink"]
                lines.append(f"[sink] {sk['path']} <- {sk['name']} ({sk['mode']})")
        return "\n".join(lines)

    # Internal helpers -------------------------------------------------------
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

    def _ensure_boolean_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        # Avoid dtype branching on FX proxies; emit a single to() call instead
        if "FxProxy" in globals() and FxProxy is not None and isinstance(tensor, FxProxy):  # type: ignore[name-defined]
            tensor = tensor.to(device=self.device, dtype=self.default_dtype)
        else:
            if tensor.device != self.device:
                tensor = tensor.to(self.device)
            if tensor.dtype != self.default_dtype:
                tensor = tensor.to(dtype=self.default_dtype)
        if not self._is_boolean_tensor(name):
            return tensor
        return (tensor > 0).to(dtype=self.default_dtype, device=self.device)

    # Sources / sinks --------------------------------------------------------
    def _run_sources(self):
        for eq in self._sources:
            name = eq.lhs.name
            tensor = None
            mode = "loaded"
            if name in self.tensors:
                tensor = self._ensure_boolean_tensor(name, self.tensors[name])
                mode = "provided"
            else:
                value = read_tensor_from_file(eq.src_file)
                tensor = self._materialize_weight(name, value)
                tensor = self._ensure_boolean_tensor(name, tensor)
                self.tensors[name] = tensor
            self._update_index_domains(eq.lhs, tensor)
            self.logs.append(
                {
                    "kind": "source",
                    "source": {
                        "name": name,
                        "path": eq.src_file,
                        "shape": tuple(tensor.shape),
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
                out = _torch_topk(array, k=k)
                mode = f"topk(k={k})"
            else:
                out = val
            if eq.sink_file is None:
                raise ValueError("Sink equation missing target file path")
            target_path = self.policies.resolve_output_path(eq.sink_file)
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
            array = out
            if isinstance(array, torch.Tensor):
                array = array.detach().cpu().numpy()
            write_tensor_to_file(target_path, array)

    # Execution --------------------------------------------------------------
    def _run_single_pass(self, cfg: ExecutionConfig):
        groups = self._groups if cfg.chaining == "forward" else list(reversed(self._groups))
        for name, eqs in groups:
            self._evaluate_group(
                name, eqs, iteration=0, tol=cfg.tol, capture_timing=cfg.explain_timings
            )

    def _run_fixpoint(self, cfg: ExecutionConfig):
        groups = self._groups if cfg.chaining == "forward" else list(reversed(self._groups))
        for iteration in range(cfg.max_iters):
            changed = False
            for name, eqs in groups:
                step_changed = self._evaluate_group(
                    name,
                    eqs,
                    iteration=iteration,
                    tol=cfg.tol,
                    capture_timing=cfg.explain_timings,
                )
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
        capture_timing: bool,
    ) -> bool:
        if not equations:
            return False

        contributions: List[torch.Tensor] = []
        metas: List[Dict[str, Any]] = []
        durations: List[Optional[float]] = []
        lhs_ref = equations[0].lhs
        lhs_name = lhs_ref.name
        group_temperature = self._resolve_group_temperature(lhs_name, iteration)
        prev_value = self.tensors.get(lhs_name)
        had_prev = lhs_name in self.tensors
        running_total: Optional[torch.Tensor] = None
        self._active_lhs = lhs_name
        self._active_prev_value = None
        if prev_value is not None and not lhs_ref.rolling:
            self._active_prev_value = prev_value.clone()
        self._active_equation_temperature = group_temperature

        try:
            for eq in equations:
                start = (
                    torch.cuda.Event(enable_timing=True)
                    if (capture_timing and self.device.type == "cuda")
                    else None
                )
                end = torch.cuda.Event(enable_timing=True) if start else None
                if start:
                    torch.cuda.synchronize(self.device)
                    start.record()
                self._sig_temperatures.clear()
                value, meta = self._eval_equation(eq)
                temps_used = list(self._sig_temperatures)
                effective_temp = None
                if temps_used:
                    effective_temp = temps_used[0] if len(temps_used) == 1 else temps_used
                elif group_temperature is not None:
                    effective_temp = group_temperature
                if effective_temp is not None:
                    meta = dict(meta)
                    meta["temperature"] = effective_temp
                value = self._ensure_boolean_tensor(lhs_name, value)
                if start and end:
                    end.record()
                    torch.cuda.synchronize(self.device)
                    duration_ms = start.elapsed_time(end)
                else:
                    duration_ms = None
                contributions.append(value)
                metas.append(meta)
                durations.append(duration_ms)
                if self._is_boolean_tensor(lhs_name):
                    running_total = (
                        value if running_total is None else torch.maximum(running_total, value)
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

        for idx, (eq, meta) in enumerate(zip(equations, metas)):
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

    # Evaluation primitives --------------------------------------------------
    def _eval_equation(self, eq: Equation) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rhs = eq.rhs
        lhs_name = eq.lhs.name
        if isinstance(rhs, Term):
            value, meta = self._eval_term(
                rhs,
                lhs=eq.lhs,
                capture_meta=True,
                projection=eq.projection,
            )
            if isinstance(value, torch.Tensor):
                value = self._ensure_boolean_tensor(lhs_name, value)
            return value, meta
        if isinstance(rhs, FuncCall):
            value = self._eval_fn(rhs, lhs=eq.lhs)
            meta = {"op": rhs.name}

            # Apply projection if specified
            if eq.projection and isinstance(value, torch.Tensor):
                lhs_indices = [idx for idx in eq.lhs.indices if idx not in eq.lhs.rolling]
                value_shape = value.shape

                # If value has more dimensions than LHS expects, project the extra ones
                if len(value_shape) > len(lhs_indices):
                    axes_to_project = list(range(len(lhs_indices), len(value_shape)))
                    if eq.projection == "sum":
                        for axis in reversed(sorted(axes_to_project)):
                            value = torch.sum(value, dim=axis)
                    elif eq.projection == "max":
                        for axis in reversed(sorted(axes_to_project)):
                            value = torch.max(value, dim=axis).values
                    elif eq.projection == "mean":
                        for axis in reversed(sorted(axes_to_project)):
                            value = torch.mean(value, dim=axis)
                    meta["projection"] = eq.projection
                    meta["projected_axes"] = axes_to_project

            if isinstance(value, torch.Tensor):
                value = self._ensure_boolean_tensor(lhs_name, value)
            return value, meta
        value = self._eval(rhs, lhs=eq.lhs)
        if isinstance(value, torch.Tensor):
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
                tensor = self._active_prev_value
            else:
                if expr.name not in self.tensors:
                    resolved = None
                    if self.policies.weight_store is not None:
                        try:
                            resolved = self.policies.weight_store.resolve(expr.name)
                        except KeyError:
                            resolved = None
                    if resolved is not None:
                        tensor = self._materialize_weight(expr.name, resolved)
                        tensor = self._ensure_boolean_tensor(expr.name, tensor)
                        self.tensors[expr.name] = tensor
                        self._update_index_domains(expr, tensor)
                    else:
                        raise KeyError(f"Tensor '{expr.name}' not yet defined")
                tensor = self.tensors[expr.name]
            tensor = self._ensure_boolean_tensor(expr.name, tensor)
            tensor = self._apply_index_specs(tensor, expr)
            self._update_index_domains(expr, tensor)
            return tensor
        if isinstance(expr, Term):
            value, _ = self._eval_term(expr, lhs=lhs, capture_meta=False, projection="sum")
            return value
        if isinstance(expr, FuncCall):
            return self._eval_fn(expr, lhs=lhs)
        if isinstance(expr, IndexFunction):
            axis_lengths = self._collect_axis_lengths(lhs, [], [])
            dtype = self._resolve_term_dtype([])
            return self._eval_index_function(expr, axis_lengths, dtype)
        if isinstance(expr, torch.Tensor):
            return self._as_tensor(expr)
        if isinstance(expr, (np.ndarray, list, tuple, int, float, bool)):
            return self._as_tensor(expr)
        if isinstance(expr, str):
            return expr
        if expr is None:
            return None
        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _apply_index_specs(self, tensor: torch.Tensor, ref: TensorRef) -> torch.Tensor:
        specs = getattr(ref, "index_specs", None)
        if not specs:
            if tensor.device != self.device:
                return tensor.to(self.device)
            return tensor
        result = tensor
        if result.device != self.device:
            result = result.to(self.device)
        if result.dim() == 0:
            return result
        static_specs = [spec for spec in specs if spec.axis not in ref.rolling]
        if not static_specs:
            return result
        if result.dim() < len(static_specs):
            raise ValueError(
                f"Tensor '{ref.name}' rank {result.dim()} smaller than indices {ref.indices}"
            )
        indexer = [slice(None)] * result.dim()
        has_slice = False
        for axis_idx, spec in enumerate(static_specs[: result.dim()]):
            if spec.slice is not None:
                sl = spec.slice
                indexer[axis_idx] = slice(sl.start, sl.stop, sl.step)
                has_slice = True
        if has_slice:
            result = result[tuple(indexer)]
        for axis_idx, spec in enumerate(static_specs[: result.dim()]):
            if spec.slice is not None:
                continue
            if spec.offset:
                result = _shift_axis_tensor(result, axis_idx, spec.offset)
        return result

    def _collect_axis_lengths(
        self,
        lhs: Optional[TensorRef],
        factors: Sequence[Any],
        evaluated: Sequence[Any],
    ) -> Dict[str, int]:
        lengths: Dict[str, int] = dict(self.index_domains)
        if lhs is not None and lhs.name in self.tensors:
            lhs_tensor = self.tensors[lhs.name]
            lhs_indices = [idx for idx in lhs.indices if idx not in lhs.rolling]
            for axis_idx, axis_name in enumerate(lhs_indices):
                if axis_idx < lhs_tensor.dim():
                    lengths.setdefault(axis_name, int(lhs_tensor.shape[axis_idx]))
        for factor, value in zip(factors, evaluated):
            if value is None:
                continue
            indices = _factor_indices(factor)
            if not indices:
                continue
            tensor = value if isinstance(value, torch.Tensor) else self._as_tensor(value)
            for axis_idx, axis_name in enumerate(indices):
                if axis_idx < tensor.dim():
                    lengths.setdefault(axis_name, int(tensor.shape[axis_idx]))
        return lengths

    def _resolve_term_dtype(self, evaluated: Sequence[Any]) -> torch.dtype:
        for value in evaluated:
            if value is None:
                continue
            tensor = value if isinstance(value, torch.Tensor) else self._as_tensor(value)
            return tensor.dtype
        return self.default_dtype

    def _eval_index_function(
        self,
        fn: IndexFunction,
        axis_lengths: Dict[str, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        length = axis_lengths.get(fn.axis)
        if length is None:
            raise ValueError(f"Axis '{fn.axis}' length unknown for index function '{fn.name}'")
        indices = torch.arange(length, device=self.device)
        if fn.name == "even":
            mask = (indices % 2) == 0
        elif fn.name == "odd":
            mask = (indices % 2) == 1
        else:
            raise ValueError(f"Unsupported index function '{fn.name}'")
        return mask.to(dtype=dtype)

    def _eval_term(
        self,
        term: Term,
        *,
        lhs: Optional[TensorRef],
        capture_meta: bool,
        projection: str = "sum",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if projection not in {"sum", "max", "mean"}:
            raise ValueError(f"Unsupported projection op '{projection}'")

        base_equation, projected, factor_order, _ = _normalized_einsum(term, lhs)
        evaluated: List[Optional[torch.Tensor]] = []
        pending: List[Tuple[int, IndexFunction]] = []
        for idx, factor in enumerate(term.factors):
            if isinstance(factor, IndexFunction):
                evaluated.append(None)
                pending.append((idx, factor))
            else:
                evaluated.append(self._as_tensor(self._eval(factor, lhs=lhs)))
        axis_lengths = self._collect_axis_lengths(lhs, term.factors, evaluated)
        dtype = self._resolve_term_dtype(evaluated)
        for pos, index_fn in pending:
            evaluated[pos] = self._eval_index_function(index_fn, axis_lengths, dtype)

        def _ordered_arrays(order: Sequence[int]) -> List[torch.Tensor]:
            return [self._as_tensor(evaluated[idx]) for idx in order]

        base_arrays = _ordered_arrays(factor_order)

        lhs_output_indices: List[str] = []
        if lhs is not None:
            lhs_output_indices = [idx for idx in lhs.indices if idx not in lhs.rolling]

        def _shapes_and_sizes(
            arrays: Sequence[torch.Tensor],
        ) -> Tuple[List[Tuple[int, ...]], List[int]]:
            shapes: List[Tuple[int, ...]] = []
            sizes: List[int] = []
            for tensor in arrays:
                t = tensor if isinstance(tensor, torch.Tensor) else self._as_tensor(tensor)
                shapes.append(tuple(int(dim) for dim in t.shape))
                sizes.append(int(t.element_size()))
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
            raw = torch.einsum(extended_equation, *extended_arrays)
            shapes, sizes = _shapes_and_sizes(extended_arrays)
            stats = compute_einsum_stats(
                extended_equation,
                shapes,
                sizes,
                tuple(int(dim) for dim in raw.shape),
                int(raw.element_size()),
            )
            axes_start = len(lhs_output_indices)
            reduce_axes = tuple(range(axes_start, axes_start + len(projected)))
            if not reduce_axes:
                result = raw
            else:
                axes_for_reduce = sorted(reduce_axes, reverse=True)
                result = raw
                if projection == "max":
                    for axis in axes_for_reduce:
                        result = torch.amax(result, dim=int(axis))
                else:
                    for axis in axes_for_reduce:
                        result = torch.mean(result, dim=int(axis))
            meta = {
                "einsum": extended_equation,
                "projected": projected,
                "projection": projection,
            }
            meta.update(stats)
        else:
            result = torch.einsum(base_equation, *base_arrays)
            meta = {"einsum": base_equation, "projected": projected}
            if projection != "sum":
                meta["projection"] = projection
            shapes, sizes = _shapes_and_sizes(base_arrays)
            stats = compute_einsum_stats(
                base_equation,
                shapes,
                sizes,
                tuple(int(dim) for dim in result.shape),
                int(result.element_size()),
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

        if name == "step":
            return _torch_step(eval_arg(args_expr[0]))
        if name == "relu":
            return torch.relu(eval_arg(args_expr[0]))
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
            return _torch_sig(base, temperature)
        if name == "gelu":
            return _torch_gelu(eval_arg(args_expr[0]))
        if name == "gelu_grad":
            return _torch_gelu_grad(eval_arg(args_expr[0]))
        if name == "lnorm":
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs)
            )
            eps = float(fn.kwargs.get("eps", 1e-5))
            return _torch_lnorm(self._as_tensor(eval_arg(args_expr[0])), axis=axis, eps=eps)
        if name in {"layernorm", "layer_norm"}:
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs)
            )
            eps = float(fn.kwargs.get("eps", 1e-5))
            return _torch_layer_norm(self._as_tensor(eval_arg(args_expr[0])), axis=axis, eps=eps)
        if name == "softmax":
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs) or -1
            )
            mask_expr: Optional[Any] = None
            if len(args_expr) > 1:
                mask_expr = args_expr[1]
            elif "mask" in fn.kwargs:
                mask_expr = fn.kwargs.get("mask")
            if mask_expr is not None:
                mask_value = self._eval(mask_expr, lhs=lhs)
                fill_expr = fn.kwargs.get("fill")
                fill_value = self._eval(fill_expr, lhs=lhs) if fill_expr is not None else None
                logits = self._as_tensor(eval_arg(args_expr[0]))
                logits = logits.to(dtype=self.default_dtype)
                mask_tensor = self._as_tensor(mask_value) if mask_value is not None else None
                return _torch_masked_softmax(logits, mask_tensor, dim=axis, fill_value=fill_value)
            logits = self._as_tensor(eval_arg(args_expr[0]))
            logits = logits.to(dtype=self.default_dtype)
            return F.softmax(logits, dim=axis)
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
            mask_value = self._eval(mask_expr, lhs=lhs) if mask_expr is not None else None
            fill_expr = fn.kwargs.get("fill")
            fill_value = self._eval(fill_expr, lhs=lhs) if fill_expr is not None else None
            return _torch_masked_softmax(
                self._as_tensor(eval_arg(args_expr[0])),
                self._as_tensor(mask_value) if mask_value is not None else None,
                dim=axis,
                fill_value=fill_value,
            )
        if name == "softmax_grad":
            if len(args_expr) < 2:
                raise ValueError("softmax_grad requires probabilities and gradient arguments")
            probs = eval_arg(args_expr[0])
            grad = eval_arg(args_expr[1])
            axis = self._axis_from_spec(
                fn.kwargs.get("axis"), args_expr[0], lhs, default=self._dotted_axis(lhs) or -1
            )
            return _torch_softmax_grad(probs, grad, axis)
        if name == "sin":
            if not args_expr:
                raise ValueError("sin requires an argument")
            return torch.sin(eval_arg(args_expr[0]))
        if name == "cos":
            if not args_expr:
                raise ValueError("cos requires an argument")
            return torch.cos(eval_arg(args_expr[0]))
        if name == "rope":
            return _torch_rope(eval_arg(args_expr[0]))
        if name == "concat":
            if not args_expr:
                raise ValueError("concat requires at least one argument")
            arrays = [eval_arg(arg) for arg in args_expr]
            axis_spec = fn.kwargs.get("axis")
            if len(arrays) == 1:
                axis = None
            else:
                axis = self._axis_from_spec(axis_spec, args_expr[0], lhs, default=-1)
            return _torch_concat(arrays, axis=axis)
        if name in {"max", "amax"}:
            arr = eval_arg(args_expr[0])
            axis = self._axis_from_spec(fn.kwargs.get("axis"), args_expr[0], lhs, default=-1)
            keepdims = bool(fn.kwargs.get("keepdims", False))
            return _torch_reduce_max(arr, axis=axis, keepdims=keepdims)
        if name in {"avg", "mean"}:
            arr = eval_arg(args_expr[0])
            axis = self._axis_from_spec(fn.kwargs.get("axis"), args_expr[0], lhs, default=-1)
            keepdims = bool(fn.kwargs.get("keepdims", False))
            return _torch_reduce_mean(arr, axis=axis, keepdims=keepdims)
        if name == "causal_mask":
            axis_expr = args_expr[0] if args_expr else fn.arg
            return _torch_causal_mask(
                int(axis_expr),
                device=self.device,
                target_dtype=self.default_dtype,
            )
        if name == "const":
            value = args_expr[0] if args_expr else fn.arg
            return _torch_const(
                value,
                device=self.device,
                target_dtype=self.default_dtype,
                zero_copy=self.zero_copy,
            )
        if name == "attention":
            if len(args_expr) < 3:
                raise ValueError("attention requires query, key, and value arguments")
            q_in = self._as_tensor(eval_arg(args_expr[0]))
            k_in = self._as_tensor(eval_arg(args_expr[1]))
            v_in = self._as_tensor(eval_arg(args_expr[2]))
            # Inline shape promotion to keep FX graphable
            q_dim = q_in.dim()
            if q_dim == 4:
                q = q_in
            elif q_dim == 3:
                q = q_in.unsqueeze(1)
            elif q_dim == 2:
                q = q_in.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError("attention expects tensors with rank >= 2")
            k_dim = k_in.dim()
            if k_dim == 4:
                k = k_in
            elif k_dim == 3:
                k = k_in.unsqueeze(1)
            elif k_dim == 2:
                k = k_in.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError("attention expects tensors with rank >= 2")
            v_dim = v_in.dim()
            if v_dim == 4:
                v = v_in
            elif v_dim == 3:
                v = v_in.unsqueeze(1)
            elif v_dim == 2:
                v = v_in.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError("attention expects tensors with rank >= 2")

            # Mask handling
            mask_expr: Optional[Any] = None
            if len(args_expr) > 3:
                mask_expr = args_expr[3]
            elif "mask" in fn.kwargs:
                mask_expr = fn.kwargs.get("mask")
            attn_mask = None
            if mask_expr is not None:
                mask_value = self._eval(mask_expr, lhs=lhs)
                m = self._as_tensor(mask_value)
                if m.dtype != torch.bool:
                    m = m != 0
                while m.dim() < 4:
                    m = m.unsqueeze(0)
                attn_mask = m.to(device=q.device)

            # Scale handling
            scale_expr = fn.kwargs.get("scale")
            raw_scale = self._eval(scale_expr, lhs=lhs) if scale_expr is not None else None
            if isinstance(raw_scale, str):
                raw_scale = self.tensors.get(raw_scale)
            scale_value: Optional[float]
            if raw_scale is None:
                scale_value = None
            elif isinstance(raw_scale, torch.Tensor):
                if raw_scale.numel() != 1:
                    raise ValueError("Attention scale must be a scalar tensor")
                scale_value = float(raw_scale.detach().cpu().reshape(()).item())
            elif isinstance(raw_scale, np.ndarray):
                if raw_scale.size != 1:
                    raise ValueError("Attention scale must be a scalar array")
                scale_value = float(raw_scale.reshape(()))
            else:
                scale_value = float(raw_scale)
            causal = bool(fn.kwargs.get("causal", False))

            # Apply scale either via argument or by pre-scaling query
            scaled_query = q
            if scale_value is not None and not _HAS_SDP_SCALE:
                head_dim = q.shape[-1]
                scaled_query = q * (scale_value * math.sqrt(head_dim))
                scale_arg = None
            else:
                scale_arg = scale_value if _HAS_SDP_SCALE else None

            # Prefer CUDA SDP kernels when available
            use_cuda_sdpa_ctx = (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "cuda")
                and hasattr(torch.backends.cuda, "sdp_kernel")
                and q.is_cuda
                and q.dtype in {torch.float16, torch.bfloat16, torch.float32}
            )
            try:
                if use_cuda_sdpa_ctx:
                    with torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
                        enable_flash=True, enable_mem_efficient=True, enable_math=False
                    ):
                        try:
                            out = F.scaled_dot_product_attention(
                                scaled_query,
                                k,
                                v,
                                attn_mask=attn_mask,
                                dropout_p=0.0,
                                is_causal=causal,
                                scale=scale_arg if _HAS_SDP_SCALE else None,
                            )
                        except TypeError:
                            out = F.scaled_dot_product_attention(
                                scaled_query,
                                k,
                                v,
                                attn_mask=attn_mask,
                                dropout_p=0.0,
                                is_causal=causal,
                            )
                else:
                    try:
                        out = F.scaled_dot_product_attention(
                            scaled_query,
                            k,
                            v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=scale_arg if _HAS_SDP_SCALE else None,
                        )
                    except TypeError:
                        out = F.scaled_dot_product_attention(
                            scaled_query,
                            k,
                            v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                        )
            except Exception:
                math_ctx = (
                    torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
                        enable_flash=False, enable_mem_efficient=False, enable_math=True
                    )
                    if hasattr(torch, "backends")
                    and hasattr(torch.backends, "cuda")
                    and hasattr(torch.backends.cuda, "sdp_kernel")
                    and q.is_cuda
                    else _nullcontext()
                )
                with math_ctx:
                    try:
                        out = F.scaled_dot_product_attention(
                            scaled_query,
                            k,
                            v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=scale_arg if _HAS_SDP_SCALE else None,
                        )
                    except TypeError:
                        out = F.scaled_dot_product_attention(
                            scaled_query,
                            k,
                            v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                        )

            # Restore original rank
            if q_dim == 4:
                return out
            if q_dim == 3:
                return out.squeeze(1)
            # q_dim == 2
            return out.squeeze(0).squeeze(0)
        if name == "tucker_dense":
            if not args_expr:
                raise ValueError("tucker_dense requires a tensor argument")
            base = self._as_tensor(eval_arg(args_expr[0]))
            rank_value: Any = None
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
            return _torch_tucker_dense(
                base,
                rank=rank_spec,
                threshold=threshold,
                target_dtype=self.default_dtype,
            )
        if name == "topk":
            arr = eval_arg(args_expr[0])
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
            result: Optional[torch.Tensor] = None
            if default_expr is not None:
                result = self._as_tensor(self._eval(default_expr, lhs=lhs))
            for cond_expr, value_expr in pairs:
                value = self._as_tensor(self._eval(value_expr, lhs=lhs))
                cond = self._as_tensor(self._eval(cond_expr, lhs=lhs)).to(dtype=torch.bool)
                # Ensure cond can broadcast with value by expanding dimensions if needed
                # The condition should have shape that is broadcastable with value
                # For example, if value is [p,d] and cond is [d], expand to [1,d]
                # If value is [i,j] and cond is [i], expand to [i,1]
                if cond.ndim < value.ndim:
                    # Determine which dimensions to expand based on the shape matching
                    # Try to match trailing dimensions
                    for _ in range(value.ndim - cond.ndim):
                        cond = cond.unsqueeze(0)
                if result is None:
                    result = torch.zeros_like(value)
                result = torch.where(cond, value, result)
            if result is None:
                raise ValueError("case evaluation produced no result")
            return result
        raise ValueError(f"Unknown torch builtin: {fn.name}")

    # Assign / compare -------------------------------------------------------
    def _assign(self, ref: TensorRef, value: torch.Tensor, tol: float) -> bool:
        value = self._ensure_boolean_tensor(ref.name, value)
        prev = self.tensors.get(ref.name)
        if prev is None:
            self.tensors[ref.name] = value
            self._update_index_domains(ref, value)
            return True
        if not self._values_close(prev, value, tol):
            self.tensors[ref.name] = value
            self._update_index_domains(ref, value)
            return True
        self._update_index_domains(ref, prev)
        return False

    def _values_close(self, a: torch.Tensor, b: torch.Tensor, tol: float) -> bool:
        if a.shape != b.shape:
            return False
        if a.dtype.is_floating_point or b.dtype.is_floating_point:
            return torch.allclose(a, b, atol=tol, rtol=1e-5)
        return torch.equal(a, b)

    # Logging ----------------------------------------------------------------
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
        self.logs.append(
            {
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

    # Utility helpers --------------------------------------------------------
    def _update_index_domains(self, ref: TensorRef, tensor: torch.Tensor):
        for i, idx in enumerate(ref.indices):
            if i < tensor.ndim:
                self.index_domains[idx] = max(self.index_domains.get(idx, 0), tensor.shape[i])

    def _as_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
            if tensor.device != self.device:
                tensor = tensor.to(device=self.device)
            if tensor.dtype != self.default_dtype:
                tensor = tensor.to(dtype=self.default_dtype)
            return tensor
        # Allow FX tracing proxies to flow through without numpy conversion
        if "FxProxy" in globals() and FxProxy is not None and isinstance(value, FxProxy):  # type: ignore[name-defined]
            try:
                return value.to(device=self.device, dtype=self.default_dtype)  # type: ignore[return-value]
            except Exception:
                return value  # type: ignore[return-value]
        array = value if isinstance(value, np.ndarray) else np.asarray(value)
        return _tensor_from_numpy_safe(
            array,
            device=self.device,
            target_dtype=self.default_dtype,
            zero_copy=self.zero_copy,
        )

    def _compute_fx_input_names(self) -> List[str]:
        # Gather all defined names (LHS) and all referenced names in RHS
        defined: Set[str] = set(eq.lhs.name for eq in self.ir.equations)
        sources: Set[str] = set(eq.lhs.name for eq in self.ir.equations if eq.is_source)
        used: Set[str] = set()

        def _collect_names(obj: Any) -> None:
            if isinstance(obj, TensorRef):
                used.add(obj.name)
                return
            if isinstance(obj, Term):
                for f in obj.factors:
                    _collect_names(f)
                return
            if isinstance(obj, FuncCall):
                arg = obj.arg
                if isinstance(arg, tuple):
                    for item in arg:
                        _collect_names(item)
                elif arg is not None:
                    _collect_names(arg)
                for val in obj.kwargs.values():
                    _collect_names(val)
                return
            # Ignore IndexFunction and literals

        for eq in self.ir.equations:
            _collect_names(eq.rhs)

        # Dynamic inputs: sources + names used in RHS that are not defined by any LHS
        return sorted(list(sources | (used - defined)))

    def _materialize_weight(self, name: str, payload: Any) -> torch.Tensor:
        import torch

        tensor = self.policies.materialize_weight(
            name,
            payload,
            backend="torch",
            device=self.device,
        )
        if not isinstance(tensor, torch.Tensor):
            tensor = self._as_tensor(tensor)
        else:
            tensor = tensor.to(device=self.device)
            if tensor.dtype != self.default_dtype:
                tensor = tensor.to(dtype=self.default_dtype)
        return tensor

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
        # Handle TensorRef representing an axis name (e.g., axis=j parsed as TensorRef)
        if isinstance(spec, TensorRef) and spec.name and not spec.indices:
            axis_name = spec.name
            if lhs and axis_name in lhs.indices:
                return lhs.indices.index(axis_name)
            tref = _first_tensor_ref(arg_expr)
            if tref and axis_name in tref.indices:
                return tref.indices.index(axis_name)
        raise ValueError(f"Cannot resolve axis specification: {spec}")

    # FX Graph ----------------------------------------------------------------
    def _load_or_build_fx(
        self,
        *,
        input_signature: Optional[Tuple[Any, ...]] = None,
        input_feed: Optional[Dict[str, Any]] = None,
    ) -> Optional[GraphModule]:
        if self.cache_manager is None or GraphModule is None:
            return self._build_fx(input_feed=input_feed)

        fingerprint = cache_fingerprint(
            program_src=self.program.src,
            backend="torch",
            artifact="fx_graph",
            device=str(self.device),
            execution_config=self.config,
            policies=self.policies,
            extra={"exports": list(self.ir.exports), "input_signature": input_signature},
        )
        cache_key = cache_key_from_fingerprint(fingerprint)
        record = self.cache_manager.load("torch", cache_key)
        if record and record.payload is not None:
            buffer = io.BytesIO(record.payload)
            try:
                module = torch.load(buffer, map_location=self.device)
                return module
            except Exception:
                pass

        module = self._build_fx(input_feed=input_feed)
        metadata = {
            "device": str(self.device),
            "exports": list(self.ir.exports),
            "cache_fingerprint": fingerprint,
            "input_signature": input_signature,
        }
        if module is not None:
            buffer = io.BytesIO()
            torch.save(module, buffer)
            self.cache_manager.store(
                "torch",
                cache_key,
                buffer.getvalue(),
                metadata=metadata,
            )
        else:
            self.cache_manager.write_metadata("torch", cache_key, metadata)
        return module

    def _build_fx(self, *, input_feed: Optional[Dict[str, Any]] = None) -> Optional[GraphModule]:
        if torch is None or symbolic_trace is None:
            return None

        trace_runner = _TorchTraceHarness(self, input_feed=input_feed)
        try:
            with torch.no_grad():
                gm = symbolic_trace(trace_runner)
            gm.to(self.device)
            _annotate_fx_graph(gm)
            return gm
        except Exception:
            return None


if torch is not None:

    class _TorchTraceHarness(torch.nn.Module):
        def __init__(self, runner: TorchRunner, input_feed: Optional[Dict[str, Any]]):
            super().__init__()
            self.runner = runner
            self.input_feed: Dict[str, Any] = dict(input_feed) if input_feed else {}

        @torch.no_grad()
        def forward(self) -> Tuple[torch.Tensor, ...]:
            prev_cfg = self.runner.config
            cfg = replace(prev_cfg, mode="single")
            prev_zero_copy = self.runner.zero_copy
            prev_temps = self.runner._temperature_schedules
            try:
                outputs = self.runner.run(inputs=self.input_feed, config=cfg, skip_sinks=True)
            finally:
                self.runner.config = prev_cfg
                self.runner.zero_copy = prev_zero_copy
                self.runner.default_dtype = _resolve_precision_dtype(
                    prev_cfg.precision, self.runner.device
                )
                self.runner._temperature_schedules = prev_temps
            return tuple(outputs[name] for name in self.runner.ir.exports)

else:

    class _TorchTraceHarness:  # pragma: no cover - torch optional
        def __init__(self, runner: TorchRunner):
            raise RuntimeError("Torch backend is unavailable")

        def forward(self) -> Tuple[Any, ...]:
            raise RuntimeError("Torch backend is unavailable")


def _torch_topk(tensor: torch.Tensor, k: int):
    if not torch.is_floating_point(tensor):
        raise ValueError("topk requires floating point tensor input")
    values, indices = torch.topk(tensor, k=k, dim=-1)
    values = values.detach().cpu().numpy()
    indices = indices.detach().cpu().numpy()
    if values.ndim == 1:
        return [(int(indices[i]), float(values[i])) for i in range(values.shape[0])]
    results = []
    for row_idx in range(values.shape[0]):
        row = []
        for col in range(values.shape[1]):
            row.append((int(indices[row_idx, col]), float(values[row_idx, col])))
        results.append(row)
    return results
