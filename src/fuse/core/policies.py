from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np

from .builtins import SparseBoolTensor
from .exceptions import BackendError

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int], float, int]

_DEFAULT_NPZ_MMAP_THRESHOLD = 64 * 1024 * 1024  # 64 MiB

if TYPE_CHECKING:
    import torch


class WeightStore(Protocol):
    def resolve(self, name: str) -> Any:
        """Return a payload representing the requested weight."""


def _hash_bytes(data: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def _array_fingerprint(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return {"repr": repr(value)}
    summary: Dict[str, Any] = {
        "shape": tuple(int(dim) for dim in arr.shape),
        "dtype": str(arr.dtype),
    }
    try:
        contig = np.ascontiguousarray(arr)
        view = contig.view(np.uint8)
        summary["sha256"] = _hash_bytes(bytes(view))
    except Exception:
        summary["sha256"] = _hash_bytes(repr(arr).encode("utf-8"))
    return summary


def _path_fingerprint(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "size": int(stat.st_size),
            "mtime": int(stat.st_mtime),
        }
    except OSError:
        return {"path": str(path), "missing": True}


def _sanitize_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _fingerprint_path(path: Optional[Union[str, Path]]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(path)


@dataclass
class InMemoryWeightStore:
    weights: Dict[str, Any]

    def resolve(self, name: str) -> Any:
        if name not in self.weights:
            raise KeyError(f"Weight '{name}' not found in store")
        return self.weights[name]

    def fingerprint(self) -> Dict[str, Any]:
        entries = []
        for name in sorted(self.weights):
            entries.append(
                {
                    "name": str(name),
                    "value": _array_fingerprint(self.weights[name]),
                }
            )
        return {"type": "in_memory", "entries": entries}


@dataclass
class ShardingPolicy:
    strategy: str = "replicated"  # e.g., replicated, row, column
    mesh: Optional[Any] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def materialize(self, name: str, value: Any) -> Any:
        """Apply sharding metadata; numpy backend treats everything as local."""
        return value

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "mesh": _sanitize_metadata(self.mesh),
            "attributes": _sanitize_metadata(self.attributes),
        }


@dataclass
class QuantizationPolicy:
    mode: str = "none"  # e.g., none, int8
    bits: int = 8
    group_size: int = 64

    def apply(self, name: str, value: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return value
        if self.mode == "int8":
            return value.astype(np.float32)
        return value

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "bits": int(self.bits),
            "group_size": int(self.group_size),
        }


@dataclass
class LoRAPolicy:
    rank: int
    alpha: float = 1.0
    adapters: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def apply(self, name: str, base: np.ndarray) -> np.ndarray:
        if name not in self.adapters:
            return base
        a, b = self.adapters[name]
        update = (self.alpha / float(self.rank)) * (a @ b)
        return base + update

    def fingerprint(self) -> Dict[str, Any]:
        adapters_fp: Dict[str, Dict[str, Any]] = {}
        for name, (up, down) in sorted(self.adapters.items()):
            adapters_fp[str(name)] = {
                "up": _array_fingerprint(up),
                "down": _array_fingerprint(down),
            }
        return {
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "adapters": adapters_fp,
        }


def _ensure_path(value: Union[str, Path], base_path: Optional[Path]) -> Path:
    path = Path(value)
    if not path.is_absolute() and base_path is not None:
        path = (base_path / path).resolve()
    return path


def _load_manifest_array(
    spec: Any,
    *,
    base_path: Optional[Path],
    dtype: Optional[Union[str, np.dtype]] = None,
    mmap: bool = False,
    mmap_threshold: Optional[int] = None,
    strict: bool = False,
) -> Optional[np.ndarray]:
    if spec is None:
        return None
    resolved_dtype = np.dtype(dtype) if dtype is not None else None
    if isinstance(spec, (str, Path)):
        path = _ensure_path(spec, base_path)
        return _load_array_from_path(
            path,
            dtype=resolved_dtype,
            mmap=mmap,
            mmap_threshold=mmap_threshold,
            strict=strict,
        )
    if isinstance(spec, Mapping):
        if "path" in spec:
            path = _ensure_path(spec["path"], base_path)
            local_dtype = spec.get("dtype", resolved_dtype)
            return _load_array_from_path(
                path,
                dtype=np.dtype(local_dtype) if local_dtype else None,
                mmap=mmap,
                mmap_threshold=mmap_threshold,
                strict=strict,
            )
        if "value" in spec:
            arr = np.asarray(spec["value"])
            if resolved_dtype is not None:
                arr = arr.astype(resolved_dtype, copy=False)
            return arr
    arr = np.asarray(spec)
    if resolved_dtype is not None:
        arr = arr.astype(resolved_dtype, copy=False)
    return arr


def _load_array_from_path(
    path: Path,
    *,
    dtype: Optional[np.dtype],
    mmap: bool,
    mmap_threshold: Optional[int] = None,
    strict: bool = False,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Weight manifest reference '{path}' not found")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        if strict and not mmap:
            raise RuntimeError(f"Strict loading requires mmap support for '{path.name}'")
        use_mmap = False
        if mmap:
            if strict:
                use_mmap = True
            elif mmap_threshold is None:
                use_mmap = True
            else:
                try:
                    size = path.stat().st_size
                    use_mmap = size >= int(mmap_threshold)
                except OSError:
                    use_mmap = True
        mmap_mode = "r" if use_mmap else None
        with np.load(path, allow_pickle=False, mmap_mode=mmap_mode) as data:
            if not data.files:
                raise ValueError(f"Weight manifest reference '{path}' contained no arrays")
            arr = data[data.files[0]]
            if strict and not isinstance(arr, np.memmap):
                raise RuntimeError(f"Strict loading forbids materializing archive '{path.name}'")
    else:
        if suffix == ".npy" and strict and not mmap:
            raise RuntimeError(f"Strict loading requires mmap support for '{path.name}'")
        mmap_mode = "r" if mmap and suffix == ".npy" else None
        arr = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
        if strict and not isinstance(arr, np.memmap):
            raise RuntimeError(f"Strict loading forbids materializing array '{path.name}'")
    result = arr
    if dtype is not None:
        converted = result.astype(dtype, copy=False)
        if strict and converted is not result and isinstance(result, np.memmap):
            raise RuntimeError(
                f"Strict loading forbids dtype conversion of '{path.name}' that materializes data"
            )
        result = converted
    if strict and not isinstance(result, np.memmap):
        raise RuntimeError(f"Strict loading forbids materializing array '{path.name}'")
    return np.asarray(result) if not isinstance(result, np.ndarray) else result


@dataclass
class QuantizedSpec:
    mode: str
    scale: np.ndarray
    zero_point: Optional[np.ndarray] = None
    dtype: str = "int8"
    group_size: Optional[int] = None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_path: Optional[Path],
        mmap: bool,
        mmap_threshold: Optional[int],
        strict: bool,
    ) -> "QuantizedSpec":
        mode = mapping.get("mode", "none")
        dtype = mapping.get("dtype", "int8")
        group_size = mapping.get("group_size")
        scale = _load_manifest_array(
            mapping.get("scale"),
            base_path=base_path,
            dtype="float32",
            mmap=mmap,
            mmap_threshold=mmap_threshold,
            strict=strict,
        )
        if scale is None:
            raise ValueError("Quantized weight requires a 'scale'")
        zero_point = _load_manifest_array(
            mapping.get("zero_point"),
            base_path=base_path,
            dtype="float32",
            mmap=mmap,
            mmap_threshold=mmap_threshold,
            strict=strict,
        )
        return cls(
            mode=mode, scale=scale, zero_point=zero_point, dtype=dtype, group_size=group_size
        )

    def is_active(self) -> bool:
        return self.mode != "none"

    def __post_init__(self) -> None:
        self.scale = np.asarray(self.scale, dtype=np.float32)
        if self.scale.ndim == 0:
            self.scale = self.scale.reshape(1)
        if self.zero_point is not None:
            self.zero_point = np.asarray(self.zero_point, dtype=np.float32)
            if self.zero_point.ndim == 0:
                self.zero_point = self.zero_point.reshape(1)
            try:
                np.broadcast_shapes(self.scale.shape, self.zero_point.shape)
            except ValueError as exc:
                raise ValueError(
                    f"Quantized zero_point shape {self.zero_point.shape} incompatible with scale shape {self.scale.shape}"
                ) from exc
        if self.dtype not in {"int8", "uint8"}:
            raise ValueError(f"Unsupported quantized dtype '{self.dtype}' (expected int8/uint8)")

    def _broadcast_params_numpy(
        self, values: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        try:
            scale = np.broadcast_to(self.scale, values.shape).astype(np.float32, copy=False)
        except ValueError as exc:
            raise ValueError(
                f"Cannot broadcast quantization scale of shape {self.scale.shape} to values with shape {values.shape}"
            ) from exc
        zero = None
        if self.zero_point is not None:
            try:
                zero = np.broadcast_to(self.zero_point, values.shape).astype(np.float32, copy=False)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot broadcast quantization zero_point of shape {self.zero_point.shape} to values with shape {values.shape}"
                ) from exc
        return scale, zero

    def dequantize_numpy(self, values: np.ndarray) -> np.ndarray:
        if not self.is_active():
            return values.astype(np.float32, copy=False)
        if values.dtype.kind not in {"i", "u"}:
            raise ValueError(f"Quantized values must be integer types, received {values.dtype}")
        arr = values.astype(np.float32, copy=False)
        scale, zero = self._broadcast_params_numpy(arr)
        if zero is not None:
            arr = (arr - zero) * scale
        else:
            arr = arr * scale
        return arr.astype(np.float32, copy=False)

    def dequantize_torch(self, tensor: "torch.Tensor") -> "torch.Tensor":
        import torch

        if not self.is_active():
            return tensor.to(dtype=torch.float32)
        if tensor.dtype not in {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}:
            raise ValueError(f"Quantized tensors must be integer types, received {tensor.dtype}")
        result = tensor.to(dtype=torch.float32)
        scale = torch.as_tensor(self.scale, device=result.device, dtype=result.dtype)
        if scale.ndim == 0:
            scale = scale.reshape(1)
        if self.zero_point is not None:
            zero = torch.as_tensor(self.zero_point, device=result.device, dtype=result.dtype)
            if zero.ndim == 0:
                zero = zero.reshape(1)
            result = (result - zero) * scale
        else:
            result = result * scale
        return result


@dataclass
class LoRAAdapter:
    name: str
    up: np.ndarray
    down: np.ndarray
    alpha: float = 1.0

    @property
    def rank(self) -> int:
        if self.up.ndim == 2:
            return self.up.shape[1]
        if self.up.ndim == 1:
            return self.up.shape[0]
        raise ValueError("LoRA adapter expects matrices")

    def __post_init__(self) -> None:
        self.up = np.asarray(self.up, dtype=np.float32)
        self.down = np.asarray(self.down, dtype=np.float32)
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")

    def _validate_shapes(self, base_shape: Tuple[int, ...]) -> None:
        if self.up.ndim != 2 or self.down.ndim != 2:
            raise ValueError("LoRA adapters require matrix-shaped 'up' and 'down' tensors")
        if self.up.shape[1] != self.down.shape[0]:
            raise ValueError(
                f"LoRA adapter '{self.name}' has mismatched inner dims: up {self.up.shape}, down {self.down.shape}"
            )
        expected_shape = (self.up.shape[0], self.down.shape[1])
        if tuple(base_shape) != expected_shape:
            raise ValueError(
                f"LoRA merge expects base shape {expected_shape}, received {tuple(base_shape)} for adapter '{self.name}'"
            )

    def merge_numpy(self, base: np.ndarray) -> np.ndarray:
        self._validate_shapes(base.shape)
        rank = max(1, self.rank)
        update = (self.alpha / float(rank)) * (self.up @ self.down)
        return base + update

    def merge_torch(self, base: "torch.Tensor") -> "torch.Tensor":
        import torch

        self._validate_shapes(tuple(base.shape))
        rank = max(1, self.rank)
        a = torch.as_tensor(self.up, device=base.device, dtype=base.dtype)
        b = torch.as_tensor(self.down, device=base.device, dtype=base.dtype)
        update = (self.alpha / float(rank)) * (a @ b)
        return base + update


@dataclass
class LoRASpec:
    slot: Optional[str]
    merge: str
    adapters: Dict[str, LoRAAdapter]
    default: Optional[str]

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_path: Optional[Path],
        mmap: bool,
        mmap_threshold: Optional[int],
        strict: bool,
    ) -> "LoRASpec":
        slot = mapping.get("slot")
        merge = mapping.get("merge", "on_the_fly")
        adapters: Dict[str, LoRAAdapter] = {}
        for name, adapter_conf in mapping.get("adapters", {}).items():
            up = _load_manifest_array(
                adapter_conf.get("up") or adapter_conf.get("up_path"),
                base_path=base_path,
                dtype="float32",
                mmap=mmap,
                mmap_threshold=mmap_threshold,
                strict=strict,
            )
            if up is None:
                raise ValueError(f"LoRA adapter '{name}' requires 'up'")
            down = _load_manifest_array(
                adapter_conf.get("down") or adapter_conf.get("down_path"),
                base_path=base_path,
                dtype="float32",
                mmap=mmap,
                mmap_threshold=mmap_threshold,
                strict=strict,
            )
            if down is None:
                raise ValueError(f"LoRA adapter '{name}' requires 'down'")
            alpha = float(adapter_conf.get("alpha", mapping.get("alpha", 1.0)))
            adapters[name] = LoRAAdapter(
                name=name, up=up.astype(np.float32), down=down.astype(np.float32), alpha=alpha
            )
        default = mapping.get("default")
        if default is None and adapters:
            default = next(iter(adapters))
        return cls(slot=slot, merge=merge, adapters=adapters, default=default)


@dataclass
class ResolvedLoRA:
    merge: str
    adapter: Optional[LoRAAdapter] = None

    def merge_numpy(self, base: np.ndarray) -> np.ndarray:
        if self.merge != "on_the_fly" or self.adapter is None:
            return base
        return self.adapter.merge_numpy(base)

    def merge_torch(self, base: "torch.Tensor") -> "torch.Tensor":
        if self.merge != "on_the_fly" or self.adapter is None:
            return base
        return self.adapter.merge_torch(base)


@dataclass
class PrefetchSpec:
    window: int = 0
    group: Optional[str] = None
    neighbors: Tuple[str, ...] = ()
    order: int = 0

    @classmethod
    def from_mapping(
        cls, mapping: Optional[Mapping[str, Any]], default_order: int
    ) -> "PrefetchSpec":
        if mapping is None:
            return cls(order=default_order)
        window = int(mapping.get("window", 0))
        group = mapping.get("group")
        neighbors = tuple(mapping.get("neighbors", ()))
        order = int(mapping.get("order", default_order))
        return cls(window=window, group=group, neighbors=neighbors, order=order)


@dataclass
class WeightShard:
    path: Path
    axis: Optional[int] = None
    dtype: Optional[np.dtype] = None

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], *, base_path: Optional[Path]
    ) -> "WeightShard":
        path = mapping.get("path")
        if path is None:
            raise ValueError("Weight shard requires 'path'")
        axis = mapping.get("axis")
        dtype = mapping.get("dtype")
        return cls(
            path=_ensure_path(path, base_path),
            axis=axis,
            dtype=np.dtype(dtype) if dtype is not None else None,
        )


@dataclass
class WeightManifestEntry:
    name: str
    path: Optional[Path]
    value: Optional[np.ndarray]
    dtype: np.dtype
    shards: Tuple[WeightShard, ...]
    shard_axis: Optional[int]
    quant: Optional[QuantizedSpec]
    lora: Optional[LoRASpec]
    scale: Optional[np.ndarray]
    prefetch: PrefetchSpec
    placement: Optional[str]
    metadata: Dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        base_path: Optional[Path],
        mmap: bool,
        mmap_threshold: Optional[int],
        strict: bool,
        default_order: int,
    ) -> "WeightManifestEntry":
        name = mapping.get("name")
        if not name:
            raise ValueError("Manifest entry missing 'name'")
        dtype = np.dtype(mapping.get("dtype", "float32"))
        path = mapping.get("path")
        resolved_path = _ensure_path(path, base_path) if path is not None else None
        value = None
        if "value" in mapping:
            value = np.asarray(mapping["value"]).astype(dtype, copy=False)
        shards_conf = mapping.get("shards") or []
        shards = tuple(WeightShard.from_mapping(item, base_path=base_path) for item in shards_conf)
        shard_axis = mapping.get("shard_axis")
        quant_spec = None
        if "quant" in mapping:
            quant_spec = QuantizedSpec.from_mapping(
                mapping["quant"],
                base_path=base_path,
                mmap=mmap,
                mmap_threshold=mmap_threshold,
                strict=strict,
            )
        lora_spec = None
        if "lora" in mapping:
            lora_spec = LoRASpec.from_mapping(
                mapping["lora"],
                base_path=base_path,
                mmap=mmap,
                mmap_threshold=mmap_threshold,
                strict=strict,
            )
        scale_spec = _load_manifest_array(
            mapping.get("scale"),
            base_path=base_path,
            dtype="float32",
            mmap=mmap,
            mmap_threshold=mmap_threshold,
            strict=strict,
        )
        prefetch_spec = PrefetchSpec.from_mapping(
            mapping.get("prefetch"), default_order=default_order
        )
        placement = mapping.get("placement")
        metadata = dict(mapping.get("metadata", {}))
        return cls(
            name=name,
            path=resolved_path,
            value=value,
            dtype=dtype,
            shards=shards,
            shard_axis=shard_axis,
            quant=quant_spec,
            lora=lora_spec,
            scale=scale_spec,
            prefetch=prefetch_spec,
            placement=placement,
            metadata=metadata,
        )


def _weight_shard_fingerprint(shard: WeightShard) -> Dict[str, Any]:
    return {
        "path": _path_fingerprint(shard.path),
        "axis": shard.axis,
        "dtype": str(shard.dtype) if shard.dtype is not None else None,
    }


def _quant_spec_fingerprint(spec: Optional[QuantizedSpec]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    return {
        "mode": spec.mode,
        "dtype": spec.dtype,
        "group_size": spec.group_size,
        "scale": _array_fingerprint(spec.scale),
        "zero_point": _array_fingerprint(spec.zero_point),
    }


def _lora_adapter_fingerprint(adapter: LoRAAdapter) -> Dict[str, Any]:
    return {
        "rank": adapter.rank,
        "alpha": float(adapter.alpha),
        "up": _array_fingerprint(adapter.up),
        "down": _array_fingerprint(adapter.down),
    }


def _lora_spec_fingerprint(spec: Optional[LoRASpec]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    return {
        "slot": spec.slot,
        "merge": spec.merge,
        "default": spec.default,
        "adapters": {
            str(name): _lora_adapter_fingerprint(adapter)
            for name, adapter in sorted(spec.adapters.items())
        },
    }


def _weight_manifest_entry_fingerprint(entry: WeightManifestEntry) -> Dict[str, Any]:
    return {
        "name": entry.name,
        "dtype": str(entry.dtype),
        "path": _path_fingerprint(entry.path),
        "value": _array_fingerprint(entry.value),
        "scale": _array_fingerprint(entry.scale),
        "shard_axis": entry.shard_axis,
        "shards": [_weight_shard_fingerprint(shard) for shard in entry.shards],
        "quant": _quant_spec_fingerprint(entry.quant),
        "lora": _lora_spec_fingerprint(entry.lora),
        "prefetch": {
            "group": entry.prefetch.group,
            "window": entry.prefetch.window,
            "order": entry.prefetch.order,
            "neighbors": list(entry.prefetch.neighbors),
        },
        "placement": entry.placement,
        "metadata": _sanitize_metadata(entry.metadata),
    }


@dataclass
class ResolvedWeight:
    name: str
    data: np.ndarray
    dtype: np.dtype
    quant: Optional[QuantizedSpec] = None
    scale: Optional[np.ndarray] = None
    lora: Optional[ResolvedLoRA] = None
    placement: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    footprint: Optional[int] = None

    def _to_numpy(self) -> np.ndarray:
        arr = np.asarray(self.data)
        if self.quant is not None and self.quant.is_active():
            arr = self.quant.dequantize_numpy(arr)
        else:
            arr = arr.astype(np.float32, copy=False)
        if self.scale is not None:
            arr = arr * self.scale
        if self.lora is not None:
            arr = self.lora.merge_numpy(arr)
        return arr

    def materialize(self, *, backend: str, device: Optional[Any] = None):
        if backend == "numpy":
            return self._to_numpy()
        if backend == "jax":
            arr = self._to_numpy()
            try:
                import jax.numpy as jnp

                return jnp.asarray(arr)
            except Exception:
                return arr
        if backend == "torch":
            import torch

            base = torch.as_tensor(np.asarray(self.data), device=device)
            if self.quant is not None and self.quant.is_active():
                base = self.quant.dequantize_torch(base)
            else:
                base = base.to(dtype=torch.float32)
            if self.scale is not None:
                scale = torch.as_tensor(self.scale, device=base.device, dtype=base.dtype)
                base = base * scale
            if self.lora is not None:
                base = self.lora.merge_torch(base)
            return base
        raise ValueError(f"Unsupported backend '{backend}'")

    def footprint_bytes(self) -> int:
        total = 0
        if hasattr(self.data, "nbytes"):
            total += int(self.data.nbytes)
        else:
            total += int(np.asarray(self.data).nbytes)
        if self.scale is not None:
            total += int(np.asarray(self.scale).nbytes)
        if self.quant is not None:
            total += int(np.asarray(self.quant.scale).nbytes)
            if self.quant.zero_point is not None:
                total += int(np.asarray(self.quant.zero_point).nbytes)
        if self.lora is not None and self.lora.adapter is not None:
            total += int(self.lora.adapter.up.nbytes + self.lora.adapter.down.nbytes)
        return total


class ManifestWeightStore(WeightStore):
    def __init__(
        self,
        manifest: Union[str, Path, Mapping[str, Any], Sequence[Mapping[str, Any]]],
        *,
        base_path: Optional[Union[str, Path]] = None,
        cache_bytes: int = 2 * 1024**3,
        mmap_mode: bool = True,
        mmap_threshold_bytes: Optional[int] = _DEFAULT_NPZ_MMAP_THRESHOLD,
        strict: bool = False,
    ):
        self.cache_bytes = int(cache_bytes)
        self.mmap_mode = bool(mmap_mode)
        self.mmap_threshold = (
            int(mmap_threshold_bytes) if mmap_threshold_bytes is not None else None
        )
        self.strict_mode = bool(strict)
        self._entries: Dict[str, WeightManifestEntry] = {}
        self._cache: OrderedDict[str, ResolvedWeight] = OrderedDict()
        self._resident_bytes = 0
        self._prefetch_groups: Dict[str, Sequence[WeightManifestEntry]] = {}
        self._group_indices: Dict[str, Dict[str, int]] = {}
        self._lora_slots: Dict[str, Sequence[str]] = {}
        self._active_adapters: Dict[str, Optional[str]] = {}

        manifest_data = manifest
        manifest_path: Optional[Path] = None
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest_data = json.load(fh)
            if base_path is None:
                base_path = manifest_path.parent
        base_path = Path(base_path) if base_path is not None else None

        if isinstance(manifest_data, Mapping):
            weights_conf = manifest_data.get("weights")
            if weights_conf is None:
                raise ValueError("Manifest mapping must include 'weights'")
        else:
            weights_conf = manifest_data

        if not isinstance(weights_conf, Sequence):
            raise ValueError("Manifest weights must be a sequence of entries")

        groups: Dict[str, list] = defaultdict(list)
        slots: Dict[str, list] = defaultdict(list)

        for idx, item in enumerate(weights_conf):
            entry = WeightManifestEntry.from_mapping(
                item,
                base_path=base_path,
                mmap=self.mmap_mode,
                mmap_threshold=self.mmap_threshold,
                strict=self.strict_mode,
                default_order=idx,
            )
            self._entries[entry.name] = entry
            if entry.prefetch.group:
                groups[entry.prefetch.group].append(entry)
            if entry.lora and entry.lora.slot:
                slots[entry.lora.slot].append(entry.name)
                self._active_adapters.setdefault(entry.lora.slot, entry.lora.default)

        for group_name, members in groups.items():
            ordered = sorted(members, key=lambda e: e.prefetch.order)
            self._prefetch_groups[group_name] = ordered
            self._group_indices[group_name] = {entry.name: idx for idx, entry in enumerate(ordered)}

        for slot_name, weights in slots.items():
            self._lora_slots[slot_name] = tuple(weights)

    def resolve(self, name: str) -> ResolvedWeight:
        return self._get_or_build(name, schedule=True)

    # Adapter controls ---------------------------------------------------------
    def available_adapters(self, slot: str) -> Sequence[str]:
        names = self._lora_slots.get(slot)
        if not names:
            return []
        sample = self._entries[names[0]]
        if sample.lora is None:
            return []
        return list(sample.lora.adapters.keys())

    def activate_adapter(self, slot: str, adapter: Optional[str]) -> None:
        weights = self._lora_slots.get(slot)
        if not weights:
            raise KeyError(f"No adapters registered for slot '{slot}'")
        sample_entry = self._entries[weights[0]]
        if sample_entry.lora is None:
            raise KeyError(f"Slot '{slot}' does not have LoRA metadata")
        if adapter is not None and adapter not in sample_entry.lora.adapters:
            raise KeyError(f"Adapter '{adapter}' not available for slot '{slot}'")
        self._active_adapters[slot] = adapter
        # Invalidate cached weights for the slot so they reload on next resolve.
        for weight_name in list(self._cache.keys()):
            entry = self._entries.get(weight_name)
            if entry and entry.lora and entry.lora.slot == slot:
                cached = self._cache.pop(weight_name)
                footprint = cached.footprint or cached.footprint_bytes()
                self._resident_bytes = max(0, self._resident_bytes - footprint)

    # Internal helpers ---------------------------------------------------------
    def _get_or_build(self, name: str, *, schedule: bool) -> ResolvedWeight:
        if name in self._cache:
            weight = self._cache.pop(name)
            self._cache[name] = weight
            if schedule:
                entry = self._entries[name]
                self._schedule_prefetch(entry)
            return weight
        if name not in self._entries:
            raise KeyError(f"Weight '{name}' not found in manifest")
        entry = self._entries[name]
        weight = self._build_weight(entry)
        self._insert_cache(name, weight)
        if schedule:
            self._schedule_prefetch(entry)
        return weight

    def _build_weight(self, entry: WeightManifestEntry) -> ResolvedWeight:
        base_data = self._materialize_entry_data(entry)
        active_lora = None
        if entry.lora is not None:
            slot = entry.lora.slot
            adapter_name = None
            if slot is not None:
                adapter_name = self._active_adapters.get(slot, entry.lora.default)
            else:
                adapter_name = entry.lora.default
            adapter = entry.lora.adapters.get(adapter_name) if adapter_name else None
            active_lora = ResolvedLoRA(merge=entry.lora.merge, adapter=adapter)
        resolved = ResolvedWeight(
            name=entry.name,
            data=base_data,
            dtype=entry.dtype,
            quant=entry.quant,
            scale=entry.scale,
            lora=active_lora,
            placement=entry.placement,
            metadata=dict(entry.metadata),
        )
        resolved.footprint = resolved.footprint_bytes()
        return resolved

    def _materialize_entry_data(self, entry: WeightManifestEntry) -> np.ndarray:
        if entry.value is not None:
            return entry.value
        if entry.shards:
            shards = []
            for shard in entry.shards:
                shard_arr = _load_array_from_path(
                    shard.path,
                    dtype=shard.dtype or entry.dtype,
                    mmap=self.mmap_mode,
                    mmap_threshold=self.mmap_threshold,
                    strict=self.strict_mode,
                )
                shards.append(np.asarray(shard_arr))
            axis = entry.shard_axis
            if axis is None:
                axis = entry.shards[0].axis or 0
            return np.concatenate(shards, axis=axis)
        if entry.path is None:
            raise ValueError(f"Weight entry '{entry.name}' lacks 'path', 'value', or 'shards'")
        return _load_array_from_path(
            entry.path,
            dtype=entry.dtype,
            mmap=self.mmap_mode,
            mmap_threshold=self.mmap_threshold,
            strict=self.strict_mode,
        )

    def _insert_cache(self, name: str, weight: ResolvedWeight) -> None:
        if name in self._cache:
            existing = self._cache.pop(name)
            self._resident_bytes -= existing.footprint or existing.footprint_bytes()
        self._cache[name] = weight
        self._resident_bytes += weight.footprint or weight.footprint_bytes()
        while self._resident_bytes > self.cache_bytes and self._cache:
            evicted_name, evicted_weight = self._cache.popitem(last=False)
            footprint = evicted_weight.footprint or evicted_weight.footprint_bytes()
            self._resident_bytes = max(0, self._resident_bytes - footprint)

    def _schedule_prefetch(self, entry: WeightManifestEntry) -> None:
        spec = entry.prefetch
        if spec.group and spec.window > 0:
            group_entries = self._prefetch_groups.get(spec.group, ())
            index_map = self._group_indices.get(spec.group, {})
            position = index_map.get(entry.name)
            if position is not None:
                for offset in range(1, spec.window + 1):
                    idx = position + offset
                    if idx < len(group_entries):
                        target = group_entries[idx].name
                        self._prefetch_weight(target)
        for neighbor in spec.neighbors:
            if neighbor != entry.name and neighbor in self._entries:
                self._prefetch_weight(neighbor)

    def _prefetch_weight(self, name: str) -> None:
        if name in self._cache:
            return
        if name not in self._entries:
            return
        entry = self._entries[name]
        weight = self._build_weight(entry)
        self._insert_cache(name, weight)

    def fingerprint(self) -> Dict[str, Any]:
        entries = []
        for name in sorted(self._entries):
            entry = self._entries[name]
            entries.append(_weight_manifest_entry_fingerprint(entry))
        return {
            "type": "manifest",
            "entries": entries,
            "cache_bytes": int(self.cache_bytes),
            "mmap_mode": bool(self.mmap_mode),
            "mmap_threshold_bytes": int(self.mmap_threshold)
            if self.mmap_threshold is not None
            else None,
            "strict_mode": bool(self.strict_mode),
            "active_adapters": {
                str(slot): adapter for slot, adapter in sorted(self._active_adapters.items())
            },
        }

    def export_state(self) -> Dict[str, Any]:
        weights: List[Dict[str, Any]] = []
        for entry in self._entries.values():
            record: Dict[str, Any] = {
                "name": entry.name,
                "dtype": str(entry.dtype),
                "path": str(entry.path) if entry.path is not None else None,
                "value_shape": list(entry.value.shape) if entry.value is not None else None,
                "shard_axis": entry.shard_axis,
                "shards": [
                    {
                        "path": str(shard.path),
                        "axis": shard.axis,
                        "dtype": str(shard.dtype) if shard.dtype is not None else None,
                    }
                    for shard in entry.shards
                ],
                "prefetch": {
                    "group": entry.prefetch.group,
                    "window": entry.prefetch.window,
                    "neighbors": list(entry.prefetch.neighbors),
                    "order": entry.prefetch.order,
                },
                "placement": entry.placement,
                "metadata": dict(entry.metadata),
            }
            if entry.quant is not None:
                record["quant"] = {
                    "mode": entry.quant.mode,
                    "dtype": entry.quant.dtype,
                    "group_size": entry.quant.group_size,
                    "scale_shape": list(entry.quant.scale.shape),
                    "has_zero_point": entry.quant.zero_point is not None,
                }
            else:
                record["quant"] = None
            if entry.lora is not None:
                record["lora"] = {
                    "slot": entry.lora.slot,
                    "merge": entry.lora.merge,
                    "default": entry.lora.default,
                    "adapters": {
                        name: {
                            "up_shape": list(adapter.up.shape),
                            "down_shape": list(adapter.down.shape),
                            "alpha": adapter.alpha,
                        }
                        for name, adapter in entry.lora.adapters.items()
                    },
                }
            else:
                record["lora"] = None
            if entry.scale is not None:
                record["scale_shape"] = list(entry.scale.shape)
            else:
                record["scale_shape"] = None
            weights.append(record)
        return {
            "cache_bytes": self.cache_bytes,
            "mmap_mode": self.mmap_mode,
            "mmap_threshold_bytes": self.mmap_threshold,
            "strict_mode": self.strict_mode,
            "active_adapters": dict(self._active_adapters),
            "weights": weights,
        }


@dataclass
class _RemoteCacheEntry:
    name: str
    base_path: Path
    data_file: str
    size: int
    etag: Optional[str]
    last_used: float
    spec_hash: str

    @property
    def data_path(self) -> Path:
        return self.base_path / self.data_file

    @property
    def meta_path(self) -> Path:
        return self.base_path / "meta.json"


@dataclass
class _RemoteObjectMetadata:
    etag: Optional[str]
    size: Optional[int]


def _normalize_etag(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("W/"):
        text = text[2:].strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text or None


def _remote_spec_digest(spec: Mapping[str, Any]) -> str:
    sanitized = _sanitize_metadata(spec)
    payload = json.dumps(sanitized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
    os.replace(tmp, path)


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class _RemoteWeightStoreBase(WeightStore):
    def __init__(
        self,
        weights: Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        *,
        cache_dir: Union[str, Path],
        max_bytes: Optional[int] = 2 * 1024**3,
        mmap_mode: bool = True,
        mmap_threshold_bytes: Optional[int] = _DEFAULT_NPZ_MMAP_THRESHOLD,
        strict: bool = False,
    ):
        self._entries = self._normalize_weights(weights)
        if not self._entries:
            raise ValueError("Remote weight store requires at least one weight specification")
        for name, spec in self._entries.items():
            self._validate_spec(name, spec)

        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if max_bytes is None:
            self.cache_bytes: Optional[int] = None
        else:
            budget = int(max_bytes)
            if budget <= 0:
                raise ValueError("Cache budget must be positive")
            self.cache_bytes = budget
        self.mmap_mode = bool(mmap_mode)
        self.mmap_threshold = (
            int(mmap_threshold_bytes) if mmap_threshold_bytes is not None else None
        )
        self.strict_mode = bool(strict)

        self._spec_hashes: Dict[str, str] = {
            name: _remote_spec_digest(spec) for name, spec in self._entries.items()
        }
        self._cache_keys: Dict[str, str] = {
            name: hashlib.sha256(f"{name}:{self._spec_hashes[name]}".encode("utf-8")).hexdigest()
            for name in self._entries
        }
        self._records: Dict[str, _RemoteCacheEntry] = {}
        self._lru: OrderedDict[str, _RemoteCacheEntry] = OrderedDict()
        self._resident_bytes = 0

        self._load_existing_cache()

    # Lifecycle helpers ------------------------------------------------------
    @staticmethod
    def _normalize_weights(
        weights: Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        if isinstance(weights, Mapping):
            iterator = weights.items()
        else:
            normalized: List[Tuple[str, Mapping[str, Any]]] = []
            for entry in weights:
                if not isinstance(entry, Mapping):
                    raise TypeError("Weight entries must be mappings")
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Weight entry must include a non-empty 'name'")
                spec = {k: v for k, v in entry.items() if k != "name"}
                normalized.append((name, spec))
            iterator = normalized
        for name, spec in iterator:
            if not isinstance(name, str) or not name:
                raise ValueError("Weight names must be non-empty strings")
            if name in result:
                raise ValueError(f"Duplicate weight entry '{name}'")
            if not isinstance(spec, Mapping):
                raise TypeError(f"Weight '{name}' specification must be a mapping")
            result[name] = dict(spec)
        return result

    def _validate_spec(self, name: str, spec: Mapping[str, Any]) -> None:
        del name, spec

    def _cache_dir_for(self, name: str) -> Path:
        key = self._cache_keys[name]
        return self.cache_dir / key[:2] / key

    def _load_existing_cache(self) -> None:
        if not self.cache_dir.exists():
            return
        recovered: Dict[str, _RemoteCacheEntry] = {}
        for meta_path in self.cache_dir.rglob("meta.json"):
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                _remove_tree(meta_path.parent)
                continue
            name = meta.get("name")
            if not isinstance(name, str) or name not in self._entries:
                _remove_tree(meta_path.parent)
                continue
            spec_hash = meta.get("spec_hash")
            if spec_hash != self._spec_hashes[name]:
                _remove_tree(meta_path.parent)
                continue
            expected_dir = self._cache_dir_for(name)
            if meta_path.parent != expected_dir:
                _remove_tree(meta_path.parent)
                continue
            data_file = meta.get("data_file")
            if not isinstance(data_file, str) or not data_file:
                _remove_tree(meta_path.parent)
                continue
            data_path = meta_path.parent / data_file
            if not data_path.exists():
                _remove_tree(meta_path.parent)
                continue
            try:
                size = int(meta.get("size"))
            except (TypeError, ValueError):
                try:
                    size = int(data_path.stat().st_size)
                except OSError:
                    _remove_tree(meta_path.parent)
                    continue
            try:
                last_used = float(meta.get("last_used"))
            except (TypeError, ValueError):
                last_used = 0.0
            etag = _normalize_etag(meta.get("etag"))
            entry = _RemoteCacheEntry(
                name=name,
                base_path=meta_path.parent,
                data_file=data_file,
                size=int(size),
                etag=etag,
                last_used=last_used,
                spec_hash=str(spec_hash),
            )
            recovered[name] = entry

        ordered = sorted(recovered.values(), key=lambda item: item.last_used)
        for entry in ordered:
            self._records[entry.name] = entry
            self._lru[entry.name] = entry
            self._resident_bytes += entry.size

    # Core operations --------------------------------------------------------
    def resolve(self, name: str) -> np.ndarray:
        if name not in self._entries:
            raise KeyError(f"Weight '{name}' not registered with remote store")
        spec = self._entries[name]
        spec_hash = self._spec_hashes[name]
        entry = self._records.get(name)
        if entry is not None and entry.spec_hash != spec_hash:
            self._remove_entry(name)
            entry = None

        metadata = self._fetch_remote_metadata(name, spec)
        remote_etag = _normalize_etag(metadata.etag)
        expected_etag = _normalize_etag(spec.get("etag"))
        if expected_etag is not None:
            if remote_etag is None:
                self._remove_entry(name)
                raise RuntimeError(
                    f"Weight '{name}' missing remote ETag header required for validation"
                )
            if remote_etag != expected_etag:
                self._remove_entry(name)
                raise RuntimeError(
                    f"Weight '{name}' failed ETag validation "
                    f"(expected '{expected_etag}', got '{remote_etag}')"
                )
        effective_etag = remote_etag or expected_etag

        if entry is not None:
            if entry.data_path.exists() and (
                effective_etag is None or entry.etag == effective_etag
            ):
                entry.etag = effective_etag
                self._touch_entry(entry)
                return self._materialize(entry, spec)
            self._remove_entry(name)
            entry = None

        base_path = self._cache_dir_for(name)
        base_path.mkdir(parents=True, exist_ok=True)
        data_file = self._local_filename(name, spec)
        target_path = base_path / data_file
        self._download_remote(name, spec, target_path, metadata)
        try:
            size = int(target_path.stat().st_size)
        except OSError as exc:
            raise RuntimeError(f"Failed to load downloaded weight '{name}': {exc}") from exc
        if self.cache_bytes is not None and size > self.cache_bytes:
            target_path.unlink(missing_ok=True)
            _remove_tree(base_path)
            raise RuntimeError(
                f"Weight '{name}' size {size} exceeds cache budget of {self.cache_bytes} bytes"
            )
        entry = _RemoteCacheEntry(
            name=name,
            base_path=base_path,
            data_file=data_file,
            size=size,
            etag=effective_etag,
            last_used=self._now(),
            spec_hash=spec_hash,
        )
        self._register_entry(entry)
        self._ensure_budget(exclude=name)
        return self._materialize(entry, spec)

    def fingerprint(self) -> Dict[str, Any]:
        entries = []
        for name in sorted(self._entries):
            spec = self._entries[name]
            cached = self._records.get(name)
            entries.append(
                {
                    "name": name,
                    "spec": _sanitize_metadata(spec),
                    "cached": cached is not None and cached.data_path.exists(),
                    "etag": cached.etag if cached is not None else None,
                    "size": cached.size if cached is not None else None,
                }
            )
        return {
            "type": self._fingerprint_kind(),
            "cache_dir": str(self.cache_dir),
            "cache_bytes": self.cache_bytes,
            "resident_bytes": self._resident_bytes,
            **self._fingerprint_extra(),
            "entries": entries,
        }

    # Abstract hooks ---------------------------------------------------------
    def _fingerprint_kind(self) -> str:
        raise NotImplementedError

    def _fingerprint_extra(self) -> Dict[str, Any]:
        return {}

    def _fetch_remote_metadata(self, name: str, spec: Mapping[str, Any]) -> _RemoteObjectMetadata:
        raise NotImplementedError

    def _download_remote(
        self,
        name: str,
        spec: Mapping[str, Any],
        target_path: Path,
        metadata: _RemoteObjectMetadata,
    ) -> None:
        raise NotImplementedError

    def _local_filename(self, name: str, spec: Mapping[str, Any]) -> str:
        raise NotImplementedError

    # Internal helpers -------------------------------------------------------
    def _register_entry(self, entry: _RemoteCacheEntry) -> None:
        existing = self._records.get(entry.name)
        if existing is not None:
            self._resident_bytes = max(0, self._resident_bytes - existing.size)
        self._records[entry.name] = entry
        self._lru.pop(entry.name, None)
        self._lru[entry.name] = entry
        self._resident_bytes += entry.size
        self._write_metadata(entry)

    def _touch_entry(self, entry: _RemoteCacheEntry) -> None:
        entry.last_used = self._now()
        self._lru.pop(entry.name, None)
        self._lru[entry.name] = entry
        self._write_metadata(entry)

    def _write_metadata(self, entry: _RemoteCacheEntry) -> None:
        payload = {
            "name": entry.name,
            "etag": entry.etag,
            "size": entry.size,
            "last_used": entry.last_used,
            "spec_hash": entry.spec_hash,
            "data_file": entry.data_file,
        }
        _write_json_atomic(entry.meta_path, payload)

    def _ensure_budget(self, *, exclude: Optional[str] = None) -> None:
        if self.cache_bytes is None:
            return
        if self._resident_bytes <= self.cache_bytes:
            return
        for name in list(self._lru.keys()):
            if self._resident_bytes <= self.cache_bytes:
                break
            if name == exclude:
                continue
            self._remove_entry(name)
        if self._resident_bytes > self.cache_bytes and exclude is not None:
            entry = self._records.get(exclude)
            if entry is not None:
                size = entry.size
                self._remove_entry(exclude)
                raise RuntimeError(
                    f"Cache budget of {self.cache_bytes} bytes too small for weight "
                    f"'{exclude}' (size {size})"
                )

    def _remove_entry(self, name: str) -> None:
        entry = self._records.pop(name, None)
        if entry is None:
            return
        self._lru.pop(name, None)
        self._resident_bytes = max(0, self._resident_bytes - entry.size)
        _remove_tree(entry.base_path)

    def _materialize(self, entry: _RemoteCacheEntry, spec: Mapping[str, Any]) -> np.ndarray:
        dtype = spec.get("dtype")
        resolved_dtype = np.dtype(dtype) if dtype is not None else None
        return _load_array_from_path(
            entry.data_path,
            dtype=resolved_dtype,
            mmap=self.mmap_mode,
            mmap_threshold=self.mmap_threshold,
            strict=self.strict_mode,
        )

    @staticmethod
    def _now() -> float:
        return time.time()


class HTTPWeightStore(_RemoteWeightStoreBase):
    def __init__(
        self,
        weights: Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        *,
        cache_dir: Union[str, Path],
        max_bytes: Optional[int] = 2 * 1024**3,
        mmap_mode: bool = True,
        mmap_threshold_bytes: Optional[int] = _DEFAULT_NPZ_MMAP_THRESHOLD,
        strict: bool = False,
        timeout: float = 30.0,
        headers: Optional[Mapping[str, str]] = None,
    ):
        self._timeout = float(timeout)
        self._base_headers = {str(k): str(v) for k, v in (headers or {}).items()}
        super().__init__(
            weights,
            cache_dir=cache_dir,
            max_bytes=max_bytes,
            mmap_mode=mmap_mode,
            mmap_threshold_bytes=mmap_threshold_bytes,
            strict=strict,
        )

    def _fingerprint_kind(self) -> str:
        return "http"

    def _fingerprint_extra(self) -> Dict[str, Any]:
        return {"timeout": self._timeout}

    def _validate_spec(self, name: str, spec: Mapping[str, Any]) -> None:
        url = spec.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError(f"HTTP weight '{name}' must include a 'url'")
        if spec.get("headers") is not None and not isinstance(spec["headers"], Mapping):
            raise ValueError(f"HTTP weight '{name}' headers must be a mapping")

    def _local_filename(self, name: str, spec: Mapping[str, Any]) -> str:
        del name
        url = str(spec["url"])
        parsed = urllib_parse.urlparse(url)
        suffix = Path(parsed.path).suffix
        if not suffix:
            suffix = ".npy"
        return f"payload{suffix}"

    def _build_headers(self, spec: Mapping[str, Any]) -> Dict[str, str]:
        headers = dict(self._base_headers)
        extra = spec.get("headers")
        if isinstance(extra, Mapping):
            headers.update({str(k): str(v) for k, v in extra.items()})
        return headers

    def _fetch_remote_metadata(self, name: str, spec: Mapping[str, Any]) -> _RemoteObjectMetadata:
        url = str(spec["url"])
        headers = self._build_headers(spec)
        request_obj = urllib_request.Request(url, method="HEAD", headers=headers)
        try:
            with urllib_request.urlopen(request_obj, timeout=self._timeout) as response:
                info = response.info()
                etag = info.get("ETag")
                length = info.get("Content-Length")
        except urllib_error.HTTPError as exc:
            if exc.code in {403, 404}:
                raise FileNotFoundError(f"Remote weight '{name}' not found at {url}") from exc
            if exc.code in {405, 501}:
                return _RemoteObjectMetadata(etag=None, size=None)
            raise RuntimeError(f"Failed to query metadata for weight '{name}': {exc}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Failed to reach '{url}' for weight '{name}': {exc}") from exc
        size: Optional[int]
        try:
            size = int(length) if length is not None else None
        except ValueError:
            size = None
        return _RemoteObjectMetadata(etag=etag, size=size)

    def _download_remote(
        self,
        name: str,
        spec: Mapping[str, Any],
        target_path: Path,
        metadata: _RemoteObjectMetadata,
    ) -> None:
        del metadata
        url = str(spec["url"])
        headers = self._build_headers(spec)
        request_obj = urllib_request.Request(url, headers=headers)
        tmp_path = target_path.with_name(f"{target_path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (
                urllib_request.urlopen(request_obj, timeout=self._timeout) as response,
                tmp_path.open("wb") as fh,
            ):
                shutil.copyfileobj(response, fh)
        except urllib_error.HTTPError as exc:
            tmp_path.unlink(missing_ok=True)
            if exc.code == 404:
                raise FileNotFoundError(f"Remote weight not found at {url}") from exc
            raise RuntimeError(f"HTTP download failed for '{url}': {exc}") from exc
        except urllib_error.URLError as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download '{url}': {exc}") from exc
        os.replace(tmp_path, target_path)


class S3WeightStore(_RemoteWeightStoreBase):
    def __init__(
        self,
        bucket: str,
        objects: Union[Mapping[str, Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        *,
        cache_dir: Union[str, Path],
        max_bytes: Optional[int] = 2 * 1024**3,
        mmap_mode: bool = True,
        mmap_threshold_bytes: Optional[int] = _DEFAULT_NPZ_MMAP_THRESHOLD,
        strict: bool = False,
        client: Optional[Any] = None,
        session_kwargs: Optional[Mapping[str, Any]] = None,
        client_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.bucket = str(bucket)
        self._session_kwargs = dict(session_kwargs or {})
        self._client_kwargs = dict(client_kwargs or {})
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - requires optional dependency
                raise RuntimeError(
                    "S3WeightStore requires boto3. Install 'boto3' to enable S3 support."
                ) from exc
            session = boto3.session.Session(**self._session_kwargs)
            self._client = session.client("s3", **self._client_kwargs)
        super().__init__(
            objects,
            cache_dir=cache_dir,
            max_bytes=max_bytes,
            mmap_mode=mmap_mode,
            mmap_threshold_bytes=mmap_threshold_bytes,
            strict=strict,
        )

    def _fingerprint_kind(self) -> str:
        return "s3"

    def _fingerprint_extra(self) -> Dict[str, Any]:
        return {"bucket": self.bucket}

    def _validate_spec(self, name: str, spec: Mapping[str, Any]) -> None:
        key = spec.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"S3 weight '{name}' must include a 'key'")
        if spec.get("extra_args") is not None and not isinstance(spec["extra_args"], Mapping):
            raise ValueError(f"S3 weight '{name}' extra_args must be a mapping")

    def _local_filename(self, name: str, spec: Mapping[str, Any]) -> str:
        del name
        key = str(spec["key"])
        suffix = Path(key).suffix
        if not suffix:
            suffix = ".npy"
        return f"payload{suffix}"

    def _fetch_remote_metadata(self, name: str, spec: Mapping[str, Any]) -> _RemoteObjectMetadata:
        params: Dict[str, Any] = {"Bucket": self.bucket, "Key": spec["key"]}
        version = spec.get("version_id")
        if version:
            params["VersionId"] = version
        try:
            response = self._client.head_object(**params)
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(
                    f"S3 object for weight '{name}' not found (bucket={self.bucket}, key={spec['key']})"
                ) from exc
            raise RuntimeError(f"Failed to query S3 metadata for weight '{name}': {exc}") from exc
        etag = response.get("ETag")
        size_raw = response.get("ContentLength")
        try:
            size = int(size_raw) if size_raw is not None else None
        except (TypeError, ValueError):
            size = None
        return _RemoteObjectMetadata(etag=etag, size=size)

    def _download_remote(
        self,
        name: str,
        spec: Mapping[str, Any],
        target_path: Path,
        metadata: _RemoteObjectMetadata,
    ) -> None:
        del metadata
        params: Dict[str, Any] = {"Bucket": self.bucket, "Key": spec["key"]}
        version = spec.get("version_id")
        if version:
            params["VersionId"] = version
        extra_args = spec.get("extra_args")
        if isinstance(extra_args, Mapping):
            for key, value in extra_args.items():
                if key not in {"Bucket", "Key"}:
                    params[key] = value
        tmp_path = target_path.with_name(f"{target_path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = self._client.get_object(**params)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download S3 object for weight '{name}': {exc}") from exc
        body = response.get("Body")
        if body is None:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("S3 response missing streaming body")
        try:
            with tmp_path.open("wb") as fh:
                while True:
                    chunk = body.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
        finally:
            if hasattr(body, "close"):
                body.close()
        os.replace(tmp_path, target_path)


@dataclass
class RuntimePolicies:
    weight_store: Optional[WeightStore] = None
    sharding: Optional[ShardingPolicy] = None
    quantization: Optional[QuantizationPolicy] = None
    lora: Optional[LoRAPolicy] = None
    output_root: Optional[Path] = None

    def apply(self, name: str, value: np.ndarray) -> np.ndarray:
        out = value
        if self.quantization is not None:
            out = self.quantization.apply(name, out)
        if self.lora is not None:
            out = self.lora.apply(name, out)
        if self.sharding is not None:
            out = self.sharding.materialize(name, out)
        return out

    def materialize_weight(
        self,
        name: str,
        payload: Any,
        *,
        backend: str,
        device: Optional[Any] = None,
    ):
        managed_quant = False
        managed_lora = False
        materialized = payload

        if isinstance(payload, ResolvedWeight):
            materialized = payload.materialize(backend=backend, device=device)
            managed_quant = payload.quant is not None and payload.quant.is_active()
            managed_lora = payload.lora is not None and payload.lora.adapter is not None
        elif isinstance(payload, np.ndarray):
            materialized = payload
        elif isinstance(payload, SparseBoolTensor):
            materialized = payload.to_dense()
        elif hasattr(payload, "to_dense"):
            try:
                materialized = payload.to_dense()
            except Exception:  # pragma: no cover - fallback for unexpected payloads
                materialized = payload

        if backend == "numpy":
            arr = np.asarray(materialized)
            if self.quantization is not None and not managed_quant:
                arr = self.quantization.apply(name, arr)
            if self.lora is not None and not managed_lora:
                arr = self.lora.apply(name, arr)
            if self.sharding is not None:
                arr = self.sharding.materialize(name, arr)
            return arr

        if backend == "torch":
            import torch

            tensor = (
                materialized
                if isinstance(materialized, torch.Tensor)
                else torch.as_tensor(materialized, device=device)
            )
            if self.quantization is not None and not managed_quant:
                tensor = torch.as_tensor(
                    self.quantization.apply(name, tensor.detach().cpu().numpy()),
                    device=tensor.device,
                )
            if self.lora is not None and not managed_lora:
                tensor_np = tensor.detach().cpu().numpy()
                tensor = torch.as_tensor(self.lora.apply(name, tensor_np), device=tensor.device)
            return tensor

        if backend == "jax":
            arr = np.asarray(materialized)
            if self.quantization is not None and not managed_quant:
                arr = self.quantization.apply(name, arr)
            if self.lora is not None and not managed_lora:
                arr = self.lora.apply(name, arr)
            try:
                import jax.numpy as jnp

                arr = jnp.asarray(arr)
            except Exception:
                arr = np.asarray(arr)
            if self.sharding is not None:
                arr = self.sharding.materialize(name, arr)
            return arr

        return materialized

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "weight_store": _fingerprint_weight_store(self.weight_store),
            "sharding": self.sharding.fingerprint() if self.sharding is not None else None,
            "quantization": self.quantization.fingerprint()
            if self.quantization is not None
            else None,
            "lora": self.lora.fingerprint() if self.lora is not None else None,
            "output_root": _fingerprint_path(self.output_root),
        }

    def resolve_output_path(self, path: Union[str, Path]) -> Path:
        if path is None:
            raise BackendError("Output path is required for sink operations")
        raw = Path(path)
        if raw.is_absolute():
            resolved_raw = raw.expanduser().resolve()
            if self.output_root is None:
                resolved_raw.parent.mkdir(parents=True, exist_ok=True)
                return resolved_raw
            try:
                raw = resolved_raw.relative_to(resolved_raw.anchor)
            except ValueError as exc:
                raise BackendError(f"Output path '{raw}' is not local to filesystem root") from exc
        root = Path(self.output_root) if self.output_root is not None else Path.cwd()
        root = root.expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise BackendError(f"Configured output root '{root}' is not a directory")
        root.mkdir(parents=True, exist_ok=True)
        candidate = root / raw
        normalized = candidate.resolve()
        try:
            normalized.relative_to(root)
        except ValueError as exc:
            raise BackendError(f"Output path '{raw}' escapes sandbox root '{root}'") from exc
        normalized.parent.mkdir(parents=True, exist_ok=True)
        return normalized


def _fingerprint_weight_store(store: Optional[WeightStore]) -> Any:
    if store is None:
        return None
    fingerprint_fn = getattr(store, "fingerprint", None)
    if callable(fingerprint_fn):
        try:
            return fingerprint_fn()
        except Exception:
            return {"repr": repr(store)}
    return {"repr": repr(store)}
