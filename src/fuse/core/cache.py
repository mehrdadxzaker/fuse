import hashlib
import importlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .evaluator_numpy import ExecutionConfig
from .exceptions import CacheError
from .policies import RuntimePolicies

CACHE_VERSION = "2"
_SHA256_KEY_RE = re.compile(r"[0-9a-f]{64}")


def compute_program_hash(src: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(src.encode("utf-8"))
    return hasher.hexdigest()


@dataclass
class CacheRecord:
    payload: Any
    metadata: dict


class CacheManager:
    def __init__(self, cache_dir: str):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, backend: str, key: str) -> Path:
        if not _SHA256_KEY_RE.fullmatch(key):
            raise CacheError(
                f"Invalid cache key '{key}'. Expected 64-character lowercase hex digest."
            )
        safe_backend = backend.replace("/", "_")
        return self.root / safe_backend / key

    def load(self, backend: str, key: str) -> Optional[CacheRecord]:
        base = self.path_for(backend, key)
        meta_path = base.with_suffix(".json")
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            return None
        if data.get("cache_version") != CACHE_VERSION:
            return None
        metadata = data.get("metadata", {})
        payload_meta = data.get("payload_meta")
        payload = None
        if payload_meta is not None:
            payload_path = base.with_suffix(".npz")
            if not payload_path.exists():
                return None
            arrays = _load_npz(payload_path)
            payload = _deserialize_payload(arrays, payload_meta)
        return CacheRecord(payload=payload, metadata=metadata)

    def store(self, backend: str, key: str, payload: Any, metadata: Optional[dict] = None):
        base = self.path_for(backend, key)
        base.parent.mkdir(parents=True, exist_ok=True)
        sanitized_metadata = _sanitize_for_json(metadata or {})
        arrays, payload_meta = _serialize_payload(payload)
        meta_payload: Dict[str, Any] = {
            "cache_version": CACHE_VERSION,
            "metadata": sanitized_metadata,
        }
        if payload_meta is not None:
            meta_payload["payload_meta"] = payload_meta
            _write_npz(base.with_suffix(".npz"), arrays)
        else:
            payload_path = base.with_suffix(".npz")
            if payload_path.exists():
                payload_path.unlink()
        _write_json(base.with_suffix(".json"), meta_payload)

    def write_metadata(self, backend: str, key: str, metadata: dict):
        base = self.path_for(backend, key)
        base.parent.mkdir(parents=True, exist_ok=True)
        payload_path = base.with_suffix(".npz")
        if payload_path.exists():
            payload_path.unlink()
        _write_json(
            base.with_suffix(".json"),
            {
                "cache_version": CACHE_VERSION,
                "metadata": _sanitize_for_json(metadata),
            },
        )


def _sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in value]
    return repr(value)


def _serialize_payload(payload: Any) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]:
    if payload is None:
        return {}, None
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
        array = np.frombuffer(raw, dtype=np.uint8).copy()
        return {"data": array}, {"type": "bytes", "dataset": "data"}
    if isinstance(payload, np.ndarray):
        array = _ensure_serializable_array(np.asarray(payload))
        return {"data": array}, {"type": "ndarray", "dataset": "data"}
    if hasattr(payload, "shape") and hasattr(payload, "dtype") and hasattr(payload, "tobytes"):
        array = _ensure_serializable_array(np.asarray(payload))
        return {"data": array}, {"type": "ndarray", "dataset": "data"}
    if isinstance(payload, dict):
        arrays: Dict[str, np.ndarray] = {}
        items_meta: List[Dict[str, str]] = []
        for idx, key in enumerate(sorted(payload.keys(), key=lambda x: str(x))):
            dataset = f"item_{idx}"
            arrays[dataset] = _ensure_serializable_array(np.asarray(payload[key]))
            items_meta.append({"key": str(key), "dataset": dataset})
        return arrays, {"type": "mapping", "items": list(items_meta)}
    if isinstance(payload, (list, tuple)):
        arrays: Dict[str, np.ndarray] = {}
        datasets: list[str] = []
        for idx, value in enumerate(payload):
            dataset = f"item_{idx}"
            arrays[dataset] = _ensure_serializable_array(np.asarray(value))
            datasets.append(dataset)
        return arrays, {"type": "sequence", "datasets": datasets}
    raise CacheError(
        "Cache payload must be bytes, numpy arrays, or JSON/array compatible structures"
    )


def _deserialize_payload(arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Any:
    payload_type = meta.get("type")
    if payload_type == "bytes":
        dataset = meta.get("dataset")
        blob = arrays.get(dataset)
        if blob is None:
            return None
        return bytes(np.asarray(blob, dtype=np.uint8).tobytes())
    if payload_type == "ndarray":
        dataset = meta.get("dataset")
        if dataset is None:
            return None
        array = arrays.get(dataset)
        return array
    if payload_type == "mapping":
        items = meta.get("items", [])
        result: Dict[str, np.ndarray] = {}
        for item in items:
            dataset = item.get("dataset")
            key = item.get("key")
            if dataset is None or key is None:
                continue
            result[str(key)] = arrays.get(dataset)
        return result
    if payload_type == "sequence":
        datasets = meta.get("datasets", [])
        return [arrays.get(name) for name in datasets]
    return None


def _ensure_serializable_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == object:
        raise CacheError("Cache payload arrays must not use object dtype")
    return arr


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _module_version(name: str) -> Optional[str]:
    module = sys.modules.get(name)
    if module is None:
        try:
            module = importlib.import_module(name)
        except Exception:
            return None
    return getattr(module, "__version__", None)


def _fuse_version() -> str:
    try:
        return importlib.metadata.version("fuse-ai")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        root = Path(__file__).resolve().parents[3]
        pyproject = root / "pyproject.toml"
        if pyproject.exists():
            text = pyproject.read_text(encoding="utf-8")
            match = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', text, flags=re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass
    return "0.0.0"


def _dependency_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "fuse": _fuse_version(),
        "numpy": getattr(np, "__version__", None),
        "torch": _module_version("torch"),
        "jax": _module_version("jax"),
    }
    return versions


def _schedule_fingerprint(schedule: Any) -> Any:
    manifest = getattr(schedule, "manifest", None)
    if callable(manifest):
        try:
            return _sanitize_for_json(manifest())
        except Exception:
            return {"repr": repr(schedule)}
    return {"repr": repr(schedule)}


def _execution_config_fingerprint(config: Optional["ExecutionConfig"]) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    cfg = config.normalized() if hasattr(config, "normalized") else config
    data: Dict[str, Any] = {
        "mode": getattr(cfg, "mode", None),
        "max_iters": int(getattr(cfg, "max_iters", 0)),
        "tol": float(getattr(cfg, "tol", 0.0)),
        "chaining": getattr(cfg, "chaining", None),
        "explain_timings": bool(getattr(cfg, "explain_timings", False)),
        "projection_strategy": getattr(cfg, "projection_strategy", None),
        "projection_samples": getattr(cfg, "projection_samples", None),
        "projection_seed": getattr(cfg, "projection_seed", None),
    }
    temperatures = getattr(cfg, "temperatures", None)
    if temperatures:
        data["temperatures"] = {
            str(name): _schedule_fingerprint(schedule)
            for name, schedule in sorted(temperatures.items())
        }
    else:
        data["temperatures"] = None
    return data


def _policies_fingerprint(policies: Optional["RuntimePolicies"]) -> Any:
    if policies is None:
        return None
    fingerprint_fn = getattr(policies, "fingerprint", None)
    if callable(fingerprint_fn):
        try:
            return _sanitize_for_json(fingerprint_fn())
        except Exception:
            return {"repr": repr(policies)}
    return {"repr": repr(policies)}


def cache_fingerprint(
    *,
    program_src: str,
    backend: str,
    artifact: Optional[str] = None,
    device: Optional[str] = None,
    execution_config: Optional["ExecutionConfig"] = None,
    policies: Optional["RuntimePolicies"] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "source": compute_program_hash(program_src),
        "backend": backend,
        "artifact": artifact,
        "device": device,
        "versions": _dependency_versions(),
        "execution_config": _execution_config_fingerprint(execution_config),
        "policies": _policies_fingerprint(policies),
        "extra": _sanitize_for_json(extra) if extra else None,
    }


def cache_key_from_fingerprint(fingerprint: Dict[str, Any]) -> str:
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def build_cache_key(
    *,
    program_src: str,
    backend: str,
    artifact: Optional[str] = None,
    device: Optional[str] = None,
    execution_config: Optional["ExecutionConfig"] = None,
    policies: Optional["RuntimePolicies"] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    fingerprint = cache_fingerprint(
        program_src=program_src,
        backend=backend,
        artifact=artifact,
        device=device,
        execution_config=execution_config,
        policies=policies,
        extra=extra,
    )
    return cache_key_from_fingerprint(fingerprint)
