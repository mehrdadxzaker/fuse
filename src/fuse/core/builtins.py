import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    from safetensors.numpy import load_file as _load_safetensors
except Exception:  # pragma: no cover - optional dependency missing
    _load_safetensors = None

_VOCAB_SCHEMA_VERSION = 1
_DEFAULT_NPZ_MMAP_THRESHOLD = 64 * 1024 * 1024  # 64 MiB


class SparseBoolTensor:
    def __init__(self, coords: np.ndarray, shape: Sequence[int]):
        coords = np.asarray(coords, dtype=np.int64)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        if coords.size == 0:
            arity = len(shape)
            coords = np.zeros((0, arity), dtype=np.int64)
        self.coords = coords
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.int8

    def to_dense(self) -> np.ndarray:
        arr = np.zeros(self.shape, dtype=self.dtype)
        if self.coords.size == 0:
            return arr
        index = tuple(self.coords[:, ax] for ax in range(self.coords.shape[1]))
        arr[index] = 1
        return arr

    def nnz(self) -> int:
        return int(self.coords.shape[0])

    def __array__(self, dtype=None):
        raise TypeError("SparseBoolTensor cannot be implicitly converted to dense; call to_dense() explicitly")

    def __repr__(self):
        return f"SparseBoolTensor(shape={self.shape}, nnz={self.nnz()})"


class BagOfWordsTensor:
    """
    Dense bag-of-words matrix that preserves the vocabulary mapping.

    Behaves like a NumPy array for downstream consumers while exposing
    ``vocab`` so callers can recover the token ordering.
    """

    def __init__(self, matrix: np.ndarray, vocab: Dict[str, int], vocab_path: Optional[Path] = None):
        self.matrix = np.asarray(matrix, dtype=np.int8)
        self.vocab = dict(vocab)
        self.vocab_path = vocab_path

    def __array__(self, dtype=None):
        if dtype is not None:
            return self.matrix.astype(dtype)
        return self.matrix

    def to_dense(self) -> np.ndarray:
        return np.asarray(self)

    def __repr__(self) -> str:  # pragma: no cover - repr only used for debugging
        return f"BagOfWordsTensor(shape={self.matrix.shape}, vocab_size={len(self.vocab)})"


def _vocab_sidecar_path(path: Path) -> Path:
    if path.suffix:
        base = path.with_suffix("")
    else:
        base = path
    return base.with_suffix(".vocab.json")


def _load_vocab_sidecar(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if "items" in data:
            items = data.get("items", [])
            mapping: Dict[str, int] = {}
            for entry in items:
                token = str(entry["token"])
                mapping[token] = int(entry["index"])
            return mapping
        return {str(k): int(v) for k, v in data.items()}
    if isinstance(data, list):
        return {str(token): idx for idx, token in enumerate(data)}
    raise ValueError(f"Unrecognised vocabulary sidecar format at '{path}'")


def _dump_vocab_sidecar(path: Path, vocab: Dict[str, int]) -> None:
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    payload = {
        "version": _VOCAB_SCHEMA_VERSION,
        "items": [{"token": token, "index": int(idx)} for token, idx in items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def _build_bow_tensor(path: Path) -> BagOfWordsTensor:
    text = path.read_text(encoding="utf-8").strip()
    tokens = text.split()
    vocab_path = _vocab_sidecar_path(path)
    vocab: Dict[str, int] = {}
    if vocab_path.exists():
        try:
            vocab = _load_vocab_sidecar(vocab_path)
        except Exception:
            vocab = {}

    changed = False
    if not vocab and tokens:
        # Initialise from first occurrence order for determinism.
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        changed = True
    else:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
                changed = True

    column_count = len(vocab)
    matrix = np.zeros((len(tokens), column_count), dtype=np.int8)
    if column_count and tokens:
        for position, token in enumerate(tokens):
            idx = vocab[token]
            matrix[position, idx] = 1

    if changed or not vocab_path.exists():
        _dump_vocab_sidecar(vocab_path, vocab)

    return BagOfWordsTensor(matrix, vocab, vocab_path)


def _is_topk_payload(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for row in value:
        if not isinstance(row, (list, tuple)):
            return False
        for pair in row:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                return False
    return True

def step(x): return (x > 0).astype(np.int8)
def relu(x): return np.maximum(0, x)
def sig(x, T=None, *, zero_tol: float = 1e-9):
    arr = np.asarray(x)
    if T is None:
        return (arr > 0).astype(arr.dtype)
    temp = float(T)
    if abs(temp) <= zero_tol:
        return (arr > 0).astype(arr.dtype)
    result = 1.0 / (1.0 + np.exp(-arr / temp))
    if np.issubdtype(arr.dtype, np.floating):
        return result.astype(arr.dtype)
    return result.astype(np.float32)
def gelu(x):
    return 0.5*x*(1.0 + np.tanh(np.sqrt(2.0/np.pi)*(x + 0.044715*np.power(x,3))))

def gelu_grad(x):
    c = np.sqrt(2.0 / np.pi)
    x3 = np.power(x, 3)
    inner = c * (x + 0.044715 * x3)
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - np.power(tanh_inner, 2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * (c * (1.0 + 0.134145 * np.power(x, 2)))
def softmax(x, axis=-1):
    arr = np.asarray(x)
    arr = arr - np.max(arr, axis=axis, keepdims=True)
    e = np.exp(arr)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / s

def masked_softmax(x, mask=None, axis=-1, fill_value=None):
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    mask_arr = None
    if mask is not None:
        mask_arr = np.asarray(mask).astype(bool)
        if fill_value is None:
            finfo = np.finfo(arr.dtype)
            fill = finfo.min
        else:
            fill = float(np.asarray(fill_value).reshape(()))
        arr = np.where(mask_arr, arr, fill)
    result = softmax(arr, axis=axis)
    if mask_arr is not None:
        result = np.where(mask_arr, result, 0.0)
    return result

def softmax_grad(y, grad, axis=-1):
    y = np.asarray(y)
    grad = np.asarray(grad)
    dot = np.sum(grad * y, axis=axis, keepdims=True)
    return (grad - dot) * y


def _ensure_float_array(value):
    if isinstance(value, SparseBoolTensor):
        arr = value.to_dense()
    else:
        arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    return arr


def _normalize_tucker_ranks(rank: Optional[Union[int, Sequence[int]]], shape: Sequence[int]) -> List[int]:
    if rank is None:
        return [max(1, min(dim, int(np.ceil(np.sqrt(dim))))) for dim in shape]
    if isinstance(rank, (int, np.integer)):
        value = int(rank)
        return [max(1, min(dim, value)) for dim in shape]
    ranks = list(rank)
    if len(ranks) != len(shape):
        raise ValueError(f"rank specification must match tensor order (got {len(ranks)} for {len(shape)})")
    normalized = []
    for dim, item in zip(shape, ranks):
        val = int(item)
        if val <= 0:
            raise ValueError("tucker_dense ranks must be positive")
        normalized.append(min(dim, val))
    return normalized


def _mode_product_np(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    moved = np.moveaxis(tensor, mode, 0)
    leading = moved.shape[0]
    rest = moved.shape[1:]
    flat = moved.reshape(leading, -1)
    if matrix.shape[1] != leading:
        raise ValueError(f"matrix shape {matrix.shape} not compatible with axis size {leading}")
    result = matrix @ flat
    new_shape = (matrix.shape[0],) + rest
    return np.moveaxis(result.reshape(new_shape), 0, mode)


def _tucker_decompose_np(arr: np.ndarray, ranks: Sequence[int], *, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    factors: List[np.ndarray] = []
    for mode in range(arr.ndim):
        unfolded = np.moveaxis(arr, mode, 0).reshape(arr.shape[mode], -1)
        try:
            u, s, _ = np.linalg.svd(unfolded, full_matrices=False)
        except np.linalg.LinAlgError:
            noise_source = rng.standard_normal if rng is not None else np.random.standard_normal
            u, s, _ = np.linalg.svd(unfolded + 1e-6 * noise_source(unfolded.shape), full_matrices=False)
        r = min(ranks[mode], u.shape[1])
        factors.append(u[:, :r])
    core = arr
    for mode, factor in enumerate(factors):
        core = _mode_product_np(core, factor.T, mode)
    return core, factors


def _tucker_reconstruct_np(core: np.ndarray, factors: Sequence[np.ndarray]) -> np.ndarray:
    result = core
    for mode, factor in enumerate(factors):
        result = _mode_product_np(result, factor, mode)
    return result


def tucker_dense(value, rank=None, threshold: float = 0.5, *, rng: Optional[np.random.Generator] = None):
    """
    Approximate a sparse/high-order relation with a low-rank Tucker reconstruction
    and return a denoised dense tensor via step().
    """
    arr = _ensure_float_array(value)
    if arr.ndim == 0:
        return step(arr - threshold)
    ranks = _normalize_tucker_ranks(rank, arr.shape)
    if all(r == dim for r, dim in zip(ranks, arr.shape)):
        approx = arr
    else:
        core, factors = _tucker_decompose_np(arr, ranks, rng=rng)
        approx = _tucker_reconstruct_np(core, factors)
    approx = approx - float(threshold)
    return step(approx)
def lnorm(x, axis=-1, eps=1e-5):
    mu = np.mean(x, axis=axis, keepdims=True)
    var = np.mean((x-mu)**2, axis=axis, keepdims=True)
    return (x-mu)/np.sqrt(var+eps)

def layernorm(x, axis=-1, eps=1e-5):
    return lnorm(x, axis=axis, eps=eps)

def rope(x, pos_axis_value):
    # Simple RoPE assuming last dim pairs (cos,sin) rotation by position index provided in pos_axis_value (int)
    # For demo: pos is an integer index per row; we assume caller broadcasts correctly.
    d = x.shape[-1]
    half = d//2
    cos = np.cos(np.arange(half))[None, ...]
    sin = np.sin(np.arange(half))[None, ...]
    xr, xi = x[..., :half], x[..., half:half*2]
    # naive rotation ignoring pos; suitable as placeholder
    return np.concatenate([xr*cos - xi*sin, xr*sin + xi*cos], axis=-1)

def causal_mask(L):
    m = np.tril(np.ones((L,L), dtype=np.int8))
    return m

def const(val): 
    return np.array(val)

def _resolve_attention_scale(scale, dim_last):
    if scale is None:
        return 1.0 / np.sqrt(float(dim_last))
    scale_arr = np.asarray(scale)
    if scale_arr.size == 1:
        return float(scale_arr.reshape(()))
    raise ValueError("attention scale must be a scalar")

def attention(query, key, value, mask=None, scale=None, causal=False):
    q = np.asarray(query)
    k = np.asarray(key)
    v = np.asarray(value)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("attention requires query/key to share last dimension")
    scale_factor = _resolve_attention_scale(scale, q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale_factor
    mask_arr = None
    if mask is not None:
        mask_arr = np.asarray(mask).astype(bool)
    if causal:
        seq_len = scores.shape[-2]
        mem_len = scores.shape[-1]
        causal_mask = np.tril(np.ones((seq_len, mem_len), dtype=bool))
        if mask_arr is None:
            mask_arr = causal_mask
        else:
            mask_arr = np.logical_and(mask_arr, causal_mask)
    fill = None
    if mask_arr is not None:
        fill = np.finfo(scores.dtype if np.issubdtype(scores.dtype, np.floating) else np.float32).min
    weights = masked_softmax(scores, mask=mask_arr, axis=-1, fill_value=fill)
    return np.matmul(weights, v)

def concat(*arrays, axis: Union[int, None]=None):
    # Allow passing a single iterable of arrays
    if len(arrays) == 1 and isinstance(arrays[0], (list, tuple)):
        arrays = tuple(arrays[0])
    arrays = tuple(np.asarray(a) for a in arrays)
    if not arrays:
        raise ValueError("concat requires at least one array")
    if axis is None and len(arrays) == 1:
        arr = arrays[0]
        if arr.ndim < 2:
            return arr
        new_shape = (*arr.shape[:-2], arr.shape[-2] * arr.shape[-1])
        return arr.reshape(new_shape)
    if axis is None:
        axis = -1
    return np.concatenate(arrays, axis=axis)

def reduce_max(x, axis=None, keepdims=False):
    return np.max(x, axis=axis, keepdims=keepdims)

def reduce_mean(x, axis=None, keepdims=False):
    return np.mean(x, axis=axis, keepdims=keepdims)

def topk(arr, k=5, *, rng: Optional[np.random.Generator] = None):
    arr = np.asarray(arr)
    single = False
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        single = True
    if rng is None:
        rng = np.random.default_rng()
    results: List[List[Tuple[int, float]]] = []
    for row in arr:
        row = np.asarray(row)
        k_eff = min(k, row.size)
        if k_eff <= 0:
            results.append([])
            continue
        idx = np.argpartition(row, -k_eff)[-k_eff:]
        values = row[idx]
        tie_break = rng.random(len(idx))
        order = np.lexsort((tie_break, -values))
        ordered_idx = idx[order]
        results.append([(int(i), float(row[i])) for i in ordered_idx])
    if single:
        return results[0]
    return results

def _should_mmap_npz(file_path: Path, threshold: Optional[int]) -> bool:
    if threshold is None:
        return True
    try:
        return file_path.stat().st_size >= int(threshold)
    except OSError:
        return True


def read_tensor_from_file(
    path: str,
    *,
    strict: bool = False,
    mmap_threshold_bytes: Optional[int] = _DEFAULT_NPZ_MMAP_THRESHOLD,
):
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext == ".npy":
        arr = np.load(file_path, allow_pickle=False, mmap_mode="r")
        if strict and not isinstance(arr, np.memmap):
            raise RuntimeError(f"Strict loading forbids materializing array '{file_path.name}'")
        return arr
    if ext == ".npz":
        use_mmap = True if strict else _should_mmap_npz(file_path, mmap_threshold_bytes)
        mmap_mode = "r" if use_mmap else None
        if strict and mmap_mode is None:
            raise RuntimeError(f"Strict loading requires mmap support for '{file_path.name}'")
        with np.load(file_path, allow_pickle=False, mmap_mode=mmap_mode) as data:
            if not data.files:
                raise ValueError(f"Archive '{file_path}' contained no arrays")
            if "arr_0" in data.files:
                payload: Union[np.ndarray, Dict[str, np.ndarray]] = data["arr_0"]
            elif len(data.files) == 1:
                payload = data[data.files[0]]
            else:
                payload = {name: data[name] for name in data.files}
            if strict:
                values = payload.values() if isinstance(payload, dict) else (payload,)
                if not all(isinstance(item, np.memmap) for item in values):
                    raise RuntimeError(
                        f"Strict loading forbids materializing archive '{file_path.name}'"
                    )
            return payload
    if ext == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return np.asarray(payload)
    if ext == ".safetensors":
        if _load_safetensors is None:
            raise ImportError("Reading .safetensors requires the 'safetensors' package; install it to enable this loader.")
        tensors = _load_safetensors(str(file_path))
        if len(tensors) == 1:
            return next(iter(tensors.values()))
        return tensors
    if ext in {".txt", ".md"}:
        return _build_bow_tensor(file_path)
    if ext in (".tsv", ".csv"):
        coords: List[Tuple[int, ...]] = []
        max_vals: List[int] = []
        with file_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.replace(",", "\t").split("\t")
                values = tuple(int(x) for x in parts if x != "")
                if not values:
                    continue
                if coords and len(values) != len(coords[0]):
                    raise ValueError(f"Inconsistent arity in facts file '{path}'")
                coords.append(values)
                if len(max_vals) < len(values):
                    max_vals.extend([-1] * (len(values) - len(max_vals)))
                for axis, val in enumerate(values):
                    if val > max_vals[axis]:
                        max_vals[axis] = val
        if not coords:
            arity = len(max_vals) if max_vals else 0
            coords_arr = np.zeros((0, arity), dtype=np.int64)
            shape = tuple((max_val + 1) if max_val >= 0 else 0 for max_val in max_vals)
            return SparseBoolTensor(coords_arr, shape)
        shape = tuple(max_val + 1 for max_val in max_vals)
        coords_arr = np.array(coords, dtype=np.int64)
        return SparseBoolTensor(coords_arr, shape)
    raise ValueError(f"Unsupported file type: {path}")

def write_tensor_to_file(path: Union[str, Path], arr):
    target = Path(path)
    ext = target.suffix.lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = arr.matrix if isinstance(arr, BagOfWordsTensor) else arr
    if ext == ".npz":
        np.savez(target, payload)
        return
    if ext == ".npy":
        np.save(target, payload)
        return
    if ext == ".jsonl":
        with target.open("w", encoding="utf-8") as handle:
            if _is_topk_payload(payload):
                handle.write(json.dumps({"schema": "fuse.topk", "version": 1}, ensure_ascii=False) + "\n")
                for row in payload:
                    normalised: List[Dict[str, float]] = []
                    for pair in row:
                        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                            raise ValueError("Top-k rows must contain (index, value) pairs")
                        normalised.append(
                            {"index": int(pair[0]), "value": float(pair[1])}
                        )
                    handle.write(json.dumps({"topk": normalised}, ensure_ascii=False) + "\n")
            else:
                tensor = np.asarray(payload)
                handle.write(
                    json.dumps(
                        {"schema": "fuse.tensor", "version": 1, "shape": list(tensor.shape)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.write(json.dumps({"value": tensor.tolist()}, ensure_ascii=False) + "\n")
        return
    if ext in (".tsv",".csv"):
        # write indices of 1s for Boolean tensors
        if isinstance(arr, SparseBoolTensor):
            idxs = arr.coords
        else:
            tensor = np.asarray(payload)
            idxs = np.argwhere(tensor != 0)
        with target.open("w", encoding="utf-8") as f:
            for r in idxs:
                f.write("\t".join(str(int(x)) for x in r)+"\n")
        return
    raise ValueError(f"Unsupported sink type: {target}")
