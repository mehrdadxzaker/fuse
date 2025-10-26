from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

__all__ = [
    "SparseBoolTensor",
    "BagOfWordsTensor",
    "step",
    "relu",
    "sig",
    "gelu",
    "gelu_grad",
    "softmax",
    "masked_softmax",
    "softmax_grad",
    "tucker_dense",
    "lnorm",
    "layernorm",
    "rope",
    "causal_mask",
    "const",
    "attention",
    "concat",
    "reduce_max",
    "reduce_mean",
    "topk",
    "read_tensor_from_file",
    "write_tensor_to_file",
]

NDArrayAny: TypeAlias = NDArray[Any]


class SparseBoolTensor:
    coords: NDArrayAny
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    def __init__(self, coords: NDArrayAny, shape: Sequence[int]) -> None: ...
    def to_dense(self) -> NDArrayAny: ...
    def nnz(self) -> int: ...
    def __array__(self, dtype: Any | None = None) -> None: ...


class BagOfWordsTensor:
    matrix: NDArrayAny
    vocab: Mapping[str, int]
    vocab_path: Path | None

    def __init__(
        self, matrix: NDArrayAny, vocab: Mapping[str, int], vocab_path: Path | None = ...
    ) -> None: ...
    def __array__(self, dtype: Any | None = None) -> NDArrayAny: ...
    def to_dense(self) -> NDArrayAny: ...


def step(x: Any, /) -> NDArrayAny: ...
def relu(x: Any, /) -> NDArrayAny: ...
def sig(x: Any, T: Any | None = None, *, zero_tol: float = ...) -> NDArrayAny: ...
def gelu(x: Any, /) -> NDArrayAny: ...
def gelu_grad(x: Any, /) -> NDArrayAny: ...
def softmax(x: Any, axis: int = ...) -> NDArrayAny: ...
def masked_softmax(
    x: Any, mask: Any | None = None, axis: int = ..., fill_value: Any | None = None
) -> NDArrayAny: ...
def softmax_grad(y: Any, grad: Any, axis: int = ...) -> NDArrayAny: ...
def tucker_dense(
    value: Any,
    rank: Sequence[int] | None = None,
    threshold: float = ...,
    *,
    rng: np.random.Generator | None = None,
) -> NDArrayAny: ...
def lnorm(x: Any, axis: int = ..., eps: float = ...) -> NDArrayAny: ...
def layernorm(x: Any, axis: int = ..., eps: float = ...) -> NDArrayAny: ...
def rope(x: Any, pos_axis_value: Any) -> NDArrayAny: ...
def causal_mask(L: int) -> NDArrayAny: ...
def const(val: Any) -> NDArrayAny: ...
def attention(
    query: Any,
    key: Any,
    value: Any,
    mask: Any | None = None,
    scale: Any | None = None,
    causal: bool = ...,
) -> NDArrayAny: ...
def concat(*arrays: Any, axis: int | None = None) -> NDArrayAny: ...
def reduce_max(x: Any, axis: Any = ..., keepdims: bool = ...) -> NDArrayAny: ...
def reduce_mean(x: Any, axis: Any = ..., keepdims: bool = ...) -> NDArrayAny: ...
def topk(
    arr: Any, k: int = ..., *, rng: np.random.Generator | None = None
) -> list[list[tuple[int, float]]]: ...
def read_tensor_from_file(
    path: str | Path,
    *,
    strict: bool = ...,
    mmap_threshold_bytes: int | None = None,
) -> NDArrayAny | BagOfWordsTensor | SparseBoolTensor: ...
def write_tensor_to_file(path: str | Path, arr: Any) -> None: ...
