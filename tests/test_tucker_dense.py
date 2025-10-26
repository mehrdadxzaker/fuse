import numpy as np
import pytest

from fuse import Program, torch as fuse_torch, jax as fuse_jax
from fuse.core.builtins import SparseBoolTensor


def _make_sparse_relation(n: int, m: int, rank: int, density: float = 0.1, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(n, rank))
    right = rng.normal(size=(m, rank))
    scores = left @ right.T
    threshold = np.quantile(scores, 1.0 - density)
    return (scores >= threshold).astype(np.int8)


def _compile_program(rank: int, threshold: float = 0.5) -> Program:
    eqs = f"""
export Dense
Dense[i,j] = tucker_dense(Rel[i,j], rank={rank}, threshold={threshold})
"""
    return Program(eqs)


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def test_tucker_dense_rank_quality_improves():
    relation = _make_sparse_relation(64, 48, rank=6, density=0.08, seed=11)
    prog_low = _compile_program(rank=2)
    prog_high = _compile_program(rank=12)

    runner_low = prog_low.compile(backend="numpy")
    runner_high = prog_high.compile(backend="numpy")

    low_out = runner_low(inputs={"Rel": relation})["Dense"]
    high_out = runner_high(inputs={"Rel": relation})["Dense"]

    low_err = _hamming(low_out, relation)
    high_err = _hamming(high_out, relation)

    assert high_err <= low_err


def test_tucker_dense_accepts_sparse_tensor():
    relation = _make_sparse_relation(40, 40, rank=5, density=0.12, seed=21)
    coords = np.argwhere(relation > 0)
    sparse = SparseBoolTensor(coords, relation.shape)
    prog = _compile_program(rank=8)
    runner = prog.compile(backend="numpy")
    outputs = runner(inputs={"Rel": sparse})
    approx = outputs["Dense"]
    # Ensure output is binary
    assert set(np.unique(approx)) <= {0, 1}
    assert approx.shape == relation.shape


def test_tucker_dense_torch_matches_numpy():
    try:
        import torch
        torch.tensor([0], dtype=torch.float32).detach().cpu().numpy()
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"torch unavailable: {exc}")

    relation = _make_sparse_relation(32, 32, rank=4, density=0.15, seed=7)
    prog = _compile_program(rank=10)

    numpy_runner = prog.compile(backend="numpy")
    torch_runner = fuse_torch.compile(prog, device="cpu")

    numpy_out = numpy_runner(inputs={"Rel": relation})["Dense"]
    torch_out = torch_runner(inputs={"Rel": relation})["Dense"]

    torch_arr = (
        torch_out.detach().cpu().numpy()
        if hasattr(torch_out, "detach")
        else np.asarray(torch_out)
    )
    np.testing.assert_array_equal(torch_arr, numpy_out)


def test_tucker_dense_jax_matches_numpy():
    pytest.importorskip("jax")

    relation = _make_sparse_relation(24, 30, rank=3, density=0.2, seed=5)
    prog = _compile_program(rank=8)

    numpy_runner = prog.compile(backend="numpy")
    jax_runner = fuse_jax.compile(prog)

    numpy_out = numpy_runner(inputs={"Rel": relation})["Dense"]
    jax_out = jax_runner(inputs={"Rel": relation})["Dense"]

    jax_arr = np.asarray(jax_out)
    np.testing.assert_array_equal(jax_arr, numpy_out)
