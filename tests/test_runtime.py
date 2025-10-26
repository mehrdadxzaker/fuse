import numpy as np
import pytest

from fuse import (
    ExecutionConfig,
    InMemoryWeightStore,
    Program,
    RuntimePolicies,
)
from tests._torch_utils import require_torch


def _compile(program_src: str, *, backend: str = "numpy", config=None, policies=None):
    prog = Program(program_src)
    return prog.compile(backend=backend, config=config, policies=policies)


def test_runner_explain_json_includes_index_summary():
    prog = Program(
        """
Z[i] = const([1,2,3])
export Z
""".strip()
    )
    runner = prog.compile(backend="numpy")
    runner()
    payload = runner.explain(json=True)
    eq_entries = [entry for entry in payload["logs"] if entry["kind"] == "equation"]
    assert eq_entries
    summary = eq_entries[0]["equation"]["index_summary"]
    assert summary["lhs"] == ["i"]


def test_fixpoint_reachability_numpy():
    prog_src = """
Reach[i,j] = Base[i,j]
Reach[i,k] = Reach[i,j] Base[j,k]
export Reach
"""
    base = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    policies = RuntimePolicies(weight_store=InMemoryWeightStore({"Base": base}))
    cfg = ExecutionConfig(mode="fixpoint", max_iters=8)
    runner = _compile(prog_src, backend="numpy", config=cfg, policies=policies)
    result = runner()["Reach"]
    assert result.shape == (3, 3)
    expected = np.array(
        [
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    assert np.array_equal((result > 0).astype(np.int32), expected)


def test_axis_ops_numpy_concat_and_reductions():
    prog_src = """
Flat[i,f] = concat(Input[i,a,b], axis="b")
RowMax[i,a] = max(Input[i,a,b], axis="b")
RowAvg[i,b] = avg(Input[i,a,b], axis="a")
export Flat
export RowMax
export RowAvg
"""
    tensor = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    policies = RuntimePolicies(weight_store=InMemoryWeightStore({"Input": tensor}))
    runner = _compile(prog_src, backend="numpy", policies=policies)
    out = runner()
    flat = out["Flat"]
    row_max = out["RowMax"]
    row_avg = out["RowAvg"]

    assert flat.shape == (2, 4)
    np.testing.assert_allclose(flat[0], np.array([1, 2, 3, 4]))
    np.testing.assert_allclose(flat[1], np.array([5, 6, 7, 8]))

    expected_max = tensor.max(axis=2)
    np.testing.assert_allclose(row_max, expected_max)

    np.testing.assert_allclose(row_avg, tensor.mean(axis=1))


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_projection_reduction_sugar_max_avg(backend):
    if backend == "torch":
        require_torch()
    if backend == "jax":
        pytest.importorskip("jax")
    prog_src = """
MaxOut[i] max= Input[i,j] Weight[j]
AvgOut[i] avg= Input[i,j] Weight[j]
SumOut[i] = Input[i,j] Weight[j]
export MaxOut
export AvgOut
export SumOut
"""
    tensor = np.array(
        [
            [1.0, -2.0, 3.0],
            [-4.0, 5.0, -6.0],
        ],
        dtype=np.float32,
    )
    weight = np.array([0.5, -1.5, 2.0], dtype=np.float32)

    policies = RuntimePolicies(
        weight_store=InMemoryWeightStore({"Input": tensor, "Weight": weight})
    )
    runner = _compile(prog_src, backend=backend, policies=policies)
    outputs = runner()

    def _to_numpy(value):
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return np.array(value.detach().cpu().tolist())
        return np.asarray(value)

    max_out = _to_numpy(outputs["MaxOut"])
    avg_out = _to_numpy(outputs["AvgOut"])
    sum_out = _to_numpy(outputs["SumOut"])

    product = tensor * weight
    expected_max = product.max(axis=1)
    expected_avg = product.mean(axis=1)
    expected_sum = product.sum(axis=1)

    np.testing.assert_allclose(max_out, expected_max)
    np.testing.assert_allclose(avg_out, expected_avg)
    np.testing.assert_allclose(sum_out, expected_sum)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_index_slices_and_case_features(backend):
    if backend == "torch":
        require_torch()
    if backend == "jax":
        pytest.importorskip("jax")
    prog_src = """
Angles[p, d] = InputAngles[p, d]
AnglesSlice[p, 0:2] = Angles[p, 0:2]
AnglesShiftFwd[p, d] = Angles[p, d+1]
AnglesShiftBack[p, d] = Angles[p, d-1]
PosEnc[p, d] = case(
    Even(d), sin(Angles[p, d]),
    Odd(d), cos(Angles[p, d-1])
)
export AnglesSlice
export AnglesShiftFwd
export AnglesShiftBack
export PosEnc
"""
    angles = np.array(
        [
            [0.0, 0.5, 1.0, 1.5],
            [0.2, 0.7, 1.2, 1.7],
            [0.4, 0.9, 1.4, 1.9],
        ],
        dtype=np.float32,
    )
    policies = RuntimePolicies(
        weight_store=InMemoryWeightStore({"InputAngles": angles})
    )
    runner = _compile(prog_src, backend=backend, policies=policies)
    outputs = runner()

    def _to_numpy(value):
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return np.array(value.detach().cpu().tolist())
        return np.asarray(value)

    slice_out = _to_numpy(outputs["AnglesSlice"])
    shift_fwd = _to_numpy(outputs["AnglesShiftFwd"])
    shift_back = _to_numpy(outputs["AnglesShiftBack"])
    pos_enc = _to_numpy(outputs["PosEnc"])

    expected_slice = angles[:, :2]
    expected_shift_fwd = np.zeros_like(angles)
    expected_shift_fwd[:, :-1] = angles[:, 1:]
    expected_shift_back = np.zeros_like(angles)
    expected_shift_back[:, 1:] = angles[:, :-1]
    even_mask = (np.arange(angles.shape[1]) % 2 == 0).astype(np.float32)[None, :]
    odd_mask = 1.0 - even_mask
    expected_pos_enc = even_mask * np.sin(angles) + odd_mask * np.cos(expected_shift_back)

    np.testing.assert_allclose(slice_out, expected_slice)
    np.testing.assert_allclose(shift_fwd, expected_shift_fwd)
    np.testing.assert_allclose(shift_back, expected_shift_back)
    np.testing.assert_allclose(pos_enc, expected_pos_enc, rtol=1e-5, atol=1e-6)
