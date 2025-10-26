import numpy as np
import pytest

from fuse import Program, torch as fuse_torch, jax as fuse_jax


def _build_program_and_inputs():
    eqs = """
export Prob
export Norm
export Attn
Scale = const(0.5)
Prob[p,q] = masked_softmax(Logits[p,q], mask=Mask[p,q])
Norm[p,d] = layernorm(X[p,d])
Attn[p,h] = attention(Q[p,d], K[q,d], V[q,h], mask=Mask[p,q], scale=Scale)
"""
    prog = Program(eqs)
    rng = np.random.default_rng(1234)
    logits = rng.normal(size=(2, 3)).astype(np.float32)
    mask = rng.random(size=(2, 3)) > 0.25
    mask[0, 0] = True
    mask[1, 1] = True
    x = rng.normal(size=(2, 4)).astype(np.float32)
    q = rng.normal(size=(2, 4)).astype(np.float32)
    k = rng.normal(size=(3, 4)).astype(np.float32)
    v = rng.normal(size=(3, 5)).astype(np.float32)
    base_inputs = {
        "Logits": logits,
        "Mask": mask,
        "X": x,
        "Q": q,
        "K": k,
        "V": v,
    }
    return prog, base_inputs


def _clone_inputs(base_inputs):
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in base_inputs.items()}


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "__array__"):
        value = np.asarray(value)
    return value


def test_numpy_attention_reference():
    prog, base_inputs = _build_program_and_inputs()
    runner = prog.compile(backend="numpy")
    outputs = runner(inputs=_clone_inputs(base_inputs))
    assert set(outputs.keys()) == {"Prob", "Norm", "Attn"}
    prob = outputs["Prob"]
    assert np.allclose(prob.sum(axis=-1), np.ones(prob.shape[0]))


def test_torch_fastpaths_match_numpy():
    try:
        import torch
        torch.tensor([0.0]).detach().cpu().numpy()
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"torch unavailable: {exc}")
    prog, base_inputs = _build_program_and_inputs()
    numpy_runner = prog.compile(backend="numpy")
    numpy_out = numpy_runner(inputs=_clone_inputs(base_inputs))

    torch_runner = fuse_torch.compile(prog, device="cpu")
    torch_out = torch_runner(inputs=_clone_inputs(base_inputs))

    for name in ("Prob", "Norm", "Attn"):
        lhs = _to_numpy(torch_out[name])
        rhs = np.asarray(numpy_out[name])
        np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-5)


def test_jax_fastpaths_match_numpy():
    pytest.importorskip("jax")
    prog, base_inputs = _build_program_and_inputs()
    numpy_runner = prog.compile(backend="numpy")
    numpy_out = numpy_runner(inputs=_clone_inputs(base_inputs))

    jax_runner = fuse_jax.compile(prog)
    jax_out = jax_runner(inputs=_clone_inputs(base_inputs))

    for name in ("Prob", "Norm", "Attn"):
        lhs = _to_numpy(jax_out[name])
        rhs = np.asarray(numpy_out[name])
        np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-5)
