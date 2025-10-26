import numpy as np
import pytest

from fuse import InMemoryWeightStore, Program, RuntimePolicies
from fuse.interop import (
    from_pytorch,
    to_onnx,
    to_torchscript,
)


def _build_program():
    eqs = """
export Out
Q[t,dk]   = X[t,d] WQ[dk,d]
K[t,dk]   = X[t,d] WK[dk,d]
V[t,dv]   = X[t,d] WV[dv,d]
Scores[t,t'] = Q[t,dk] K[t',dk]
Prob[t,t'.] = softmax(Scores[t,t'])
Out[t,dv] = Prob[t,t'] V[t',dv]
"""
    return Program(eqs)


def _baseline_head(d_model: int, d_k: int, d_v: int):
    torch = pytest.importorskip("torch")

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(d_model, d_k, bias=False)
            self.k_proj = torch.nn.Linear(d_model, d_k, bias=False)
            self.v_proj = torch.nn.Linear(d_model, d_v, bias=False)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            scores = torch.matmul(q, k.transpose(-2, -1))
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)

    return Head()


@pytest.mark.parametrize("seq_len,d_model,d_k,d_v", [(4, 8, 4, 6)])
def test_transformer_head_roundtrip(tmp_path, seq_len, d_model, d_k, d_v):
    torch = pytest.importorskip("torch")

    try:
        import onnx  # noqa: F401
        import onnxruntime as ort  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        ort = None

    head = _baseline_head(d_model, d_k, d_v)
    head.eval()

    state_dict = head.state_dict()
    mapping = {
        "WQ": {"key": "q_proj.weight", "source_axes": ["out", "in"], "target_axes": ["dk", "d"]},
        "WK": {"key": "k_proj.weight", "source_axes": ["out", "in"], "target_axes": ["dk", "d"]},
        "WV": {"key": "v_proj.weight", "source_axes": ["out", "in"], "target_axes": ["dv", "d"]},
    }
    fuse_weights = from_pytorch(state_dict, mapping)
    policies = RuntimePolicies(weight_store=InMemoryWeightStore(fuse_weights))

    prog = _build_program()
    x_np = np.random.default_rng(0).normal(size=(seq_len, d_model)).astype(np.float32)
    x_tensor = torch.tensor(x_np.tolist(), dtype=torch.float32)

    baseline_out = np.asarray(head(x_tensor).detach().cpu().tolist(), dtype=np.float32)

    runner = prog.compile(backend="torch", device="cpu", policies=policies)
    fuse_out_tensor = runner(inputs={"X": x_tensor})["Out"]
    fuse_out = np.asarray(fuse_out_tensor.detach().cpu().tolist(), dtype=np.float32)
    np.testing.assert_allclose(fuse_out, baseline_out, rtol=1e-5, atol=1e-5)

    example_inputs = {"X": x_tensor}
    ts_module = to_torchscript(prog, example_inputs, policies=policies, device="cpu")
    ts_out_tensor = ts_module(x_tensor)[0]
    ts_out = np.asarray(ts_out_tensor.detach().cpu().tolist(), dtype=np.float32)
    np.testing.assert_allclose(ts_out, baseline_out, rtol=1e-5, atol=1e-5)

    if ort is not None:
        onnx_path = tmp_path / "head.onnx"
        to_onnx(prog, example_inputs, policies=policies, device="cpu", file_path=onnx_path)
        assert onnx_path.exists()
        session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {session.get_inputs()[0].name: x_np}
        ort_out = session.run(None, ort_inputs)[0]
        np.testing.assert_allclose(ort_out, baseline_out, rtol=1e-5, atol=1e-5)
