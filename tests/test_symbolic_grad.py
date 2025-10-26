import numpy as np

from fuse import (
    ExecutionConfig,
    Program,
    generate_gradient_program,
)
from tests._torch_utils import require_torch

torch = require_torch()
import torch.nn.functional as F  # noqa: E402


def _run_program(program_src: str, export_grads):
    program = Program(program_src)
    grad_prog = generate_gradient_program(
        program,
        seeds={"Loss": "const(1.0)"},
        export_grads=export_grads,
    )
    runner = grad_prog.program.compile(backend="numpy", config=ExecutionConfig(mode="single"))
    return runner()


def _gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def test_mlp_symbolic_grad_matches_autodiff():
    program_src = """
Input[b,d]      = const([[0.8,0.2,0.5], [0.1,0.6,0.4]])
W1[h,d]         = const([[0.3,-0.2,0.5], [0.7,0.1,-0.4]])
W2[o,h]         = const([[0.2,0.6], [-0.5,0.3]])
Target[b,o]     = const([[0.4,0.6], [0.3,0.7]])

HiddenLinear[b,h] = W1[h,d] Input[b,d]
    Activation[b,h]   = gelu(HiddenLinear[b,h])
Logits[b,o]       = W2[o,h] Activation[b,h]
Probs[b,o]        = softmax(Logits[b,o], axis="o")
Loss              = Probs[b,o] Target[b,o]
export Loss
"""
    outputs = _run_program(program_src, export_grads=["W1", "W2"])

    # Torch autograd reference
    input_np = np.array([[0.8, 0.2, 0.5], [0.1, 0.6, 0.4]], dtype=np.float32)
    target_np = np.array([[0.4, 0.6], [0.3, 0.7]], dtype=np.float32)
    w1_np = np.array([[0.3, -0.2, 0.5], [0.7, 0.1, -0.4]], dtype=np.float32)
    w2_np = np.array([[0.2, 0.6], [-0.5, 0.3]], dtype=np.float32)

    inp = torch.tensor(input_np.tolist(), dtype=torch.float32)
    targ = torch.tensor(target_np.tolist(), dtype=torch.float32)
    w1 = torch.tensor(w1_np.tolist(), dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor(w2_np.tolist(), dtype=torch.float32, requires_grad=True)

    hidden_linear = torch.matmul(inp, w1.T)
    activation = _gelu_tanh(hidden_linear)
    logits = torch.matmul(activation, w2.T)
    probs = torch.softmax(logits, dim=1)
    loss = torch.sum(probs * targ)
    loss.backward()

    w1_grad = np.array(w1.grad.detach().cpu().to(torch.float32).tolist(), dtype=np.float32)
    w2_grad = np.array(w2.grad.detach().cpu().to(torch.float32).tolist(), dtype=np.float32)

    np.testing.assert_allclose(outputs["Grad_W1"], w1_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(outputs["Grad_W2"], w2_grad, rtol=1e-5, atol=1e-6)


def test_attention_symbolic_grad_matches_autodiff():
    rng = np.random.default_rng(0)
    X = rng.random((2, 3), dtype=np.float32)
    WQ = rng.random((4, 3), dtype=np.float32)
    WK = rng.random((4, 3), dtype=np.float32)
    WV = rng.random((5, 3), dtype=np.float32)
    Mask = rng.random((2, 2), dtype=np.float32)
    Target = rng.random((2, 5), dtype=np.float32)
    inv_sqrt = np.float32(0.35355339)

    program_src = f"""
X[p,d]       = const({X.tolist()})
WQ[dk,d]     = const({WQ.tolist()})
WK[dk,d]     = const({WK.tolist()})
WV[dv,d]     = const({WV.tolist()})
Mask[p,p']   = const({Mask.tolist()})
InvSqrtDk    = const({float(inv_sqrt)})
Target[p,dv] = const({Target.tolist()})

Q[p,dk]      = WQ[dk,d] X[p,d]
K[p,dk]      = WK[dk,d] X[p,d]
V[p,dv]      = WV[dv,d] X[p,d]
Score[p,p']  = Q[p,dk] K[p',dk] InvSqrtDk
Score[p,p']  = Mask[p,p']
Comp[p,p'.]  = softmax(Score[p,p'], axis="p'")
Attn[p,dv]   = Comp[p,p'] V[p',dv]
Loss         = Attn[p,dv] Target[p,dv]
export Loss
"""
    outputs = _run_program(program_src, export_grads=["WQ", "WK", "WV"])

    x_t = torch.tensor(X.tolist(), dtype=torch.float32)
    wq_t = torch.tensor(WQ.tolist(), dtype=torch.float32, requires_grad=True)
    wk_t = torch.tensor(WK.tolist(), dtype=torch.float32, requires_grad=True)
    wv_t = torch.tensor(WV.tolist(), dtype=torch.float32, requires_grad=True)
    mask_t = torch.tensor(Mask.tolist(), dtype=torch.float32)
    target_t = torch.tensor(Target.tolist(), dtype=torch.float32)
    inv = torch.tensor(float(inv_sqrt), dtype=torch.float32)

    q_t = torch.matmul(x_t, wq_t.T)
    k_t = torch.matmul(x_t, wk_t.T)
    v_t = torch.matmul(x_t, wv_t.T)
    score_t = torch.matmul(q_t, k_t.T) * inv + mask_t
    comp_t = torch.softmax(score_t, dim=1)
    attn_t = torch.matmul(comp_t, v_t)
    loss_t = torch.sum(attn_t * target_t)
    loss_t.backward()

    grad_wq = np.array(wq_t.grad.detach().cpu().to(torch.float32).tolist(), dtype=np.float32)
    grad_wk = np.array(wk_t.grad.detach().cpu().to(torch.float32).tolist(), dtype=np.float32)
    grad_wv = np.array(wv_t.grad.detach().cpu().to(torch.float32).tolist(), dtype=np.float32)

    np.testing.assert_allclose(outputs["Grad_WQ"], grad_wq, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(outputs["Grad_WK"], grad_wk, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(outputs["Grad_WV"], grad_wv, rtol=1e-5, atol=1e-6)
