import os

from fuse import Program
from fuse import torch as fuse_torch

os.chdir(os.path.dirname(__file__))
PROGRAM_SRC = r"""
# Masked attention in ~3 equations

# Sources
X[p, d]       = "ckpt/emb.npy"          # pretend this is an activation matrix (P×D)
WQ[dk, d]     = "ckpt/wq.npy"
WK[dk, d]     = "ckpt/wk.npy"
WV[dv, d]     = "ckpt/wv.npy"
Causal[p, p'] = causal_mask(32)
NEG           = const(-10000.0)

# Q, K, V
Q[p, dk]      = WQ[dk, d] X[p, d]
K[p, dk]      = WK[dk, d] X[p, d]
V[p, dv]      = WV[dv, d] X[p, d]

# Scaled dot-product attention with causal mask
# (Use a precomputed inverse sqrt for dk=8 in this toy; 1/sqrt(8) ≈ 0.353553…)
InvSqrtDk     = const(0.35355339059)
Score[p, p']  = (Q[p, dk] K[p', dk]) InvSqrtDk + (1 - Causal[p, p']) NEG
Comp[p, p'.]  = softmax(Score[p, p'])
Attn[p, dv]   = Comp[p, p'] V[p', dv]

"runs/attn.npz" = Attn[p, dv]
export Attn
""".strip()

with open("01_attention_block.fuse", "w", encoding="utf-8") as handle:
    handle.write(PROGRAM_SRC)

with open("01_attention_block.fuse", "r", encoding="utf-8") as handle:
    prog = Program(handle.read())
runner = fuse_torch.compile(prog, device="auto")
out = runner()
attn = out["Attn"]
print("Attn shape:", None if attn is None else attn.shape)
print(runner.explain())
