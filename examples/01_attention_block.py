import os
import numpy as np
from fuse import Program, torch as fuse_torch

os.chdir(os.path.dirname(__file__))
eqs = open("01_attention_block.fuse","w"); eqs.write('\\\n# Masked attention in ~3 equations\n\n# Sources\nX[p, d]       = "ckpt/emb.npy"          # pretend this is an activation matrix (P×D)\nWQ[dk, d]     = "ckpt/wq.npy"\nWK[dk, d]     = "ckpt/wk.npy"\nWV[dv, d]     = "ckpt/wv.npy"\nCausal[p, p\'] = causal_mask(32)\nNEG           = const(-10000.0)\n\n# Q, K, V\nQ[p, dk]      = WQ[dk, d] X[p, d]\nK[p, dk]      = WK[dk, d] X[p, d]\nV[p, dv]      = WV[dv, d] X[p, d]\n\n# Scaled dot-product attention with causal mask\n# (Use a precomputed inverse sqrt for dk=8 in this toy; 1/sqrt(8) ≈ 0.353553…)\nInvSqrtDk     = const(0.35355339059)\nScore[p, p\']  = (Q[p, dk] K[p\', dk]) InvSqrtDk + (1 - Causal[p, p\']) NEG\nComp[p, p\'.]  = softmax(Score[p, p\'])\nAttn[p, dv]   = Comp[p, p\'] V[p\', dv]\n\n"runs/attn.npz" = Attn[p, dv]\nexport Attn\n'); eqs.close()

prog = Program(open("01_attention_block.fuse").read())
runner = fuse_torch.compile(prog, device="auto")
out = runner()
attn = out["Attn"]
print("Attn shape:", None if attn is None else attn.shape)
print(runner.explain())
