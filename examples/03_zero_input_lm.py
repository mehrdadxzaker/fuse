import os

from fuse import Program
from fuse import torch as fuse_torch

os.chdir(os.path.dirname(__file__))
open("03_zero_input_lm.fuse", "w").write(
    '\\\n# Zero-input: program declares its own sources and sinks\n\nX[p, t]       = "data/prompt.txt"    # text -> Boolean (pos×token) matrix\nEmb[t, d]     = "ckpt/emb.npy"\nWO[t, d]      = "ckpt/lm_head.npy"\n\nEmbX[p, d]    = X(p, t) Emb[t, d]\nLogits[p, t]  = WO[t, d] EmbX[p, d]\nY[p, t.]      = softmax(Logits[p, t])\n\n"runs/topk.jsonl" = topk(Y[p, t], k=5)\nexport Y\n'
)
prog = Program(open("03_zero_input_lm.fuse").read())
runner = fuse_torch.compile(prog, device="auto")
out = runner()
print("Top-k written to runs/topk.jsonl; Y available via export.")
print("Y shape:", out["Y"].shape if out.get("Y") is not None else None)
print(runner.explain())
