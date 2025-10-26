import json
import os
from pathlib import Path
from typing import Optional

from fuse import ExecutionConfig, Program, torch as fuse_torch


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PROGRAM_PATH = ROOT / "02_rules_and_embeddings.fuse"

PROGRAM_SOURCE = """# Deductive rule + analogical reasoning via embeddings and temperature T

# Facts as Boolean relations
Sister(x, y)   = "data/sister.tsv"
Parent(y, z)   = "data/parent.tsv"

# Deductive rule (Boolean join + step)
Aunt[x, z]     = step(Sister[x, y] Parent[y, z])

# Embeddings & analogical score
Emb[obj, d]    = "ckpt/object_emb.npy"
EmbAunt[i, j]  = Aunt(x, y) Emb[x, i] Emb[y, j]
Score[a, b]    = sig( EmbAunt[i, j] Emb[a, i] Emb[b, j] )

"runs/aunts.tsv"    = Aunt[x, z]
"runs/sally.jsonl"  = topk(Score[a, b], k=3)
export Aunt
"""

PROGRAM_PATH.write_text(PROGRAM_SOURCE)

program = Program(PROGRAM_PATH.read_text())
runner = fuse_torch.compile(program, device="auto")

RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def save_manifest(label: str) -> Path:
    manifest = {
        "label": label,
        "temperature": runner.temperature_manifest(),
        "explain": runner.explain().splitlines(),
    }
    manifest_path = RUNS_DIR / f"temperature_{label}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def snapshot_topk(label: str) -> Optional[Path]:
    base = RUNS_DIR / "sally.jsonl"
    if not base.exists():
        return None
    dest = RUNS_DIR / f"sally_{label}.jsonl"
    dest.write_text(base.read_text())
    return dest


def run_case(cfg: ExecutionConfig, label: str) -> None:
    runner.run(config=cfg)
    manifest_path = save_manifest(label)
    topk_path = snapshot_topk(label)
    message = f"{label.title()} reasoning manifest saved to {manifest_path.name}"
    if topk_path is not None:
        message += f"; top-k copied to {topk_path.name}"
    print(message)
    print(runner.explain())


if __name__ == "__main__":
    run_case(ExecutionConfig(temperatures={"Score": 0.0}), "deductive")
    run_case(ExecutionConfig(temperatures={"Score": 0.5}), "analogical")
    print("Aunt tensor saved to runs/aunts.tsv")
