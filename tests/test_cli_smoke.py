from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_cli_run_smoke(tmp_path: Path):
    examples = Path(__file__).resolve().parent.parent / "examples"
    program = examples / "04_mlp.fuse"
    assert program.exists(), "expected example program to exist"

    # Run from the examples directory so file sources/sinks resolve
    env = os.environ.copy()
    proc = subprocess.run(
        [
            "python",
            "-m",
            "fuse",
            "run",
            str(program),
            "--backend",
            "numpy",
            "--out",
            str(tmp_path / "out.npy"),
        ],
        cwd=str(examples),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"CLI failed: {proc.returncode}\n{proc.stdout}\n{proc.stderr}")

