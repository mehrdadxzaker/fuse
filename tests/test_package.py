import json
import zipfile

import numpy as np

from fuse import Program, build_package


def test_build_package_creates_artifacts(tmp_path):
    eqs = """
export Out
Out[i,j] = A[i,k] B[k,j]
"""
    prog = Program(eqs)
    package_path = tmp_path / "simple.fusepkg"

    inputs = {
        "A": np.eye(4, dtype=np.float32),
        "B": np.eye(4, dtype=np.float32),
    }

    build_package(
        prog,
        package_path=package_path,
        backend="numpy",
        warm_run=True,
        inputs=inputs,
    )

    assert package_path.exists()
    with zipfile.ZipFile(package_path, "r") as zf:
        files = set(zf.namelist())
        assert "source/program.fuse" in files
        assert "ir/program.json" in files
        assert "plans/plan.json" in files
        assert "profile.json" in files
        explain = zf.read("explain.md").decode("utf-8")
        assert "Cold run" in explain
        assert "Normalized Einsums" in explain
        profile = json.loads(zf.read("profile.json"))
        assert profile["runs"], "profile must contain run data"
        cold = profile["runs"][0]
        assert cold["kind"] == "cold"
        assert cold["kernels"], "profile must include kernel stats"
        kernel = cold["kernels"][0]
        assert "einsum" in kernel
