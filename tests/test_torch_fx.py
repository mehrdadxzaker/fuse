import json

import numpy as np

from fuse import ExecutionConfig, Program
from tests._torch_utils import require_torch


def _softmax_program() -> Program:
    return Program(
        """
Scores[b,j] = Input[b,j]
Soft[b,j] = softmax(Scores[b,j])
export Soft
""".strip()
    )


def test_torch_softmax_matches_numpy_and_fx_annotations():
    torch = require_torch()
    prog = _softmax_program()

    inputs = {"Input": np.array([[0.2, -1.5, 0.3], [0.0, 0.5, -0.1]], dtype=np.float32)}
    numpy_runner = prog.compile(backend="numpy")
    expected = numpy_runner.run(inputs=inputs)["Soft"]

    torch_runner = prog.compile(backend="torch", config=ExecutionConfig(device="cpu"))
    result = torch_runner.run(inputs=inputs)["Soft"].detach().cpu().numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    gm = torch_runner.fx_module
    assert gm is not None
    node_names = [node.name for node in gm.graph.nodes]
    assert any("__float" in name for name in node_names)


def test_torch_topk_sink_matches_numpy(tmp_path):
    require_torch()
    out_path = tmp_path / "topk.jsonl"
    prog = Program(
        f'''
Scores[i,j] = Input[i,j]
"{out_path}" = topk(Scores[i,j], k=2)
'''.strip()
    )
    inputs = {"Input": np.array([[0.1, 0.9, -0.4], [0.2, 0.3, 0.8]], dtype=np.float32)}

    numpy_runner = prog.compile(backend="numpy")
    numpy_runner.run(inputs=inputs)
    expected = out_path.read_text(encoding="utf-8")

    torch_runner = prog.compile(backend="torch", config=ExecutionConfig(device="cpu"))
    torch_runner.run(inputs=inputs)
    actual = out_path.read_text(encoding="utf-8")

    assert actual == expected


def test_fx_cache_includes_fingerprint(tmp_path):
    require_torch()
    cache_dir = tmp_path / "cache"
    prog = _softmax_program()

    prog.compile(
        backend="torch",
        cache_dir=str(cache_dir),
        config=ExecutionConfig(device="cpu"),
    )

    meta_files = list(cache_dir.rglob("*.json"))
    assert meta_files, "expected cache metadata files to be created"
    fingerprint_found = False
    for file in meta_files:
        data = json.loads(file.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {})
        if "cache_fingerprint" in metadata:
            fingerprint_found = True
            break
    assert fingerprint_found, "cache metadata missing expanded fingerprint"
