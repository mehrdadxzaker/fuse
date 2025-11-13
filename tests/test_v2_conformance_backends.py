import numpy as np
import pytest

from fuse.core.evaluator_numpy import ExecutionConfig
from fuse.core.program import Program

PROG_SRC = (
    "param D:int = 16; axis i; axis j; axis d;\n"
    "sim[i,j] = reduce(sum, d) Emb[i,d] * Emb[j,d];\n"  # Explicit reduction over d
    "score[i] = reduce(sum, j) select(mask[i], sim[i,j], 0);\n"
    "export score;"
)


def _inputs(N=8, D=16, *, seed=0):
    rng = np.random.default_rng(seed)
    Emb = rng.normal(size=(N, D)).astype(np.float32)
    mask = rng.integers(0, 2, size=(N,), dtype=np.int8)
    return {"Emb": Emb, "mask": mask}


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_v2_conformance_select_piecewise_fn_inline(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("jax.numpy")

    prog = Program(PROG_SRC, parser="v2")
    cfg = ExecutionConfig(mode="single", device="cpu")
    runner = prog.compile(backend=backend, config=cfg)

    expected = prog.compile(backend="numpy", config=cfg).run(inputs=_inputs())["score"]
    result = runner.run(inputs=_inputs())
    if isinstance(result, dict):
        actual = result["score"]
    else:
        actual = result
    # Convert torch/jax tensors to numpy
    try:
        if hasattr(actual, "detach"):
            actual = actual.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(actual, "__array__") and not isinstance(actual, np.ndarray):
        actual = np.asarray(actual)
    assert np.allclose(expected, actual, atol=1e-5, rtol=1e-5)
