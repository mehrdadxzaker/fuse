import numpy as np
import pytest

from fuse import ExecutionConfig, Program

jax = pytest.importorskip("jax", reason="JAX backend not available")


def test_jax_xla_callable_lazily_builds(tmp_path):
    prog = Program(
        """
Scores[b,j] = Input[b,j]
Prob[b,j] = softmax(Scores[b,j])
export Prob
""".strip()
    )
    cfg = ExecutionConfig(device="cpu")
    runner = prog.compile(backend="jax", config=cfg)

    # Lazily built: private cache starts empty.
    assert runner._xla_callable is None  # type: ignore[attr-defined]

    inputs = {"Input": np.asarray([[0.1, 0.9], [0.2, -0.4]], dtype=np.float32)}
    expected = runner.run(inputs=inputs)["Prob"]

    compiled_fn = runner.xla_callable
    result = compiled_fn(inputs)
    assert isinstance(result, tuple)
    np.testing.assert_allclose(np.asarray(result[0]), np.asarray(expected), rtol=1e-5, atol=1e-6)

    # Subsequent access reuses cached callable.
    assert runner.xla_callable is compiled_fn
