import numpy as np
import pytest

from fuse.core.evaluator_numpy import ExecutionConfig
from fuse.core.program import Program

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@st.composite
def shapes_and_axes(draw):
    # 1-3 axes, sizes 1..4
    axes = draw(st.lists(st.sampled_from(list("ijk")), min_size=1, max_size=3, unique=True))
    sizes = {ax: draw(st.integers(min_value=1, max_value=4)) for ax in axes}
    return axes, sizes


@given(shapes_and_axes())
def test_elementwise_broadcast_sum(draw_input):
    axes, sizes = draw_input
    # Build program: z[axes...] = sum over extra axis of x[...] + y[...] where one term lacks last axis
    idx = ",".join(axes)
    if len(axes) == 1:
        return  # trivial
    head = axes[:-1]
    idx_head = ",".join(head)
    src = f"z[{idx_head}] = reduce(sum, {axes[-1]}) (x[{idx}] + y[{idx_head}]); export z;"
    prog = Program(src, parser="v2")
    runner = prog.compile(backend="numpy", config=ExecutionConfig())
    shape_full = tuple(sizes[a] for a in axes)
    shape_head = tuple(sizes[a] for a in head)
    x = np.random.randn(*shape_full).astype(np.float32)
    y = np.random.randn(*shape_head).astype(np.float32)
    out = runner.run(inputs={"x": x, "y": y})["z"]
    assert out.shape == shape_head
