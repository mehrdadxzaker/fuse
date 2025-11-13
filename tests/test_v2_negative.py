import numpy as np
import pytest

from fuse.core.ast_lowering import lower_to_ir
from fuse.core.evaluator_numpy import ExecutionConfig
from fuse.core.parser_expr import parse_program
from fuse.core.program import Program


def test_fn_axis_ambiguity_error():
    src = (
        "fn bad(a[x], b[x]) -> y { y = a[x] * b[x]; }\nz[i,j] = bad(E[i,d], F[i,e]);\nexport z;"
    )
    ast_prog = parse_program(src)
    with pytest.raises(ValueError):
        lower_to_ir(ast_prog)


def test_masked_softmax_non_broadcastable_mask():
    src = "y[i,j] = @softmax(x[i,j], axis=j, mask=m[i]); export y;"
    prog = Program(src, parser="v2")
    runner = prog.compile(backend="numpy", config=ExecutionConfig())
    x = np.ones((4, 5), dtype=np.float32)
    m = np.ones((3,), dtype=np.int8)  # wrong shape
    with pytest.raises(Exception):  # noqa: B017
        runner.run(inputs={"x": x, "m": m})


def test_reduce_op_unsupported():
    src = "y = reduce(prod, i) x[i]; export y;"
    ast_prog = parse_program(src)
    with pytest.raises(NotImplementedError):
        lower_to_ir(ast_prog)
