import numpy as np

from fuse import Program


STREAMING_PROGRAM = """
Input[i]          = const([0.2,-0.1])
Weight[i,j]       = const([[0.5,-0.3],[0.1,0.4]])
Pre[i]            = Weight[i,j] Hidden[j,*t]
Sum[i]            = Pre[i] + Input[i]
Hidden[i,*t+1]    = relu(Sum[i])
export Hidden
"""


def test_streaming_hidden_updates_over_time():
    program = Program(STREAMING_PROGRAM.strip())
    runner = program.compile(backend="numpy")

    # Seed initial hidden state at time t
    out_step1 = runner(inputs={"Hidden": np.zeros(2, dtype=np.float32)})
    assert np.allclose(out_step1["Hidden"], np.array([0.2, 0.0], dtype=np.float32))

    # Next invocation should advance virtual time without explicit axes
    out_step2 = runner()
    expected_step2 = np.array([0.3, 0.0], dtype=np.float32)
    assert np.allclose(out_step2["Hidden"], expected_step2)
