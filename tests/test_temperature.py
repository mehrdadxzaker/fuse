import numpy as np

from fuse import ExecutionConfig, Program
from fuse.core.temperature import ConstantSchedule, PiecewiseSchedule, coerce_temperature_value


def _build_sig_program() -> Program:
    return Program(
        """
X[i] = const([-2.0, 2.0])
Y[i] = sig(X[i])
export Y
"""
    )


def test_sig_defaults_to_deductive_step():
    prog = _build_sig_program()
    runner = prog.compile(backend="numpy")
    outputs = runner()
    assert np.array_equal(outputs["Y"], np.array([0, 1], dtype=np.int8))
    explanation = runner.explain()
    assert "T=0" in explanation


def test_sig_temperature_schedule_numpy():
    prog = _build_sig_program()
    runner = prog.compile(backend="numpy")

    analog_cfg = ExecutionConfig(temperatures={"Y": 0.5})
    outputs = runner.run(config=analog_cfg)
    scores = outputs["Y"]
    assert scores.shape == (2,)
    assert scores[0] < 0.15
    assert scores[1] > 0.85

    manifest = runner.temperature_manifest()
    assert manifest == {"Y": {"type": "constant", "temperature": 0.5}}
    explanation = runner.explain()
    assert "T=0.5" in explanation


def test_sig_temperature_default_key():
    prog = _build_sig_program()
    runner = prog.compile(backend="numpy")

    cfg = ExecutionConfig(temperatures={"*": ConstantSchedule(0.25)})
    outputs = runner.run(config=cfg)
    scores = outputs["Y"]
    assert scores[0] < 0.2
    assert scores[1] > 0.6
    explanation = runner.explain()
    assert "T=0.25" in explanation


def test_piecewise_schedule_manifest_and_values():
    schedule = PiecewiseSchedule([(0, 0.0), (3, 0.75)])
    assert schedule(0) == 0.0
    assert schedule(2) == 0.0
    assert schedule(3) == 0.75
    assert schedule(10) == 0.75
    assert schedule.manifest() == {
        "type": "piecewise",
        "points": [(0, 0.0), (3, 0.75)],
    }


def test_coerce_temperature_value_scalar_inputs():
    assert coerce_temperature_value(1) == 1.0
    assert coerce_temperature_value(1.5) == 1.5
    assert coerce_temperature_value(np.array([0.25])) == 0.25
