import pytest

from fuse import ExecutionConfig, Program


def _sample_program() -> Program:
    return Program(
        """
Value[i] = const([1.0, 0.0])
export Value
""".strip()
    )


def test_execution_config_normalization_handles_precision_device_zero_copy():
    cfg = ExecutionConfig(precision="BF16", device="GPU:1", zero_copy=0).normalized()
    assert cfg.precision == "bf16"
    assert cfg.device == "cuda:1"
    assert cfg.zero_copy is False


def test_numpy_compile_rejects_non_cpu_device():
    prog = _sample_program()
    cfg = ExecutionConfig(device="cuda")
    with pytest.raises(ValueError, match="NumPy backend only supports CPU execution"):
        prog.compile(backend="numpy", config=cfg)


def test_numpy_compile_rejects_non_fp32_precision():
    prog = _sample_program()
    cfg = ExecutionConfig(precision="bf16")
    with pytest.raises(ValueError, match="NumPy backend only supports fp32 precision"):
        prog.compile(backend="numpy", config=cfg)
