from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np
import pytest

from fuse import ExecutionConfig, Program, generate_gradient_program

try:
    from tests._torch_utils import require_torch
except ModuleNotFoundError:  # pragma: no cover - direct invocation fallback
    import importlib.util

    _UTILS_SPEC = importlib.util.spec_from_file_location("_torch_utils", Path(__file__).with_name("_torch_utils.py"))
    _UTILS_MODULE = importlib.util.module_from_spec(_UTILS_SPEC)
    assert _UTILS_SPEC.loader is not None
    _UTILS_SPEC.loader.exec_module(_UTILS_MODULE)
    require_torch = _UTILS_MODULE.require_torch


def _run_loss(program_src: str, *, backend: str) -> float:
    program = Program(program_src)
    runner = program.compile(backend=backend, config=ExecutionConfig(mode="single"))
    outputs = runner()
    return float(outputs["Loss"])


def _finite_difference(
    program_builder: Callable[[np.ndarray], str],
    base_value: np.ndarray,
    *,
    backend: str,
    epsilon: float = 1e-3,
) -> np.ndarray:
    grads = np.zeros_like(base_value, dtype=np.float32)
    for idx in np.ndindex(base_value.shape):
        positive = base_value.copy()
        negative = base_value.copy()
        positive[idx] += epsilon
        negative[idx] -= epsilon
        loss_plus = _run_loss(program_builder(positive), backend=backend)
        loss_minus = _run_loss(program_builder(negative), backend=backend)
        grads[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return grads


def _build_gelu_program(weights: np.ndarray, *, inputs: np.ndarray, targets: np.ndarray) -> str:
    return f"""
Input[b,d]   = const({inputs.tolist()})
Weights[h,d] = const({weights.tolist()})
Target[b,h]  = const({targets.tolist()})

Hidden[b,h]  = Weights[h,d] Input[b,d]
Activated[b,h] = gelu(Hidden[b,h])
Loss          = Activated[b,h] Target[b,h]
export Loss
"""


def _build_softmax_program(logits: np.ndarray, *, targets: np.ndarray) -> str:
    return f"""
Logits[b,d]   = const({logits.tolist()})
Target[b,d]   = const({targets.tolist()})

Probs[b,d.]   = softmax(Logits[b,d], axis="d")
Loss          = Probs[b,d] Target[b,d]
export Loss
"""


def _gradient_outputs(program_src: str, *, backend: str, export_grads: Iterable[str]) -> Dict[str, np.ndarray]:
    program = Program(program_src)
    grad_program = generate_gradient_program(
        program,
        seeds={"Loss": "const(1.0)"},
        export_grads=export_grads,
    )
    runner = grad_program.program.compile(backend=backend, config=ExecutionConfig(mode="single"))
    outputs = runner()
    return {name: np.asarray(outputs[f"Grad_{name}"], dtype=np.float32) for name in export_grads}


@pytest.mark.parametrize(
    "backend",
    [
        "numpy",
        pytest.param("torch", marks=pytest.mark.slow),
    ],
)
def test_gelu_gradient_matches_finite_difference(backend):
    if backend == "torch":
        require_torch()

    rng = np.random.default_rng(7)
    inputs = rng.standard_normal((2, 3), dtype=np.float32)
    weights = rng.standard_normal((3, 3), dtype=np.float32)
    targets = rng.standard_normal((2, 3), dtype=np.float32)

    base_program = _build_gelu_program(weights, inputs=inputs, targets=targets)
    grads = _gradient_outputs(base_program, backend=backend, export_grads=["Weights"])
    analytic = grads["Weights"]

    fd = _finite_difference(
        lambda w: _build_gelu_program(w, inputs=inputs, targets=targets),
        weights,
        backend=backend,
        epsilon=1e-3,
    )

    np.testing.assert_allclose(analytic, fd, rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize(
    "backend",
    [
        "numpy",
        pytest.param("torch", marks=pytest.mark.slow),
    ],
)
def test_softmax_gradient_matches_finite_difference(backend):
    if backend == "torch":
        require_torch()

    rng = np.random.default_rng(11)
    logits = rng.standard_normal((2, 4), dtype=np.float32)
    targets = rng.random((2, 4), dtype=np.float32)

    base_program = _build_softmax_program(logits, targets=targets)
    grads = _gradient_outputs(base_program, backend=backend, export_grads=["Logits"])
    analytic = grads["Logits"]

    fd = _finite_difference(
        lambda L: _build_softmax_program(L, targets=targets),
        logits,
        backend=backend,
        epsilon=2e-3,
    )

    np.testing.assert_allclose(analytic, fd, rtol=5e-3, atol=5e-4)
