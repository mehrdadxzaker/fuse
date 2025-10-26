from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..core.evaluator_numpy import ExecutionConfig
from ..core.program import Program
from ..core.policies import RuntimePolicies
from ..inference.grad_builder import generate_gradient_program


def _grad_name(tensor: str) -> str:
    return f"Grad_{tensor}"


def gradients_for_program(
    program: Program,
    *,
    seeds: Dict[str, str],
    grad_tensors: Sequence[str],
    backend: str = "numpy",
    config: Optional[ExecutionConfig] = None,
    policies: Optional[RuntimePolicies] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Evaluate per-example gradients for the provided program by generating
    a symbolic gradient program (BPTS-style).

    Parameters
    ----------
    program:
        Compiled representation of the example-specific Fuse program.
    seeds:
        Mapping of tensor name to Fuse expression used to seed its gradient
        (e.g., ``{\"Loss\": \"const(1.0)\"}``).
    grad_tensors:
        Iterable of tensor names whose gradients should be returned.
    backend:
        Backend used to execute the gradient program (defaults to "numpy").
    config:
        Optional execution config; defaults to ``ExecutionConfig(mode=\"single\")``.
    policies:
        Optional runtime policies passed to the compiled gradient program.

    Returns
    -------
    grads:
        Dictionary mapping tensor names to ``np.ndarray`` gradients.
    raw_outputs:
        Complete output dictionary from the gradient program execution.
    """
    grad_program = generate_gradient_program(
        program,
        seeds=seeds,
        export_grads=grad_tensors,
    )
    runner = grad_program.program.compile(
        backend=backend,
        config=config or ExecutionConfig(mode="single"),
        policies=policies,
    )
    outputs = runner()
    grads: Dict[str, np.ndarray] = {}
    for tensor in grad_tensors:
        name = _grad_name(tensor)
        grad_value = outputs.get(name)
        if grad_value is None:
            raise KeyError(f"Gradient output '{name}' missing from gradient program")
        grads[tensor] = np.asarray(grad_value)
    return grads, outputs
