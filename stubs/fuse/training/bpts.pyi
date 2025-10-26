from typing import Sequence

import numpy as np

from ..core.evaluator_numpy import ExecutionConfig as ExecutionConfig
from ..core.policies import RuntimePolicies as RuntimePolicies
from ..core.program import Program as Program
from ..inference.grad_builder import generate_gradient_program as generate_gradient_program

def gradients_for_program(program: Program, *, seeds: dict[str, str], grad_tensors: Sequence[str], backend: str = 'numpy', config: ExecutionConfig | None = None, policies: RuntimePolicies | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: ...
