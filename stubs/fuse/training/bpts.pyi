from typing import Any, Sequence

from numpy.typing import NDArray

from ..core.evaluator_numpy import ExecutionConfig as ExecutionConfig
from ..core.policies import RuntimePolicies as RuntimePolicies
from ..core.program import Program as Program
from ..inference.grad_builder import generate_gradient_program as generate_gradient_program

def gradients_for_program(program: Program, *, seeds: dict[str, str], grad_tensors: Sequence[str], backend: str = 'numpy', config: ExecutionConfig | None = None, policies: RuntimePolicies | None = None) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]]]: ...
