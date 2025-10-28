from dataclasses import dataclass
from typing import Any, Sequence

from numpy.typing import NDArray

from ..core.evaluator_numpy import ExecutionConfig as ExecutionConfig
from ..core.policies import InMemoryWeightStore as InMemoryWeightStore
from ..core.policies import RuntimePolicies as RuntimePolicies
from ..core.program import Program as Program

@dataclass(frozen=True)
class Variable:
    name: str
    cardinality: int

@dataclass(frozen=True)
class Factor:
    name: str
    scope: tuple[str, ...]
    table: NDArray[Any]
    def normalized(self) -> Factor: ...

@dataclass
class TreeProgram:
    program: Program
    weights: dict[str, NDArray[Any]]
    query_vars: tuple[str, ...]
    variables: dict[str, Variable]
    def compile(self, *, config: ExecutionConfig | None = None) -> Program: ...
    @property
    def source(self) -> str: ...

class TreeFactorGraph:
    variables: dict[str, Variable]
    factors: dict[str, Factor]
    def __init__(self, variables: Sequence[Variable], factors: Sequence[Factor]) -> None: ...
    def build_program(self, *, query_vars: Sequence[str], evidence: dict[str, Any] | None = None) -> TreeProgram: ...

def conditional_probability(outputs: dict[str, Any], query_vars: Sequence[str], assignment: dict[str, int]) -> float: ...
def brute_force_joint(factors: Sequence[Factor], variables: Sequence[Variable], evidence: dict[str, int] | None = None) -> NDArray[Any]: ...
