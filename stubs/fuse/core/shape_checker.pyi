from dataclasses import dataclass, field

from _typeshed import Incomplete

from .exceptions import ShapeError as ShapeError
from .ir import Equation as Equation
from .ir import FuncCall as FuncCall
from .ir import IndexFunction as IndexFunction
from .ir import ProgramIR as ProgramIR
from .ir import TensorRef as TensorRef
from .ir import Term as Term

@dataclass
class AxisUsage:
    count: int = ...
    sources: set[str] = field(default_factory=set)
    def merge(self, other: AxisUsage) -> None: ...
    def add(self, label: str) -> None: ...

REDUCTION_FUNCS: Incomplete

def validate_program_shapes(ir: ProgramIR) -> None: ...
