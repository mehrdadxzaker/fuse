from dataclasses import dataclass
from typing import Iterable

from _typeshed import Incomplete

from ..core.ir import Equation as Equation
from ..core.ir import FuncCall as FuncCall
from ..core.ir import ProgramIR as ProgramIR
from ..core.ir import SliceSpec as SliceSpec
from ..core.ir import TensorRef as TensorRef
from ..core.ir import Term as Term
from ..core.program import Program as Program

@dataclass
class GradientProgram:
    program: Program
    gradient_names: set[str]

class GradientBuilder:
    program: Incomplete
    ir: ProgramIR
    def __init__(self, program: Program) -> None: ...
    def build(self, seeds: dict[str, str], export_grads: Iterable[str] | None = None) -> GradientProgram: ...

def generate_gradient_program(program: Program, *, seeds: dict[str, str], export_grads: Iterable[str] | None = None) -> GradientProgram: ...
