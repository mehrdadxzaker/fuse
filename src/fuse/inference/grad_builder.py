from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.ir import (
    Equation,
    FuncCall,
    ProgramIR,
    SliceSpec,
    TensorRef,
    Term,
)
from ..core.program import Program


@dataclass
class GradientProgram:
    program: Program
    gradient_names: Set[str]


class GradientBuilder:
    def __init__(self, program: Program):
        self.program = program
        self.ir: ProgramIR = program.ir
        if self.ir.has_streaming():
            raise ValueError("Gradient generation does not yet support streaming programs")
        self._tensor_indices: Dict[str, Tuple[str, ...]] = {}
        for eq in self.ir.equations:
            if eq.is_source or eq.is_sink:
                continue
            if eq.lhs.name not in self._tensor_indices:
                self._tensor_indices[eq.lhs.name] = tuple(eq.lhs.indices)

    # Public API -------------------------------------------------------------
    def build(
        self,
        seeds: Dict[str, str],
        export_grads: Optional[Iterable[str]] = None,
    ) -> GradientProgram:
        gradient_lines: List[str] = []
        exports: Set[str] = set(self.program.ir.exports)
        grad_exports: Set[str] = set()

        # Seed gradients (e.g., Grad_Loss = const(1.0))
        for tensor, expr in seeds.items():
            indices = self._tensor_indices.get(tensor, ())
            grad_ref = self._format_grad_ref(TensorRef(name=tensor, indices=list(indices)))
            gradient_lines.append(f"{grad_ref} = {expr}")
            grad_exports.add(self._grad_name(tensor))

        # Traverse equations in reverse to emit gradient contributions
        for eq in reversed(self.ir.equations):
            if eq.is_source or eq.is_sink or eq.rhs is None:
                continue
            contribs = self._gradient_for_equation(eq)
            gradient_lines.extend(contribs)
            for line in contribs:
                target = line.split("=", 1)[0].strip()
                grad_name = target.split("[", 1)[0]
                grad_exports.add(grad_name)

        if export_grads is not None:
            grad_exports = {self._grad_name(name) for name in export_grads}

        # Build final program text: original source plus gradient equations and exports
        lines = list(line.rstrip() for line in self.program.src.splitlines())
        lines.append("")
        lines.append("# Auto-generated gradient equations")
        lines.extend(gradient_lines)
        for grad in sorted(grad_exports):
            if grad not in exports:
                lines.append(f"export {grad}")

        gradient_src = "\n".join(lines)
        grad_program = Program(gradient_src)
        return GradientProgram(program=grad_program, gradient_names=grad_exports)

    # Internal helpers -------------------------------------------------------
    def _grad_name(self, name: str) -> str:
        return f"Grad_{name}"

    def _format_indices(self, indices: Sequence[str]) -> str:
        if not indices:
            return ""
        return "[" + ", ".join(indices) + "]"

    def _format_tensor_ref(self, ref: TensorRef, override_name: Optional[str] = None) -> str:
        name = override_name or ref.name
        if not ref.indices:
            return name
        idx_parts: List[str] = []
        spec_map = {spec.axis: spec for spec in getattr(ref, "index_specs", [])}
        for idx in ref.indices:
            part = idx
            spec = spec_map.get(idx)
            if spec is not None:
                if spec.slice is not None:
                    part = self._format_slice(idx, spec.slice)
                elif spec.offset:
                    offset = spec.offset
                    if offset > 0:
                        part = f"{idx}+{offset}"
                    else:
                        part = f"{idx}{offset}"
            idx_parts.append(part)
        return f"{name}[{', '.join(idx_parts)}]"

    @staticmethod
    def _format_slice(idx: str, sl: SliceSpec) -> str:
        start = "" if sl.start is None else str(sl.start)
        stop = "" if sl.stop is None else str(sl.stop)
        step = "" if sl.step is None else str(sl.step)
        if step:
            return f"{idx}[{start}:{stop}:{step}]"
        return f"{idx}[{start}:{stop}]"

    def _format_grad_ref(self, ref: TensorRef) -> str:
        grad_ref = TensorRef(name=self._grad_name(ref.name), indices=list(ref.indices))
        return self._format_tensor_ref(grad_ref)

    def _gradient_for_equation(self, eq: Equation) -> List[str]:
        rhs = eq.rhs
        lines: List[str] = []
        if isinstance(rhs, Term):
            lines.extend(self._grad_for_term(eq.lhs, rhs))
        elif isinstance(rhs, TensorRef):
            grad_lhs = self._format_grad_ref(eq.lhs)
            grad_rhs = self._format_grad_ref(rhs)
            lines.append(f"{grad_rhs} = {grad_lhs}")
        elif isinstance(rhs, FuncCall):
            lines.extend(self._grad_for_func(eq.lhs, rhs))
        elif isinstance(rhs, (int, float)):
            # Constant RHS – no gradient contribution
            pass
        else:
            raise ValueError(f"Unsupported RHS type for gradient: {type(rhs).__name__}")
        return lines

    def _grad_for_term(self, lhs: TensorRef, term: Term) -> List[str]:
        grad_lhs = self._format_grad_ref(lhs)
        lines: List[str] = []
        factor_exprs = [self._expr_to_string(factor) for factor in term.factors]
        for idx, factor in enumerate(term.factors):
            if not isinstance(factor, TensorRef):
                continue
            other_terms = [factor_exprs[j] for j in range(len(factor_exprs)) if j != idx]
            expr_terms = [grad_lhs] + other_terms
            target = self._format_grad_ref(factor)
            lines.append(f"{target} = {' '.join(expr_terms)}")
        return lines

    def _grad_for_func(self, lhs: TensorRef, fn: FuncCall) -> List[str]:
        name = fn.name.lower()
        grad_lhs = self._format_grad_ref(lhs)
        if name == "gelu":
            arg = self._ensure_tensor_arg(fn)
            grad_arg = self._format_grad_ref(arg)
            arg_expr = self._format_tensor_ref(arg)
            lines = [f"{grad_arg} = {grad_lhs} gelu_grad({arg_expr})"]
            return lines
        if name == "softmax":
            arg = self._ensure_tensor_arg(fn)
            grad_arg = self._format_grad_ref(arg)
            arg_expr = self._format_tensor_ref(arg)
            axis_expr = self._axis_kwarg(fn, default=-1)
            if axis_expr:
                lines = [
                    f"{grad_arg} = softmax_grad({self._format_tensor_ref(lhs)}, {grad_lhs}, {axis_expr})"
                ]
            else:
                lines = [f"{grad_arg} = softmax_grad({self._format_tensor_ref(lhs)}, {grad_lhs})"]
            return lines
        if name == "const":
            return []
        raise ValueError(f"Gradient for function '{fn.name}' is not implemented")

    def _axis_kwarg(self, fn: FuncCall, default: int) -> Optional[str]:
        axis = fn.kwargs.get("axis")
        if axis is None:
            if default is None:
                return None
            return f"axis={default}"
        if isinstance(axis, str):
            return f'axis="{axis}"'
        return f"axis={int(axis)}"

    @staticmethod
    def _ensure_tensor_arg(fn: FuncCall) -> TensorRef:
        arg = fn.arg
        if isinstance(arg, tuple):
            if not arg:
                raise ValueError(f"{fn.name} requires an argument")
            arg = arg[0]
        if not isinstance(arg, TensorRef):
            raise ValueError(f"{fn.name} gradient expects tensor argument")
        return arg

    def _expr_to_string(self, expr) -> str:
        if isinstance(expr, TensorRef):
            return self._format_tensor_ref(expr)
        if isinstance(expr, FuncCall):
            args: Iterable[str] = []
            if isinstance(expr.arg, tuple):
                args = [self._expr_to_string(item) for item in expr.arg]
            elif expr.arg is not None:
                args = [self._expr_to_string(expr.arg)]
            kw_parts = [f"{key}={value}" for key, value in expr.kwargs.items()]
            inner = ", ".join(list(args) + kw_parts)
            return f"{expr.name}({inner})"
        if isinstance(expr, Term):
            return " ".join(self._expr_to_string(f) for f in expr.factors)
        if isinstance(expr, (int, float)):
            return str(expr)
        raise ValueError(f"Cannot stringify expression of type {type(expr).__name__}")


def generate_gradient_program(
    program: Program,
    *,
    seeds: Dict[str, str],
    export_grads: Optional[Iterable[str]] = None,
) -> GradientProgram:
    builder = GradientBuilder(program)
    return builder.build(seeds=seeds, export_grads=export_grads)
