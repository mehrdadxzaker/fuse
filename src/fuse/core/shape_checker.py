from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from .exceptions import ShapeError
from .ir import Equation, FuncCall, IndexFunction, ProgramIR, TensorRef, Term


@dataclass
class AxisUsage:
    count: int = 0
    sources: Set[str] = field(default_factory=set)

    def merge(self, other: "AxisUsage") -> None:
        self.count += other.count
        self.sources.update(other.sources)

    def add(self, label: str) -> None:
        self.count += 1
        self.sources.add(label)


REDUCTION_FUNCS = {
    "avg",
    "mean",
    "max",
    "amax",
    "reduce_max",
    "reduce_mean",
    "sum",
}


def validate_program_shapes(ir: ProgramIR) -> None:
    # Group equations by LHS name to detect accumulation patterns
    lhs_groups: Dict[str, List[Equation]] = {}
    for eq in ir.equations:
        if eq.is_source or eq.is_sink or eq.rhs is None:
            continue
        lhs_groups.setdefault(eq.lhs.name, []).append(eq)

    # Track which equations are part of accumulation patterns
    accumulation_eqs: Set[int] = set()
    for _, eqs in lhs_groups.items():
        if len(eqs) > 1:
            # Multiple equations with same LHS = accumulation pattern
            for eq in eqs:
                accumulation_eqs.add(id(eq))

    for eq in ir.equations:
        if eq.is_source or eq.is_sink or eq.rhs is None:
            continue
        _validate_equation(eq, is_accumulation=id(eq) in accumulation_eqs)


def _validate_equation(eq: Equation, is_accumulation: bool = False) -> None:
    usage = _collect_axis_usage(eq.rhs, eq)
    if isinstance(eq.rhs, FuncCall):
        usage = _adjust_usage_for_func(eq, eq.rhs, usage)
    if not usage:
        # Scalars or pure broadcasts are allowed.
        return

    lhs_axes_original: List[str] = list(dict.fromkeys(eq.lhs.indices))
    tensor_names = _collect_tensor_names(eq.rhs)
    lhs_axes: List[str] = []
    for axis in lhs_axes_original:
        if axis in eq.lhs.rolling and eq.lhs.name not in tensor_names:
            continue
        lhs_axes.append(axis)
    lhs_axis_set: Set[str] = set(lhs_axes)
    dotted_axes: Set[str] = set(eq.lhs.dotted_axes)

    # In accumulation patterns, RHS can have subset of LHS axes (broadcast)
    if is_accumulation:
        # Don't require all LHS axes to be present in RHS
        missing_axes = []
    else:
        missing_axes = sorted(axis for axis in lhs_axis_set if axis not in usage)

    # Check for stray axes (in RHS but not LHS)
    stray_axes = []

    for axis, info in usage.items():
        if axis in lhs_axis_set:
            continue

        # If axis appears multiple times, it's being contracted (not stray)
        if info.count > 1:
            continue

        # For projection operations on compiler temporaries (like __red0),
        # extra axes are expected as they're being explicitly reduced
        is_reduce_temp = eq.lhs.name.startswith("__red") and eq.projection in ("sum", "max", "mean")
        if is_reduce_temp:
            # DEBUG: Uncomment to trace
            # print(f"DEBUG: Skipping axis {axis} for reduce temp {eq.lhs.name}")
            continue

        # Single source, single appearance = potentially stray
        if len(info.sources) <= 1:
            stray_axes.append(axis)

    stray_axes = sorted(stray_axes)

    # TensorRef-only assignments behave like broadcast; project-only axes are not allowed.
    rhs_is_tensor = isinstance(eq.rhs, TensorRef)
    if rhs_is_tensor:
        # For reduce temporaries, extra axes are being reduced and are allowed
        is_reduce_temp = eq.lhs.name.startswith("__red") and eq.projection in ("sum", "max", "mean")
        if not is_reduce_temp:
            stray_axes = sorted(axis for axis in usage if axis not in lhs_axis_set)

    if not missing_axes and not stray_axes:
        return

    join_axes = sorted(usage.keys())
    projected_axes = sorted(axis for axis in usage if axis not in lhs_axis_set)

    message = (
        f"{eq.lhs.name}: RHS joins on {{{_format_axes(join_axes, dotted_axes)}}}, "
        f"projects {{{_format_axes(projected_axes, dotted_axes)}}}, "
        f"but LHS expects {{{_format_axes(lhs_axes, dotted_axes)}}}"
    )

    details: List[str] = []
    if missing_axes:
        missing_descriptions = []
        for axis in missing_axes:
            label = _format_axis(axis, dotted_axes)
            location = f"line {eq.line}" if eq.line is not None else "LHS"
            missing_descriptions.append(f"{label} expected on {eq.lhs.name} ({location})")
        details.append("missing indices " + "; ".join(missing_descriptions))
    if stray_axes:
        stray_descriptions = []
        for axis in stray_axes:
            label = _format_axis(axis, dotted_axes)
            sources = ", ".join(sorted(usage[axis].sources)) or "RHS"
            stray_descriptions.append(f"{label} from {sources}")
        details.append("unmatched RHS indices " + "; ".join(stray_descriptions))

    if details:
        message += ". " + ". ".join(details) + "."

    raise ShapeError(
        message,
        line=eq.line,
        column=eq.column,
        line_text=eq.source,
    )


def _collect_axis_usage(expr, eq: Equation) -> Dict[str, AxisUsage]:
    if isinstance(expr, Term):
        usage: Dict[str, AxisUsage] = {}
        for factor in expr.factors:
            usage = _merge_usage(usage, _collect_axis_usage(factor, eq))
        return usage

    if isinstance(expr, TensorRef):
        usage: Dict[str, AxisUsage] = {}
        rolling_axes = set(expr.rolling.keys())
        for axis in expr.indices:
            if axis in rolling_axes:
                continue
            info = usage.setdefault(axis, AxisUsage())
            info.add(_format_source(expr.name, eq))
        return usage

    if isinstance(expr, FuncCall):
        usage: Dict[str, AxisUsage] = {}
        args: Iterable = ()
        if isinstance(expr.arg, tuple):
            args = expr.arg
        elif expr.arg is not None:
            args = (expr.arg,)
        for arg in args:
            usage = _merge_usage(usage, _collect_axis_usage(arg, eq))
        lowered_name = expr.name.lower()
        if lowered_name in REDUCTION_FUNCS:
            axes_to_remove = _axes_removed_by_reduction(expr, usage)
            for axis in axes_to_remove:
                usage.pop(axis, None)
        return usage

    if isinstance(expr, IndexFunction):
        usage: Dict[str, AxisUsage] = {}
        info = usage.setdefault(expr.axis, AxisUsage())
        info.add(_format_source(expr.name, eq, suffix=" (index)"))
        return usage

    if isinstance(expr, (list, tuple)):
        usage: Dict[str, AxisUsage] = {}
        for item in expr:
            usage = _merge_usage(usage, _collect_axis_usage(item, eq))
        return usage

    return {}


def _adjust_usage_for_func(
    eq: Equation,
    fn: FuncCall,
    usage: Dict[str, AxisUsage],
) -> Dict[str, AxisUsage]:
    name = fn.name.lower()
    if name == "concat":
        lhs_axes = [axis for axis in eq.lhs.indices if axis not in eq.lhs.rolling]
        lhs_set = set(lhs_axes)
        for axis in list(usage.keys()):
            if axis not in lhs_set:
                usage.pop(axis, None)
        for axis in lhs_axes:
            info = usage.setdefault(axis, AxisUsage())
            info.add("concat")
        return usage
    return usage


def _collect_tensor_names(expr) -> Set[str]:
    if isinstance(expr, TensorRef):
        return {expr.name}
    if isinstance(expr, Term):
        names: Set[str] = set()
        for factor in expr.factors:
            names.update(_collect_tensor_names(factor))
        return names
    if isinstance(expr, FuncCall):
        names: Set[str] = set()
        args: Iterable = ()
        if isinstance(expr.arg, tuple):
            args = expr.arg
        elif expr.arg is not None:
            args = (expr.arg,)
        for arg in args:
            names.update(_collect_tensor_names(arg))
        return names
    if isinstance(expr, (list, tuple)):
        names: Set[str] = set()
        for item in expr:
            names.update(_collect_tensor_names(item))
        return names
    return set()


def _merge_usage(
    base: Dict[str, AxisUsage],
    incoming: Dict[str, AxisUsage],
) -> Dict[str, AxisUsage]:
    if not incoming:
        return base
    for axis, info in incoming.items():
        target = base.setdefault(axis, AxisUsage())
        target.merge(info)
    return base


def _axes_removed_by_reduction(fn: FuncCall, usage: Dict[str, AxisUsage]) -> Set[str]:
    axis_spec = fn.kwargs.get("axis")
    if axis_spec is None:
        return set()
    if isinstance(axis_spec, str):
        return {axis_spec}
    if isinstance(axis_spec, (list, tuple)):
        axes: Set[str] = set()
        for item in axis_spec:
            if isinstance(item, str):
                axes.add(item)
        return axes
    # Unsupported axis spec (e.g., int) – fall back to removing nothing.
    return set()


def _format_axes(axes: Iterable[str], dotted_axes: Set[str]) -> str:
    axes_list = list(dict.fromkeys(axes))
    if not axes_list:
        return "∅"
    formatted = [_format_axis(axis, dotted_axes) for axis in sorted(axes_list)]
    return ", ".join(formatted)


def _format_axis(axis: str, dotted_axes: Set[str]) -> str:
    return f"{axis}′" if axis in dotted_axes else axis


def _format_source(name: str, eq: Optional[Equation], *, suffix: str = "") -> str:
    if eq is not None and eq.line is not None:
        return f"{name}{suffix} (line {eq.line})"
    return f"{name}{suffix}"
