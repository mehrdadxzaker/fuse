from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

try:  # pragma: no cover - numpy optional at type-check time
    import numpy as _np
except Exception:  # pragma: no cover - consumer may not have numpy when only parsing
    _np = None


@dataclass
class TensorRef:
    name: str
    indices: List[str]  # in order as written
    dotted_axes: List[str] = field(
        default_factory=list
    )  # indices marked with '.' in LHS (for softmax/lnorm)
    rolling: Dict[str, int] = field(default_factory=dict)  # streaming indices with offsets
    index_specs: List["IndexSpec"] = field(default_factory=list)
    is_paren: bool = False


@dataclass
class SliceSpec:
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None


@dataclass
class IndexSpec:
    axis: str
    offset: int = 0
    slice: Optional[SliceSpec] = None


@dataclass
class Term:
    # A product of factors (all TensorRef for now) compiled into einsum
    factors: List[Any]


@dataclass
class FuncCall:
    name: str
    arg: Any  # could be Term or TensorRef or FuncCall
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexFunction:
    name: str
    axis: str


Expr = Any  # Term | FuncCall | TensorRef | IndexFunction


@dataclass
class Equation:
    lhs: TensorRef
    rhs: Expr  # sums handled as multiple equations with same LHS (pre-split)
    projection: str = "sum"  # "sum" | "max" | "mean" for projected indices (syntactic sugar)
    src_file: Optional[str] = None  # if reading from file
    sink_file: Optional[str] = None  # if writing to file
    is_source: bool = False
    is_sink: bool = False
    export: bool = False
    line: Optional[int] = None
    column: Optional[int] = None
    source: Optional[str] = None


@dataclass
class ProgramIR:
    equations: List[Equation]
    exports: List[str] = field(default_factory=list)

    def streaming_axes(self) -> List[str]:
        axes = set()
        for eq in self.equations:
            axes.update(eq.lhs.rolling.keys())
            axes.update(_rolling_axes_in_expr(eq.rhs))
        return sorted(axes)

    def has_streaming(self) -> bool:
        return bool(self.streaming_axes())

    def boolean_tensors(self) -> Set[str]:
        names: Set[str] = set()
        for eq in self.equations:
            if eq.lhs.is_paren:
                names.add(eq.lhs.name)
        return names


def _rolling_axes_in_expr(expr: Any) -> List[str]:
    axes: List[str] = []
    if isinstance(expr, TensorRef):
        axes.extend(expr.rolling.keys())
        return axes
    if isinstance(expr, Term):
        for factor in expr.factors:
            axes.extend(_rolling_axes_in_expr(factor))
        return axes
    if isinstance(expr, FuncCall):
        arg = expr.arg
        if isinstance(arg, tuple):
            for item in arg:
                axes.extend(_rolling_axes_in_expr(item))
        elif arg is not None:
            axes.extend(_rolling_axes_in_expr(arg))
    if isinstance(expr, IndexFunction):
        return []
    return axes


def _collect_object_indices(obj: Any, seen: Dict[str, None]) -> None:
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            _collect_object_indices(item, seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            _collect_object_indices(item, seen)
    elif isinstance(obj, (Term, FuncCall, TensorRef, IndexFunction)):
        _collect_expr_indices(obj, seen)


def _collect_expr_indices(expr: Expr, seen: Dict[str, None]) -> None:
    if isinstance(expr, TensorRef):
        for idx in expr.indices:
            if idx not in seen:
                seen[idx] = None
        for idx in expr.rolling.keys():
            if idx not in seen:
                seen[idx] = None
        for spec in expr.index_specs:
            axis = spec.axis
            if axis not in seen:
                seen[axis] = None
        return
    if isinstance(expr, Term):
        for factor in expr.factors:
            _collect_expr_indices(factor, seen)
        return
    if isinstance(expr, FuncCall):
        arg = expr.arg
        if isinstance(arg, tuple):
            for item in arg:
                _collect_expr_indices(item, seen)
        elif arg is not None:
            _collect_expr_indices(arg, seen)
        for value in expr.kwargs.values():
            _collect_object_indices(value, seen)
        return
    if isinstance(expr, IndexFunction):
        if expr.axis not in seen:
            seen[expr.axis] = None
        return
    _collect_object_indices(expr, seen)


def lhs_indices(eq: Equation) -> List[str]:
    ordered: Dict[str, None] = {}
    for idx in eq.lhs.indices:
        if idx not in ordered:
            ordered[idx] = None
    return list(ordered.keys())


def rhs_indices(eq: Equation) -> List[str]:
    seen: Dict[str, None] = {}
    _collect_expr_indices(eq.rhs, seen)
    return list(seen.keys())


def equation_index_summary(eq: Equation, projected: Sequence[str]) -> Dict[str, List[str]]:
    lhs = lhs_indices(eq)
    rhs = rhs_indices(eq)
    lhs_set = set(lhs)
    rhs_set = set(rhs)
    projected_unique: Dict[str, None] = {}
    for idx in projected:
        key = str(idx)
        if key not in projected_unique:
            projected_unique[key] = None
    projected_list = list(projected_unique.keys())
    introduced = sorted(str(idx) for idx in rhs_set - lhs_set)
    projected_set = set(projected_list)
    dangling = sorted(str(idx) for idx in rhs_set - lhs_set - projected_set)
    return {
        "lhs": lhs,
        "rhs_only": introduced,
        "projected": projected_list,
        "dangling": dangling,
    }


def format_index_summary(summary: Mapping[str, Sequence[str]]) -> str:
    def _format(key: str, label: str) -> str:
        values = summary.get(key, []) or []
        if isinstance(values, (list, tuple, set)):
            items = [str(v) for v in values]
        else:
            items = [str(values)] if values else []
        text = ",".join(items) if items else "-"
        return f"{label}:{text}"

    return " | ".join(
        [
            _format("lhs", "LHS"),
            _format("rhs_only", "RHS+"),
            _format("projected", "proj"),
            _format("dangling", "dang"),
        ]
    )


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if _np is not None:
        if isinstance(value, (_np.integer, _np.floating)):
            return value.item()
        if isinstance(value, _np.ndarray):
            return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    return str(value)
