from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# NOTE: This module defines a lightweight, decoupled AST for the new
# statement + expression grammar. It does not replace the existing IR.
# Callers can parse to this AST first and then lower to the existing
# ProgramIR when ready.


@dataclass
class Node:
    line: Optional[int] = None
    column: Optional[int] = None
    source: Optional[str] = None


# Program structure ------------------------------------------------------------


@dataclass
class Import(Node):
    path: str = ""
    alias: Optional[str] = None


@dataclass
class Param(Node):
    name: str = ""
    type_name: Optional[str] = None
    default: Any = None


@dataclass
class AxisDecl(Node):
    name: str = ""
    value: Any = None


@dataclass
class Const(Node):
    name: str = ""
    value: Any = None


@dataclass
class Let(Node):
    lhs: "Lhs" = None  # type: ignore
    value: Any = None


@dataclass
class FnDef(Node):
    name: str = ""
    params: List["Lhs"] = field(default_factory=list)
    returns: Optional["Lhs"] = None
    body: "Block" = None  # type: ignore


@dataclass
class Program(Node):
    imports: List[Import] = field(default_factory=list)
    params: List[Param] = field(default_factory=list)
    axes: List[AxisDecl] = field(default_factory=list)
    consts: List[Const] = field(default_factory=list)
    fns: List[FnDef] = field(default_factory=list)
    statements: List[Node] = field(default_factory=list)  # Equation | Let | Block
    exports: List[str] = field(default_factory=list)


# Expressions -----------------------------------------------------------------


Expr = Any  # Union of literals, BinOp, Call/FnCall, Reduce, Select, Piecewise, etc.


@dataclass
class BinOp(Node):
    left: Expr = None  # type: ignore
    op: str = ""
    right: Expr = None  # type: ignore


@dataclass
class UnaryOp(Node):
    op: str = ""
    value: Expr = None  # type: ignore


@dataclass
class Call(Node):
    name: str = ""
    args: List[Expr] = field(default_factory=list)
    kwargs: Dict[str, Expr] = field(default_factory=dict)


@dataclass
class FnCall(Call):
    pass


@dataclass
class Macro(Node):
    name: str = ""
    args: List[Expr] = field(default_factory=list)
    kwargs: Dict[str, Expr] = field(default_factory=dict)


@dataclass
class Reduce(Node):
    op: str = ""
    value: Expr = None  # type: ignore
    axes: List[str] = field(default_factory=list)


@dataclass
class Select(Node):
    condition: Expr = None  # type: ignore
    then: Expr = None  # type: ignore
    otherwise: Expr = None  # type: ignore


@dataclass
class Piecewise(Node):
    branches: List[Tuple[Expr, Expr]] = field(default_factory=list)  # (cond, value)
    default: Optional[Expr] = None


# Statements ------------------------------------------------------------------


@dataclass
class Lhs(Node):
    name: str = ""
    dims: List[str] = field(default_factory=list)


@dataclass
class Equation(Node):
    lhs: Lhs = None  # type: ignore
    rhs: Expr = None  # type: ignore
    guard: Optional[Expr] = None


@dataclass
class Block(Node):
    statements: List[Node] = field(default_factory=list)


@dataclass
class Tensor(Node):
    name: str = ""
    dims: List[str] = field(
        default_factory=list
    )  # raw index tokens, may include dotted/offset/slice


# Pretty printer ---------------------------------------------------------------


_PREC = {
    "||": 1,
    "&&": 2,
    "+": 3,
    "-": 3,
    "*": 4,
    "/": 4,
}


def _needs_paren(parent_op: Optional[str], child: Expr, is_right: bool) -> bool:
    if not isinstance(child, BinOp):
        return False
    if parent_op is None:
        return False
    p = _PREC.get(parent_op, 0)
    c = _PREC.get(child.op, 0)
    if c < p:
        return True
    if c == p and is_right and child.op in {"-", "/"}:
        return True
    return False


def _pp_expr(expr: Expr, parent_op: Optional[str] = None, side: str = "") -> str:
    if isinstance(expr, BinOp):
        left = _pp_expr(expr.left, expr.op, "L")
        right = _pp_expr(expr.right, expr.op, "R")
        left_s = f"({left})" if _needs_paren(expr.op, expr.left, False) else left
        right_s = f"({right})" if _needs_paren(expr.op, expr.right, True) else right
        return f"{left_s} {expr.op} {right_s}"
    if isinstance(expr, UnaryOp):
        return f"{expr.op}{_pp_expr(expr.value, None)}"
    if isinstance(expr, Call) or isinstance(expr, FnCall):
        args = ", ".join(_pp_expr(a, None) for a in expr.args)
        if getattr(expr, "kwargs", None):
            kw = ", ".join(f"{k}={_pp_expr(v)}" for k, v in expr.kwargs.items())
            args = ", ".join([x for x in [args, kw] if x])
        return f"{expr.name}({args})"
    if isinstance(expr, Reduce):
        axes = ", ".join(expr.axes)
        return f"reduce({expr.op}, {axes}) {_pp_expr(expr.value)}"
    if isinstance(expr, Select):
        return f"{_pp_expr(expr.condition)} ? {_pp_expr(expr.then)} : {_pp_expr(expr.otherwise)}"
    if isinstance(expr, Piecewise):
        parts = [f"({_pp_expr(c)}) ? {_pp_expr(v)}" for c, v in expr.branches]
        if expr.default is not None:
            parts.append(f": {_pp_expr(expr.default)}")
        return " ".join(parts)
    if isinstance(expr, Tensor):
        return f"{expr.name}[{', '.join(expr.dims)}]" if expr.dims else expr.name
    try:
        from .ir import TensorRef as IRTensorRef  # type: ignore
    except Exception:  # pragma: no cover - defensive import
        IRTensorRef = None  # type: ignore
    if IRTensorRef is not None and isinstance(expr, IRTensorRef):  # type: ignore
        idxs = getattr(expr, "indices", []) or []
        dotted = set(getattr(expr, "dotted_axes", []) or [])
        raw = [f"{ax}." if ax in dotted else ax for ax in idxs]
        return f"{expr.name}[{', '.join(raw)}]" if raw else expr.name
    # Literals and bare identifiers are printed as-is
    return str(expr)


def pretty_print_old_style(prog: Program) -> str:
    """Render a Program to the legacy line-based style.

    - Equations print as: `lhs = expr;`
    - Let/Const print as: `let x = expr;` / `const x = expr;`
    - Other constructs are emitted with a simple, readable form.
    """
    lines: List[str] = []

    for imp in prog.imports:
        if imp.alias:
            lines.append(f'import "{imp.path}" as {imp.alias};')
        else:
            lines.append(f'import "{imp.path}";')

    for p in prog.params:
        type_part = f": {p.type_name}" if p.type_name else ""
        if p.default is not None:
            lines.append(f"param {p.name}{type_part} = {_pp_expr(p.default)};")
        else:
            lines.append(f"param {p.name}{type_part};")

    for ax in prog.axes:
        if ax.value is not None:
            lines.append(f"axis {ax.name} = {_pp_expr(ax.value)};")
        else:
            lines.append(f"axis {ax.name};")

    for c in prog.consts:
        lines.append(f"const {c.name} = {_pp_expr(c.value)};")

    def _pp_lhs(lhs: Lhs) -> str:
        return f"{lhs.name}[{', '.join(lhs.dims)}]" if lhs.dims else lhs.name

    for st in prog.statements:
        if isinstance(st, Equation):
            if st.guard is not None:
                lines.append(f"{_pp_lhs(st.lhs)} when {_pp_expr(st.guard)} = {_pp_expr(st.rhs)};")
            else:
                lines.append(f"{_pp_lhs(st.lhs)} = {_pp_expr(st.rhs)};")
        elif isinstance(st, Let):
            lines.append(f"let {_pp_lhs(st.lhs)} = {_pp_expr(st.value)};")
        elif isinstance(st, Block):
            lines.append("{")
            for inner in st.statements:
                if isinstance(inner, Equation):
                    if inner.guard is not None:
                        lines.append(
                            f"  {_pp_lhs(inner.lhs)} when {_pp_expr(inner.guard)} = {_pp_expr(inner.rhs)};"
                        )
                    else:
                        lines.append(f"  {_pp_lhs(inner.lhs)} = {_pp_expr(inner.rhs)};")
                elif isinstance(inner, Let):
                    lines.append(f"  let {_pp_lhs(inner.lhs)} = {_pp_expr(inner.value)};")
            lines.append("}")

    for fn in prog.fns:

        def _pp_lhs(lhs: Lhs) -> str:
            return f"{lhs.name}[{', '.join(lhs.dims)}]" if lhs.dims else lhs.name

        header = f"fn {fn.name}({', '.join(_pp_lhs(p) for p in fn.params)})"
        if fn.returns is not None:
            header += f" -> {_pp_lhs(fn.returns)}"
        lines.append(header + " {")
        for inner in fn.body.statements:
            if isinstance(inner, Equation):
                if inner.guard is not None:
                    lines.append(
                        f"  {_pp_lhs(inner.lhs)} when {_pp_expr(inner.guard)} = {_pp_expr(inner.rhs)};"
                    )
                else:
                    lines.append(f"  {_pp_lhs(inner.lhs)} = {_pp_expr(inner.rhs)};")
            elif isinstance(inner, Let):
                lines.append(f"  let {_pp_lhs(inner.lhs)} = {_pp_expr(inner.value)};")
        lines.append("}")

    return "\n".join(lines)
    for name in prog.exports:
        lines.append(f"export {name};")
