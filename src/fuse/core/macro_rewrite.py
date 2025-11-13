from __future__ import annotations

from dataclasses import replace
from typing import Any, List

from .ast import (
    Block,
    Call,
    Equation,
    Let,
    Macro,
    Program,
)


class MacroExpander:
    def __init__(self):
        self._counter = 0

    def gensym(self, prefix: str = "__m") -> str:
        s = f"{prefix}{self._counter}"
        self._counter += 1
        return s

    # Public API ---------------------------------------------------------------
    def expand(self, prog: Program) -> Program:
        # Expand in statements and function bodies
        new_stmts: List[Any] = [self._expand_stmt(st) for st in prog.statements]
        new_fns = []
        for fn in prog.fns:
            body = self._expand_stmt(fn.body) if fn.body is not None else None
            new_fns.append(replace(fn, body=body))
        return replace(prog, statements=new_stmts, fns=new_fns)

    # Node rewrites ------------------------------------------------------------
    def _expand_stmt(self, st: Any) -> Any:
        if isinstance(st, Let):
            return replace(st, value=self._expand_expr(st.value))
        if isinstance(st, Equation):
            return replace(
                st,
                rhs=self._expand_expr(st.rhs),
                guard=self._expand_expr(st.guard) if st.guard is not None else None,
            )
        if isinstance(st, Block):
            return replace(st, statements=[self._expand_stmt(x) for x in st.statements])
        return st

    def _expand_expr(self, expr: Any) -> Any:
        if expr is None:
            return None
        if isinstance(expr, Macro):
            return self._expand_macro(expr)
        if isinstance(expr, Call):
            return replace(
                expr,
                args=[self._expand_expr(a) for a in expr.args],
                kwargs={k: self._expand_expr(v) for k, v in expr.kwargs.items()},
            )
        # For structures that contain expressions, add cases as needed
        try:
            from .ast import BinOp, Piecewise, Reduce, Select, UnaryOp
        except Exception:
            BinOp = UnaryOp = Select = Piecewise = Reduce = tuple()  # type: ignore
        if isinstance(expr, BinOp):
            return replace(
                expr, left=self._expand_expr(expr.left), right=self._expand_expr(expr.right)
            )
        if isinstance(expr, UnaryOp):
            return replace(expr, value=self._expand_expr(expr.value))
        if isinstance(expr, Select):
            return replace(
                expr,
                condition=self._expand_expr(expr.condition),
                then=self._expand_expr(expr.then),
                otherwise=self._expand_expr(expr.otherwise),
            )
        if isinstance(expr, Piecewise):
            return replace(
                expr,
                branches=[(self._expand_expr(c), self._expand_expr(v)) for c, v in expr.branches],
                default=self._expand_expr(expr.default) if expr.default is not None else None,
            )
        if isinstance(expr, Reduce):
            return replace(expr, value=self._expand_expr(expr.value))
        return expr

    # Macro expansions ---------------------------------------------------------
    def _expand_macro(self, node: Macro) -> Any:
        name = node.name.lower()
        if name == "softmax":
            return self._expand_softmax(node)
        if name in {"layer_norm", "layernorm"}:
            return self._expand_layer_norm(node)
        # Unknown macro: conservatively drop '@' and treat as a call
        return Call(name=name, args=list(node.args), kwargs=dict(node.kwargs))

    def _expand_softmax(self, node: Macro) -> Call:
        # @softmax(x, axis=?, mask=?, fill=?): dispatch to masked_softmax or softmax
        args = list(node.args)
        kwargs = dict(node.kwargs)
        if "mask" in kwargs:
            return Call(name="masked_softmax", args=args, kwargs=kwargs)
        return Call(name="softmax", args=args, kwargs=kwargs)

    def _expand_layer_norm(self, node: Macro) -> Call:
        # @layer_norm(x, axis=?, eps=?): normalized builtin call
        return Call(name="layernorm", args=list(node.args), kwargs=dict(node.kwargs))


def expand_macros(program: Program) -> Program:
    return MacroExpander().expand(program)
