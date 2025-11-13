from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .ast import (
    BinOp,
    Block,
    Call,
    FnDef,
    Let,
    Piecewise,
    Select,
    UnaryOp,
)
from .ast import (
    Equation as AstEquation,
)
from .ast import (
    Lhs as AstLhs,
)
from .ast import (
    Program as AstProgram,
)
from .ast import (
    Reduce as AstReduce,
)
from .ast import (
    Tensor as AstTensor,
)
from .ir import Equation, FuncCall, ProgramIR, TensorRef, Term


def _collect_ir_indices(expr: Any, out: Dict[str, None]) -> None:
    from .ir import FuncCall as IRFuncCall
    from .ir import TensorRef as IRTensorRef
    from .ir import Term as IRTerm

    if isinstance(expr, IRTensorRef):
        for idx in getattr(expr, "indices", []) or []:
            out.setdefault(str(idx), None)
        for idx in getattr(expr, "rolling", {}).keys():
            out.setdefault(str(idx), None)
        for spec in getattr(expr, "index_specs", []) or []:
            out.setdefault(str(getattr(spec, "axis", "")), None)
        return
    if isinstance(expr, IRTerm):
        for f in expr.factors:
            _collect_ir_indices(f, out)
        return
    if isinstance(expr, IRFuncCall):
        arg = expr.arg
        if isinstance(arg, tuple):
            for item in arg:
                _collect_ir_indices(item, out)
        elif arg is not None:
            _collect_ir_indices(arg, out)
        for v in (expr.kwargs or {}).values():
            if isinstance(v, (IRTerm, IRFuncCall, IRTensorRef)):
                _collect_ir_indices(v, out)
        return
    if (
        hasattr(expr, "axis") and hasattr(expr, "name") and expr.name == "index_fn"
    ):  # pragma: no cover
        out.setdefault(str(expr.axis), None)
        return
    # literals/others ignored


def _ir_indices(expr: Any) -> List[str]:
    seen: Dict[str, None] = {}
    _collect_ir_indices(expr, seen)
    return list(seen.keys())


class _Lowerer:
    def __init__(self) -> None:
        self.equations: List[Equation] = []
        self.exports: List[str] = []
        self._tmp_counter = 0
        self._funcs: Dict[str, FnDef] = {}
        self._consts: Dict[str, Any] = {}

    def gensym(self, prefix: str = "__t") -> str:
        name = f"{prefix}{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def lower_program(self, prog: AstProgram) -> ProgramIR:
        # Register functions
        for fn in prog.fns:
            self._funcs[fn.name] = fn
        # Load imports and register namespaced functions (best-effort)
        self._load_imports(prog)
        # Evaluate params/consts to constants for folding
        self._evaluate_constants(prog)
        for stmt in prog.statements:
            self.lower_stmt(stmt)
        # Exports: propagate from AST
        if getattr(prog, "exports", None):
            self.exports.extend([str(x) for x in prog.exports])
        return ProgramIR(equations=self.equations, exports=self.exports)

    def lower_stmt(self, stmt: Any) -> None:
        if isinstance(stmt, Let):
            self._lower_let(stmt)
            return
        if isinstance(stmt, AstEquation):
            self._lower_equation(stmt)
            return
        if isinstance(stmt, Block):
            for inner in stmt.statements:
                self.lower_stmt(inner)
            return
        if isinstance(stmt, FnDef):
            # Functions are not executed at lowering time; ignore for now.
            return
        raise NotImplementedError(f"Unsupported statement: {type(stmt)}")

    def _lhs_to_ir(self, lhs: AstLhs) -> TensorRef:
        return TensorRef(name=lhs.name, indices=list(lhs.dims), dotted_axes=[])

    def _tensor_to_ir(self, t: AstTensor) -> TensorRef:
        # Parse rich index tokens like legacy: dotted axes, streaming, offsets, slices
        indices: List[str] = []
        dotted_axes: List[str] = []
        rolling: Dict[str, int] = {}
        from .ir import IndexSpec, SliceSpec  # local import to avoid cycles

        STREAM_RE = re.compile(r"^\*([A-Za-z_][A-Za-z0-9_']*)([+-]\d+)?$")
        OFFSET_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_']*)([+-]\d+)?$")

        index_specs: List[IndexSpec] = []
        for raw in t.dims or []:
            token = str(raw).strip()
            if not token:
                continue
            dotted = token.endswith(".")
            if dotted:
                token = token[:-1].strip()

            slice_spec = None
            offset = 0
            axis_name = ""

            if token.startswith("*"):
                m = STREAM_RE.match(token)
                if not m:
                    raise ValueError(f"Invalid streaming index token: {raw}")
                axis_name = m.group(1)
                rolling_offset = int(m.group(2) or 0)
                rolling[axis_name] = rolling_offset
            elif ":" in token:
                parts = [p.strip() for p in token.split(":")]
                if len(parts) not in (2, 3):
                    raise ValueError(f"Invalid slice token: {raw}")

                def _b(s: str) -> Optional[int]:
                    return None if s == "" else int(s)

                slice_spec = SliceSpec(
                    start=_b(parts[0]),
                    stop=_b(parts[1]),
                    step=_b(parts[2]) if len(parts) == 3 else None,
                )
                axis_name = ":".join(parts)
            else:
                m = OFFSET_RE.match(token)
                if not m:
                    raise ValueError(f"Invalid index token: {raw}")
                axis_name = m.group(1)
                offset = int(m.group(2) or 0)

            indices.append(axis_name)
            if dotted:
                dotted_axes.append(axis_name)
            index_specs.append(IndexSpec(axis=axis_name, offset=offset, slice=slice_spec))

        return TensorRef(
            name=t.name,
            indices=indices,
            dotted_axes=dotted_axes,
            rolling=rolling,
            index_specs=index_specs,
        )

    def _term(self, *factors: Any) -> Term:
        flat: List[Any] = []
        for f in factors:
            if isinstance(f, Term):
                flat.extend(f.factors)
            else:
                flat.append(f)
        return Term(factors=flat)

    def _case(self, *args: Any) -> FuncCall:
        # case(cond1, v1, cond2, v2, [default])
        return FuncCall(name="case", arg=tuple(args), kwargs={})

    def _select_to_case(self, sel: Select) -> FuncCall:
        return self._case(
            self._lower_expr(sel.condition),
            self._lower_expr(sel.then),
            self._lower_expr(sel.otherwise),
        )

    def _piecewise_to_case(self, node: Piecewise) -> FuncCall:
        args: List[Any] = []
        for cond, val in node.branches:
            args.append(self._lower_expr(cond))
            args.append(self._lower_expr(val))
        if node.default is not None:
            args.append(self._lower_expr(node.default))
        return self._case(*args)

    def _lower_bool_not(self, value: Any) -> FuncCall:
        # not x  => case(x, 0, 1)
        return self._case(self._lower_expr(value), 0, 1)

    def _lower_reduce_expr(self, node: AstReduce) -> TensorRef:
        # Lower to a temporary with projection semantics; supports sum/max/mean
        op = node.op.lower()
        if op in {"avg"}:
            op = "mean"
        if op not in {"sum", "max", "mean"}:
            raise NotImplementedError(f"Unsupported reduce op '{node.op}'")
        value_ir = self._lower_expr(node.value)
        # Determine output indices by removing reduced axes
        rhs_indices = _ir_indices(value_ir)
        out_indices = [ax for ax in rhs_indices if ax not in set(node.axes)]
        name = self.gensym("__red")
        lhs = TensorRef(name=name, indices=out_indices, dotted_axes=[])
        eq = Equation(lhs=lhs, rhs=value_ir, projection=op)
        self.equations.append(eq)
        return TensorRef(name=name, indices=out_indices, dotted_axes=[])

    def _lower_binop(self, node: BinOp) -> Any:
        op = node.op
        if op == "*":
            left = self._lower_expr(node.left)
            right = self._lower_expr(node.right)
            return self._term(left, right)
        if op in {"+", "-"}:
            left = self._lower_expr(node.left)
            right = self._lower_expr(node.right)
            # Hoist addition into separate equations over a fresh temporary
            name = self.gensym("__add")
            idxs = []
            seen: Dict[str, None] = {}
            for sym in _ir_indices(left) + _ir_indices(right):
                if sym not in seen:
                    seen[sym] = None
                    idxs.append(sym)
            lhs = TensorRef(name=name, indices=idxs, dotted_axes=[])
            self.equations.append(Equation(lhs=lhs, rhs=left, projection="sum"))
            if op == "+":
                self.equations.append(Equation(lhs=lhs, rhs=right, projection="sum"))
            else:
                self.equations.append(
                    Equation(lhs=lhs, rhs=self._term(-1, right), projection="sum")
                )
            return TensorRef(name=name, indices=idxs, dotted_axes=[])
        if op == "/":
            left = self._lower_expr(node.left)
            right = self._lower_expr(node.right)
            inv = FuncCall(name="pow", arg=(right, -1), kwargs={})
            return self._term(left, inv)
        if op == "**":
            left = self._lower_expr(node.left)
            right = self._lower_expr(node.right)
            return FuncCall(name="pow", arg=(left, right), kwargs={})
        if op == "&&":
            return self._term(self._lower_expr(node.left), self._lower_expr(node.right))
        if op == "||":
            # 1 - (1-a)*(1-b)  == case(a, 1, case(b, 1, 0))
            a = self._lower_expr(node.left)
            b = self._lower_expr(node.right)
            return self._case(a, 1, self._case(b, 1, 0))
        raise NotImplementedError(f"Unsupported binary op '{op}'")

    def _lower_expr(self, expr: Any) -> Any:
        if isinstance(expr, AstTensor):
            return self._tensor_to_ir(expr)
        if isinstance(expr, BinOp):
            return self._lower_binop(expr)
        if isinstance(expr, UnaryOp):
            if expr.op == "-":
                inner = self._lower_expr(expr.value)
                return self._term(-1, inner)
            if expr.op == "!":
                return self._lower_bool_not(expr.value)
            raise NotImplementedError(f"Unsupported unary op '{expr.op}'")
        if isinstance(expr, Select):
            return self._select_to_case(expr)
        if isinstance(expr, Piecewise):
            return self._piecewise_to_case(expr)
        if isinstance(expr, AstReduce):
            return self._lower_reduce_expr(expr)
        if isinstance(expr, Call):
            name = expr.name
            # Inline user-defined pure functions
            if name in self._funcs:
                return self._inline_function(self._funcs[name], expr)
            if name.lower() == "select" and len(expr.args) == 3:
                return self._case(
                    self._lower_expr(expr.args[0]),
                    self._lower_expr(expr.args[1]),
                    self._lower_expr(expr.args[2]),
                )
            args = tuple(self._lower_expr(a) for a in (expr.args or []))
            kwargs = {k: self._lower_expr(v) for k, v in (expr.kwargs or {}).items()}
            # Constant-fold kwargs that are plain identifiers
            for k, v in list(kwargs.items()):
                if isinstance(v, str) and v in self._consts:
                    kwargs[k] = self._consts[v]
            return FuncCall(name=name, arg=args if len(args) != 1 else args[0], kwargs=kwargs)
        # literals (numbers/bools)
        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, str):
            # Replace known params/consts with literal values; otherwise tensor symbol
            if expr in self._consts:
                return self._consts[expr]
            return TensorRef(name=expr, indices=[], dotted_axes=[])
        return expr

    def _apply_guard(self, rhs: Any, guard: Any) -> Any:
        # Multiply rhs by guard indicator: rhs * guard
        return self._term(self._lower_expr(rhs), self._lower_expr(guard))

    def _lower_let(self, stmt: Let) -> None:
        lhs = self._lhs_to_ir(stmt.lhs)
        rhs = self._lower_expr(stmt.value)
        self.equations.append(Equation(lhs=lhs, rhs=rhs, projection="sum"))

    def _lower_equation(self, stmt: AstEquation) -> None:
        lhs = self._lhs_to_ir(stmt.lhs)
        rhs = stmt.rhs
        if stmt.guard is not None:
            lowered = self._apply_guard(rhs, stmt.guard)
            self.equations.append(Equation(lhs=lhs, rhs=lowered, projection="sum"))
            return
        lowered = self._lower_expr(rhs)
        # Wrap non-Term rhs into a single-factor Term to ensure projection of extra axes
        if not isinstance(lowered, Term):
            lowered = Term(factors=[lowered])
        self.equations.append(Equation(lhs=lhs, rhs=lowered, projection="sum"))

    def _load_imports(self, prog: AstProgram) -> None:
        from pathlib import Path

        from .parser_expr import parse_program as _parse

        for imp in getattr(prog, "imports", []) or []:
            path_str = getattr(imp, "path", "")
            alias = getattr(imp, "alias", None)
            if not path_str:
                continue
            # Resolve file path (best effort): try exact, then append .fuse
            path = Path(path_str)
            if not path.exists():
                if not path.suffix:
                    cand = path.with_suffix(".fuse")
                    if cand.exists():
                        path = cand
                    else:
                        continue
                else:
                    continue
            try:
                text = path.read_text(encoding="utf-8")
                imported = _parse(text)
            except Exception:
                continue
            ns = alias or path.stem
            for fn in imported.fns:
                qualified = f"{ns}::{fn.name}"
                # Copy def but with qualified name
                self._funcs[qualified] = fn

    # Constants evaluation -----------------------------------------------------
    def _eval_const_expr(self, expr: Any) -> Any:
        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, str):
            if expr in self._consts:
                return self._consts[expr]
            raise ValueError(f"Unknown constant identifier: {expr}")
        if isinstance(expr, UnaryOp):
            val = self._eval_const_expr(expr.value)
            if expr.op == "-":
                return -val
            if expr.op == "+":
                return +val
            raise ValueError(f"Unsupported unary in constant: {expr.op}")
        if isinstance(expr, BinOp):
            a = self._eval_const_expr(expr.left)
            b = self._eval_const_expr(expr.right)
            if expr.op == "+":
                return a + b
            if expr.op == "-":
                return a - b
            if expr.op == "*":
                return a * b
            if expr.op == "/":
                return a / b
            if expr.op == "**":
                return a**b
            raise ValueError(f"Unsupported binary in constant: {expr.op}")
        raise ValueError("Unsupported constant expression")

    def _evaluate_constants(self, prog: AstProgram) -> None:
        # Params may depend on earlier params/consts
        for c in prog.consts:
            if c.value is not None:
                self._consts[c.name] = self._eval_const_expr(c.value)
        for p in prog.params:
            if p.default is not None:
                try:
                    self._consts[p.name] = self._eval_const_expr(p.default)
                except Exception:
                    # leave unevaluated if not reducible
                    pass

    # Function inlining --------------------------------------------------------
    def _inline_function(self, fn: FnDef, call: Call) -> Any:
        # Map formals to actuals (positional only for now)
        if len(call.args) != len(fn.params):
            raise ValueError(
                f"Function '{fn.name}' expects {len(fn.params)} args; got {len(call.args)}"
            )
        actual_ir = [self._lower_expr(arg) for arg in call.args]

        # Compute extras dims: right-align mapping of formals to actuals
        formal_dims: List[List[str]] = [p.dims for p in fn.params]
        actual_dims: List[List[str]] = []
        for expr in actual_ir:
            if hasattr(expr, "indices"):
                actual_dims.append(list(expr.indices))
            else:
                actual_dims.append([])
        # derive global extras and dims mapping
        dims_map: Dict[str, str] = {}
        extras: List[str] = []
        for f_dims, a_dims in zip(formal_dims, actual_dims):
            fN, aN = len(f_dims), len(a_dims)
            if fN > aN:
                raise ValueError("Function argument has fewer axes than formal dims")
            # right align
            for k in range(1, fN + 1):
                f_ax = f_dims[-k]
                a_ax = a_dims[-k]
                prev = dims_map.get(f_ax)
                if prev is None:
                    dims_map[f_ax] = a_ax
                elif prev != a_ax:
                    raise ValueError(f"Inconsistent binding for axis '{f_ax}' across arguments")
            for ax in a_dims[: aN - fN]:
                if ax not in extras:
                    extras.append(ax)

        # Prefix for local names
        prefix = self.gensym("__fn") + "_"
        name_map: Dict[str, str] = {}
        local_dims: Dict[str, List[str]] = {}

        def remap_dims(dims: List[str]) -> List[str]:
            mapped = [dims_map.get(ax, ax) for ax in dims]
            full = list(mapped)
            for ax in extras:
                if ax not in full:
                    full.append(ax)
            return full

        # Predeclare return symbol
        ret_name = None
        if fn.returns is not None:
            ret_name = prefix + fn.returns.name
            local_dims[ret_name] = remap_dims(fn.returns.dims)
            name_map[fn.returns.name] = ret_name

        # Walk statements and emit IR equations
        for st in fn.body.statements:
            if isinstance(st, Let) or isinstance(st, AstEquation):
                _lhs = st.lhs
                minted = name_map.get(_lhs.name) or (prefix + _lhs.name)
                name_map[_lhs.name] = minted
                dims_full = remap_dims(_lhs.dims)
                local_dims[minted] = dims_full
                lhs_ir = TensorRef(name=minted, indices=dims_full, dotted_axes=[])
                # Lower RHS in local env
                rhs_ir = self._lower_expr_with_env(st.rhs, name_map, local_dims, fn, actual_ir)
                self.equations.append(Equation(lhs=lhs_ir, rhs=rhs_ir, projection="sum"))
            else:
                raise ValueError("Only let/equations are allowed in pure functions")

        if ret_name is None:
            raise ValueError(f"Function '{fn.name}' missing return declaration")
        return TensorRef(name=ret_name, indices=local_dims[ret_name], dotted_axes=[])

    def _lower_expr_with_env(
        self,
        expr: Any,
        name_map: Dict[str, str],
        local_dims: Dict[str, List[str]],
        fn: FnDef,
        actual_ir: List[Any],
    ) -> Any:
        # Substitute param tensors and local tensors by name
        if isinstance(expr, AstTensor):
            # Param substitution
            for i, p in enumerate(fn.params):
                if expr.name == p.name:
                    return actual_ir[i]
            # Local references
            minted = name_map.get(expr.name)
            if minted is not None:
                dims = local_dims.get(minted) or []
                return TensorRef(name=minted, indices=list(dims), dotted_axes=[])
            # Global tensor
            return self._tensor_to_ir(expr)
        if isinstance(expr, BinOp):
            left = self._lower_expr_with_env(expr.left, name_map, local_dims, fn, actual_ir)
            right = self._lower_expr_with_env(expr.right, name_map, local_dims, fn, actual_ir)
            return self._lower_binop(BinOp(left=left, op=expr.op, right=right))
        if isinstance(expr, UnaryOp):
            value = self._lower_expr_with_env(expr.value, name_map, local_dims, fn, actual_ir)
            return self._lower_expr(UnaryOp(op=expr.op, value=value))
        if isinstance(expr, Select):
            return self._select_to_case(
                Select(
                    condition=self._lower_expr_with_env(
                        expr.condition, name_map, local_dims, fn, actual_ir
                    ),
                    then=self._lower_expr_with_env(expr.then, name_map, local_dims, fn, actual_ir),
                    otherwise=self._lower_expr_with_env(
                        expr.otherwise, name_map, local_dims, fn, actual_ir
                    ),
                )
            )
        if isinstance(expr, Piecewise):
            return self._piecewise_to_case(
                Piecewise(
                    branches=[
                        (
                            self._lower_expr_with_env(c, name_map, local_dims, fn, actual_ir),
                            self._lower_expr_with_env(v, name_map, local_dims, fn, actual_ir),
                        )
                        for c, v in expr.branches
                    ],
                    default=self._lower_expr_with_env(
                        expr.default, name_map, local_dims, fn, actual_ir
                    )
                    if expr.default is not None
                    else None,
                )
            )
        if isinstance(expr, AstReduce):
            return self._lower_reduce_expr(
                AstReduce(
                    op=expr.op,
                    value=self._lower_expr_with_env(
                        expr.value, name_map, local_dims, fn, actual_ir
                    ),
                    axes=list(expr.axes),
                )
            )
        if isinstance(expr, Call):
            name = expr.name
            if name in self._funcs:
                # nested call
                return self._inline_function(self._funcs[name], expr)
            args = tuple(
                self._lower_expr_with_env(a, name_map, local_dims, fn, actual_ir)
                for a in (expr.args or [])
            )
            kwargs = {
                k: self._lower_expr_with_env(v, name_map, local_dims, fn, actual_ir)
                for k, v in (expr.kwargs or {}).items()
            }
            return FuncCall(name=name, arg=args if len(args) != 1 else args[0], kwargs=kwargs)
        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, str):
            if expr in self._consts:
                return self._consts[expr]
            return TensorRef(name=expr, indices=[], dotted_axes=[])
        return expr


def lower_to_ir(prog: AstProgram) -> ProgramIR:
    return _Lowerer().lower_program(prog)
