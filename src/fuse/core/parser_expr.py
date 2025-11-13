from __future__ import annotations

import ast as _ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError, UnexpectedInput

from .ast import (
    AxisDecl,
    BinOp,
    Block,
    Call,
    Const,
    Equation,
    FnDef,
    Import,
    Let,
    Lhs,
    Macro,
    Param,
    Piecewise,
    Program,
    Reduce,
    Select,
    Tensor,
    UnaryOp,
)
from .exceptions import ParseError

GRAMMAR_FILE = Path(__file__).with_name("expr_grammar.lark")


def _build_parser() -> Lark:
    return Lark(
        GRAMMAR_FILE.read_text(),
        parser="earley",
        start="program",
        propagate_positions=True,
        maybe_placeholders=False,
        ambiguity="resolve",
    )


class _AstXform(Transformer):
    def __init__(self, text: str):
        super().__init__()
        self._text = text
        self._lines = text.splitlines()

    # Helpers -----------------------------------------------------------------
    def _slice(self, meta) -> str:
        return self._text[meta.start_pos : meta.end_pos]

    def _line_text(self, line: int) -> str:
        if 1 <= line <= len(self._lines):
            return self._lines[line - 1]
        return ""

    # Top-level ---------------------------------------------------------------
    def toplevel(self, items):
        # toplevel rule just passes through its single child
        return items[0] if items else None

    def program(self, items: List[Any]) -> Program:
        prog = Program()
        for item in items:
            if item is None:
                continue
            # Skip NEWLINE and separator tokens
            if isinstance(item, Token) and item.type in ('NEWLINE', 'SEMICOLON'):
                continue
            if isinstance(item, Import):
                prog.imports.append(item)
            elif isinstance(item, Param):
                prog.params.append(item)
            elif isinstance(item, AxisDecl):
                prog.axes.append(item)
            elif isinstance(item, Const):
                prog.consts.append(item)
            elif isinstance(item, FnDef):
                prog.fns.append(item)
            elif isinstance(item, tuple) and item and item[0] == "__export__":
                prog.exports.append(item[1])
            else:
                prog.statements.append(item)
        return prog

    # Declarations ------------------------------------------------------------
    @v_args(meta=True)
    def import_stmt(self, meta, items):
        path_tok: Token = items[0]
        alias: Optional[str] = None
        if len(items) > 1:
            alias_tok: Token = items[1]
            alias = alias_tok.value
        return Import(path=_ast.literal_eval(path_tok.value), alias=alias, line=meta.line, column=meta.column, source=self._slice(meta).strip())

    @v_args(meta=True)
    def export_stmt(self, meta, items):
        name_tok: Token = items[0]
        return ("__export__", name_tok.value)

    @v_args(meta=True)
    def param(self, meta, items):
        name_tok: Token = items[0]
        type_name: Optional[str] = None
        default: Any = None
        if len(items) >= 2 and isinstance(items[1], Token):
            type_name = items[1].value
            items = items[2:]
        else:
            items = items[1:]
        if items:
            default = items[0]
        return Param(name=name_tok.value, type_name=type_name, default=default, line=meta.line, column=meta.column, source=self._slice(meta).strip())

    @v_args(meta=True)
    def axis(self, meta, items):
        name_tok: Token = items[0]
        value = items[1] if len(items) > 1 else None
        return AxisDecl(name=name_tok.value, value=value, line=meta.line, column=meta.column, source=self._slice(meta).strip())

    @v_args(meta=True)
    def const(self, meta, items):
        name_tok: Token = items[0]
        value = items[1]
        return Const(name=name_tok.value, value=value, line=meta.line, column=meta.column, source=self._slice(meta).strip())

    # Functions ---------------------------------------------------------------
    def fn_params(self, items):
        return list(items)

    @v_args(meta=True)
    def fn_def(self, meta, items):
        # Items: name [, params] [, -> lhs] , body
        name = items[0]
        body = items[-1]
        ret_lhs = None
        params: List[Lhs] = []
        cursor = 1
        if cursor < len(items) - 1 and isinstance(items[cursor], list):
            params = items[cursor]
            cursor += 1
        if cursor < len(items) - 1 and isinstance(items[cursor], Lhs):
            ret_lhs = items[cursor]
        return FnDef(name=name, params=params or [], returns=ret_lhs, body=body, line=meta.line, column=meta.column, source=self._slice(meta).strip())

    # Statements --------------------------------------------------------------
    def stmt(self, items):
        return items[0]

    @v_args(meta=True)
    def equation(self, meta, items):
        return Equation(lhs=items[0], rhs=items[1], line=meta.line, column=meta.column, source=self._slice(meta).strip())

    @v_args(meta=True)
    def equation_when(self, meta, items):
        lhs = items[0]
        guard = items[1]
        rhs = items[2]
        return Equation(lhs=lhs, rhs=rhs, guard=guard, line=meta.line, column=meta.column, source=self._slice(meta).strip())

    def dims(self, items):
        return [tok.value if isinstance(tok, Token) else str(tok) for tok in items]

    @v_args(meta=True)
    def lhs(self, meta, items):
        name_tok: Token = items[0]
        dims = items[1] if len(items) > 1 else []
        return Lhs(name=name_tok.value, dims=dims, line=meta.line, column=meta.column, source=self._slice(meta).strip())

    @v_args(meta=True)
    def let_stmt(self, meta, items):
        lhs = items[0]
        value = items[1]
        return Let(lhs=lhs, value=value, line=meta.line, column=meta.column, source=self._slice(meta).strip())

    def block(self, items):
        # items may contain nested statements
        stmts: List[Any] = []
        for it in items:
            if it is None:
                continue
            if isinstance(it, list):
                stmts.extend(it)
            else:
                stmts.append(it)
        return Block(statements=stmts)

    # Expressions -------------------------------------------------------------
    def group(self, items):
        return items[0]

    def number(self, items):
        tok: Token = items[0]
        text = tok.value
        # Use Python literal semantics: int or float
        try:
            if any(ch in text for ch in ".eE"):
                return float(text)
            return int(text)
        except Exception:
            return float(text)

    def ident(self, items):
        tok: Token = items[0]
        return tok.value

    def call_args(self, items):
        # mixed positional and kwargs
        pos: List[Any] = []
        kwargs: Dict[str, Any] = {}
        for it in items:
            if isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str):
                kwargs[it[0]] = it[1]
            else:
                pos.append(it)
        return (pos, kwargs)

    def call(self, items):
        name = items[0]
        pos: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if len(items) > 1:
            pos, kwargs = items[1]
        return Call(name=name, args=pos, kwargs=kwargs)

    def macro_call(self, items):
        name_tok: Token = items[0]
        pos: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if len(items) > 1:
            pos, kwargs = items[1]
        return Macro(name=name_tok.value, args=pos, kwargs=kwargs)

    def tensor_ref(self, items):
        name_tok: Token = items[0]
        dims = items[1] if len(items) > 1 else []
        return Tensor(name=name_tok.value, dims=dims)

    def index_suffix(self, items):
        return items[0] if items else []

    def index_list(self, items):
        return [tok.value if isinstance(tok, Token) else str(tok) for tok in items]

    def index_symbol(self, items):
        return items[0]

    def _fold_binop(self, first: Any, rest: List[Any]) -> Any:
        expr = first
        # rest comes as [op_tok, value, op_tok, value, ...]
        for i in range(0, len(rest), 2):
            op_tok = rest[i]
            rhs = rest[i + 1]
            op = op_tok.value if isinstance(op_tok, Token) else str(op_tok)
            expr = BinOp(left=expr, op=op, right=rhs)
        return expr

    def or_expr(self, items):
        if len(items) == 1:
            return items[0]
        return self._fold_binop(items[0], items[1:])

    def and_expr(self, items):
        if len(items) == 1:
            return items[0]
        return self._fold_binop(items[0], items[1:])

    def add_expr(self, items):
        if len(items) == 1:
            return items[0]
        return self._fold_binop(items[0], items[1:])

    def mul_expr(self, items):
        if len(items) == 1:
            return items[0]
        return self._fold_binop(items[0], items[1:])

    def pow_expr(self, items):
        # unary (** pow_expr)?  -> either 1 item or 2 items
        if len(items) == 1:
            return items[0]
        # items[0] ** items[1] (right-assoc)
        return BinOp(left=items[0], op="**", right=items[1])

    def unary(self, items):
        op_tok: Token = items[0]
        value = items[1]
        return UnaryOp(op=op_tok.value, value=value)

    def ternary(self, items):
        # items: or_expr ["?", expr, ":", expr]
        if len(items) == 1:
            return items[0]
        cond = items[0]
        then_v = items[1]
        else_v = items[2]
        return Select(condition=cond, then=then_v, otherwise=else_v)

    # Piecewise / reduce sugar -------------------------------------------------
    def piece_entry(self, items):
        # cond -> value
        return (items[0], items[1])

    def piece_entries(self, items):
        return list(items)

    def default_entry(self, items):
        return (None, items[0])

    def piecewise(self, items):
        branches: List[Any] = []
        default: Optional[Any] = None
        for it in items:
            if isinstance(it, list):
                branches.extend(it)
            elif isinstance(it, tuple) and it[0] is None:
                default = it[1]
            else:
                branches.append(it)
        return Piecewise(branches=branches, default=default)

    def reduce_axes(self, items):
        return [tok.value if isinstance(tok, Token) else str(tok) for tok in items]

    def reduce_expr(self, items):
        # items: IDENT, axes(list), expr
        op_tok: Token = items[0]
        axes = items[1] if len(items) > 2 else []
        value = items[-1]
        return Reduce(op=op_tok.value, value=value, axes=axes)

    def qual_ident(self, items):
        parts = [t.value if isinstance(t, Token) else str(t) for t in items]
        return "::".join(parts)

    def kwarg(self, items):
        key_tok: Token = items[0]
        val = items[1]
        return (key_tok.value, val)


def parse_program(text: str) -> Program:
    parser = _build_parser()
    try:
        tree = parser.parse(text)
    except UnexpectedInput as exc:
        line = exc.line or 1
        column = exc.column or 1
        raise ParseError("Syntax error while parsing program", line=line, column=column) from exc
    except LarkError as exc:  # pragma: no cover - defensive
        raise ParseError(str(exc)) from exc
    xform = _AstXform(text)
    return xform.transform(tree)
