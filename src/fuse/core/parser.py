from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lark import Lark, Token, Transformer, Tree, v_args
from lark.exceptions import LarkError, UnexpectedInput

from .exceptions import ParseError
from .ir import (
    Equation,
    FuncCall,
    IndexFunction,
    IndexSpec,
    ProgramIR,
    SliceSpec,
    TensorRef,
    Term,
)

GRAMMAR_PATH = Path(__file__).with_name("fuse_grammar.lark")

PROJECTION_MAP = {
    "=": "sum",
    "+=": "sum",
    "max=": "max",
    "avg=": "mean",
}

INDEX_FUNCTION_NAMES = {"even", "odd"}

STREAM_RE = re.compile(r"^\*([A-Za-z_][A-Za-z0-9_']*)([+-]\d+)?$")
OFFSET_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_']*)([+-]\d+)?$")


@lru_cache(maxsize=1)
def _build_lark() -> Lark:
    grammar = GRAMMAR_PATH.read_text()
    return Lark(
        grammar,
        parser="earley",
        start="program",
        ambiguity="resolve",
        propagate_positions=True,
        maybe_placeholders=False,
    )


@dataclass
class ExportNode:
    name: str
    line: int
    column: int
    source: str


@dataclass
class AssignmentNode:
    lhs: TensorRef
    op: str
    rhs_terms: List[Any]
    line: int
    column: int
    source: str


@dataclass
class SinkNode:
    target: str
    value: Any
    line: int
    column: int
    source: str


@dataclass
class ProgramNodes:
    exports: List[ExportNode]
    statements: List[Any]


@dataclass
class KwArg:
    name: str
    value: Any
    line: int
    column: int


@dataclass
class IndexSuffix:
    tokens: List[Token]
    is_paren: bool


class SumList(list):
    """Marker list for additive expression groups."""


class FuseTransformer(Transformer):
    def __init__(self, text: str):
        super().__init__()
        self.text = text
        self.lines = text.splitlines()

    # ------------------------------------------------------------------ helpers
    def _slice(self, meta) -> str:
        return self.text[meta.start_pos : meta.end_pos]

    def _line_text(self, line: int) -> str:
        if 1 <= line <= len(self.lines):
            return self.lines[line - 1]
        return ""

    def _error(self, meta, message: str) -> None:
        raise ParseError(
            message,
            line=meta.line,
            column=meta.column,
            line_text=self._line_text(meta.line),
        )

    def _error_token(self, token: Token, message: str) -> None:
        raise ParseError(
            message,
            line=token.line,
            column=token.column,
            line_text=self._line_text(token.line),
        )

    def _score_tree(self, tree: Tree) -> int:
        score = 0
        for subtree in tree.iter_subtrees():
            if subtree.data == "func_call":
                score += 1
        return score

    def _ambig(self, trees):
        best = max(trees, key=self._score_tree)
        return self._transform_tree(best)

    def _ensure_single(self, exprs: Sequence[Any], meta, context: str) -> Any:
        if len(exprs) != 1:
            self._error(meta, f"{context} must resolve to a single expression")
        return exprs[0]

    def _normalize_literal(self, value: str) -> Any:
        return ast.literal_eval(value)

    def _normalize_number(self, value: str) -> Any:
        literal = ast.literal_eval(value)
        return literal

    def _normalize_kwarg_value(self, value: Any) -> Any:
        if isinstance(value, TensorRef) and not value.indices and not value.dotted_axes:
            return value.name
        return value

    def _extract_axis_symbol(self, value: Any, meta, func_name: str) -> str:
        if isinstance(value, TensorRef):
            if value.indices or value.dotted_axes or value.rolling:
                self._error(meta, f"{func_name} axis must be a scalar symbol")
            return value.name
        if isinstance(value, str):
            return value
        self._error(meta, f"{func_name} axis must be a symbol")
        return ""

    def _parse_slice_bound(self, raw: str, token: Token) -> Optional[int]:
        value = raw.strip()
        if value == "":
            return None
        try:
            return int(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ParseError(
                f"Slice bounds must be integers: {raw}",
                line=token.line,
                column=token.column,
                line_text=self._line_text(token.line),
            ) from exc

    def _build_tensor_ref(
        self,
        name_token: Token,
        suffix: Optional[IndexSuffix],
        meta,
    ) -> TensorRef:
        indices: List[str] = []
        dotted_axes: List[str] = []
        rolling: Dict[str, int] = {}
        specs: List[IndexSpec] = []

        if suffix is None or not suffix.tokens:
            return TensorRef(name=name_token.value, indices=[], dotted_axes=[])

        for token in suffix.tokens:
            raw_token = token.value.strip()
            if not raw_token:
                self._error_token(token, "Empty index specifier")

            dotted = raw_token.endswith(".")
            if dotted:
                raw_token = raw_token[:-1].strip()
            slice_spec: Optional[SliceSpec] = None
            offset = 0
            rolling_offset: Optional[int] = None
            axis_name: str

            if raw_token.startswith("*"):
                match = STREAM_RE.match(raw_token)
                if not match:
                    self._error_token(token, "Invalid streaming index syntax")
                axis_name = match.group(1)
                rolling_offset = int(match.group(2) or 0)
            elif ":" in raw_token:
                parts = [part.strip() for part in raw_token.split(":")]
                if len(parts) not in (2, 3):
                    self._error_token(token, "Invalid slice specifier")
                start = self._parse_slice_bound(parts[0], token)
                stop = self._parse_slice_bound(parts[1], token)
                step = self._parse_slice_bound(parts[2], token) if len(parts) == 3 else None
                slice_spec = SliceSpec(start=start, stop=stop, step=step)
                axis_name = ":".join(part for part in parts)
            else:
                match = OFFSET_RE.match(raw_token)
                if not match:
                    self._error_token(token, f"Invalid index token '{token.value}'")
                axis_name = match.group(1)
                offset = int(match.group(2) or 0)

            if raw_token.startswith("*") and rolling_offset is not None:
                rolling[axis_name] = rolling_offset
                offset = 0

            indices.append(axis_name)
            if dotted:
                dotted_axes.append(axis_name)
            specs.append(IndexSpec(axis=axis_name, offset=offset, slice=slice_spec))

        lower_name = name_token.value.lower()
        if suffix.is_paren and lower_name in INDEX_FUNCTION_NAMES:
            if len(specs) != 1:
                self._error_token(suffix.tokens[0], f"{name_token.value} requires exactly one axis")
            if dotted_axes:
                self._error_token(
                    suffix.tokens[0],
                    f"{name_token.value} axis cannot include dotted indices",
                )
            spec = specs[0]
            if spec.slice is not None:
                self._error_token(
                    suffix.tokens[0], f"{name_token.value} axis does not support slices"
                )
            if spec.offset != 0:
                self._error_token(
                    suffix.tokens[0], f"{name_token.value} axis does not support offsets"
                )
            if spec.axis.startswith("*"):
                self._error_token(suffix.tokens[0], f"{name_token.value} axis cannot be streaming")
            return IndexFunction(name=lower_name, axis=spec.axis)

        return TensorRef(
            name=name_token.value,
            indices=indices,
            dotted_axes=dotted_axes,
            rolling=rolling,
            index_specs=specs,
            is_paren=suffix.is_paren,
        )

    # ------------------------------------------------------------------ visitors
    def program(self, items: List[Any]) -> ProgramNodes:
        exports: List[ExportNode] = []
        statements: List[Any] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, Token):
                continue
            if isinstance(item, ExportNode):
                exports.append(item)
            else:
                statements.append(item)
        return ProgramNodes(exports=exports, statements=statements)

    def statement(self, items):
        return items[0]

    def _nl(self, _items):
        return None

    def assign_ref(self, items):
        name_token: Token = items[0]
        suffix: Optional[IndexSuffix] = items[1] if len(items) > 1 else None
        return self._build_tensor_ref(name_token, suffix, name_token)

    @v_args(meta=True)
    def export_stmt(self, meta, items):
        name_token: Token = items[0]
        return ExportNode(
            name=name_token.value,
            line=meta.line,
            column=meta.column,
            source=self._slice(meta).strip(),
        )

    @v_args(meta=True)
    def sink_stmt(self, meta, items):
        target_token: Token = items[0]
        exprs: List[Any] = items[1]
        target = self._normalize_literal(target_token.value)
        value = self._ensure_single(exprs, meta, "sink expression")
        return SinkNode(
            target=target,
            value=value,
            line=meta.line,
            column=meta.column,
            source=self._slice(meta).strip(),
        )

    @v_args(meta=True)
    def assignment_stmt(self, meta, items):
        lhs = items[0]
        op = items[1]
        rhs_terms = items[2]
        if not isinstance(lhs, TensorRef):
            self._error(meta, "Left-hand side must be a tensor reference or symbol")
        return AssignmentNode(
            lhs=lhs,
            op=op,
            rhs_terms=list(rhs_terms),
            line=meta.line,
            column=meta.column,
            source=self._slice(meta).strip(),
        )

    def eq(self, _):
        return "="

    def plus(self, _):
        return "+="

    def max(self, _):
        return "max="

    def avg(self, _):
        return "avg="

    def sum_terms(self, items):
        return SumList(items)

    def term_product(self, items):
        factors: List[Any] = []
        for item in items:
            if isinstance(item, SumList):
                factors.extend(item)
            else:
                factors.append(item)
        if len(factors) == 1:
            return factors[0]
        return Term(factors=factors)

    @v_args(meta=True)
    def grouped_sum(self, meta, items):
        exprs = items[0]
        if isinstance(exprs, SumList):
            return self._ensure_single(exprs, meta, "Parenthesised expression")
        return exprs

    def literal(self, items):
        return items[0]

    def string(self, items):
        token: Token = items[0]
        return self._normalize_literal(token.value)

    def number(self, items):
        token: Token = items[0]
        return self._normalize_number(token.value)

    def true(self, _):
        return True

    def false(self, _):
        return False

    def none(self, _):
        return None

    def base_literal(self, items):
        return items[0]

    def list_literal(self, items):
        if not items:
            return []
        values = items[0]
        if isinstance(values, list):
            return list(values)
        return [values]

    def list_items(self, items):
        seq: List[Any] = []
        for item in items:
            if isinstance(item, Token) and item.type in {"NEWLINE", "COMMA"}:
                continue
            if item is None:
                continue
            seq.append(item)
        return seq

    def dict_pair(self, items):
        key, value = items
        return key, value

    def dict_literal(self, items):
        if not items:
            return {}
        entries = items[0]
        return {key: value for key, value in entries}

    def dict_items(self, items):
        seq: List[Any] = []
        for item in items:
            if isinstance(item, Token) and item.type in {"NEWLINE", "COMMA"}:
                continue
            if item is None:
                continue
            seq.append(item)
        return seq

    def tensor_ref(self, items):
        name_token: Token = items[0]
        suffix: Optional[IndexSuffix] = items[1] if len(items) > 1 else None
        return self._build_tensor_ref(name_token, suffix, name_token)

    def square(self, items):
        tokens = items[0] if items else []
        return IndexSuffix(tokens=tokens, is_paren=False)

    def paren(self, items):
        tokens = items[0] if items else []
        return IndexSuffix(tokens=tokens, is_paren=True)

    def index_list(self, items):
        return list(items)

    def index_token(self, items):
        token: Token = items[0]
        return token

    def index_symbol(self, items):
        return items[0]

    def slice_number(self, items):
        token: Token = items[0]
        return token.value

    def slice_empty(self, _items):
        return ""

    def slice_index(self, items):
        parts = [items[0], items[1]]
        if len(items) == 3:
            parts.append(items[2])
        return Token("SLICE", ":".join(parts))

    def call_args(self, items):
        seq: List[Any] = []
        for item in items:
            if isinstance(item, Token) and item.type in {"NEWLINE", "COMMA"}:
                continue
            if item is None:
                continue
            if isinstance(item, list) and not item:
                continue
            seq.append(item)
        return seq

    def call_empty(self, _items):
        return []

    def func_call_body(self, items):
        seq = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, list) and not item:
                continue
            if isinstance(item, Token) and item.type in {"NEWLINE", "COMMA"}:
                continue
            seq.append(item)
        if not seq:
            return []
        if len(seq) == 1:
            return seq[0]
        return seq

    @v_args(meta=True)
    def kwarg(self, meta, items):
        name_token: Token = items[0]
        exprs = items[-1]
        value = self._ensure_single(exprs, meta, f"keyword argument '{name_token.value}'")
        normalized = self._normalize_kwarg_value(value)
        return KwArg(
            name=name_token.value,
            value=normalized,
            line=meta.line,
            column=meta.column,
        )

    @v_args(meta=True)
    def func_call(self, meta, items):
        name_token: Token = items[0]
        name = name_token.value
        lower_name = name.lower()

        positional: List[Any] = []
        keyword_map: Dict[str, Any] = {}

        if len(items) > 1:
            for entry in items[1]:
                if isinstance(entry, KwArg):
                    if entry.name in keyword_map:
                        self._error(meta, f"Duplicate keyword argument '{entry.name}'")
                    keyword_map[entry.name] = entry.value
                else:
                    value = entry
                    if isinstance(value, SumList):
                        value = self._ensure_single(
                            value,
                            meta,
                            f"positional argument {len(positional) + 1} for {name}",
                        )
                    positional.append(value)

        if lower_name in INDEX_FUNCTION_NAMES:
            axis_value: Optional[Any] = None
            if positional:
                if len(positional) > 1:
                    self._error(meta, f"{name} accepts at most one positional argument")
                axis_value = positional[0]
            if "axis" in keyword_map:
                if axis_value is not None:
                    self._error(meta, f"{name} axis specified twice")
                axis_value = keyword_map.pop("axis")
            if keyword_map:
                unexpected = ", ".join(sorted(keyword_map))
                self._error(meta, f"Unexpected keyword arguments for {name}: {unexpected}")
            if axis_value is None:
                self._error(meta, f"{name} requires an axis argument")
            axis_name = self._extract_axis_symbol(axis_value, meta, name)
            return IndexFunction(name=lower_name, axis=axis_name)

        kwargs = dict(keyword_map)

        if not positional:
            arg_value: Any = None
        elif len(positional) == 1:
            arg_value = positional[0]
        else:
            arg_value = tuple(positional)

        return FuncCall(name=name, arg=arg_value, kwargs=kwargs)


def _clone_tensor_ref(ref: TensorRef) -> TensorRef:
    return TensorRef(
        name=ref.name,
        indices=list(ref.indices),
        dotted_axes=list(ref.dotted_axes),
        rolling=dict(ref.rolling),
        index_specs=[
            IndexSpec(
                axis=spec.axis,
                offset=spec.offset,
                slice=SliceSpec(
                    start=spec.slice.start,
                    stop=spec.slice.stop,
                    step=spec.slice.step,
                )
                if spec.slice is not None
                else None,
            )
            for spec in ref.index_specs
        ],
        is_paren=ref.is_paren,
    )


def _slice_signature(spec: Optional[SliceSpec]) -> Optional[Tuple[Any, Any, Any]]:
    if spec is None:
        return None
    return (spec.start, spec.stop, spec.step)


def _tensor_signature(ref: TensorRef) -> Tuple[Any, ...]:
    return (
        ref.name,
        tuple(ref.indices),
        tuple(ref.dotted_axes),
        tuple(sorted(ref.rolling.items())),
        tuple(
            (
                spec.axis,
                spec.offset,
                _slice_signature(spec.slice),
            )
            for spec in ref.index_specs
        ),
        ref.is_paren,
    )


def _object_signature(value: Any) -> Any:
    if isinstance(value, (TensorRef, Term, FuncCall, IndexFunction)):
        return _expr_signature(value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_object_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_object_signature(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((key, _object_signature(val)) for key, val in value.items())),
        )
    return value


def _expr_signature(expr: Any) -> Any:
    if expr is None:
        return None
    if isinstance(expr, TensorRef):
        return ("tensor", _tensor_signature(expr))
    if isinstance(expr, Term):
        return ("term", tuple(_expr_signature(factor) for factor in expr.factors))
    if isinstance(expr, FuncCall):
        return (
            "func",
            expr.name,
            _object_signature(expr.arg),
            tuple((key, _object_signature(val)) for key, val in sorted(expr.kwargs.items())),
        )
    if isinstance(expr, IndexFunction):
        return ("index_fn", expr.name, expr.axis)
    if isinstance(expr, tuple):
        return ("tuple", tuple(_object_signature(item) for item in expr))
    if isinstance(expr, list):
        return ("list", tuple(_object_signature(item) for item in expr))
    if isinstance(expr, dict):
        return (
            "dict",
            tuple(sorted((key, _object_signature(val)) for key, val in expr.items())),
        )
    return expr


def _equation_signature(eq: Equation) -> Tuple[Any, ...]:
    return (
        eq.is_source,
        eq.is_sink,
        eq.export,
        eq.projection,
        eq.src_file,
        eq.sink_file,
        _tensor_signature(eq.lhs),
        _expr_signature(eq.rhs),
    )


def _build_program_ir(nodes: ProgramNodes) -> ProgramIR:
    equations: List[Equation] = []

    for statement in nodes.statements:
        if isinstance(statement, AssignmentNode):
            projection = PROJECTION_MAP.get(statement.op)
            if projection is None:
                raise ParseError(
                    f"Unsupported assignment operator '{statement.op}'",
                    line=statement.line,
                    column=statement.column,
                    line_text=statement.source,
                )
            if (
                len(statement.rhs_terms) == 1
                and isinstance(statement.rhs_terms[0], str)
                and statement.op == "="
            ):
                lhs = _clone_tensor_ref(statement.lhs)
                equations.append(
                    Equation(
                        lhs=lhs,
                        rhs=None,
                        projection="sum",
                        src_file=statement.rhs_terms[0],
                        is_source=True,
                        line=statement.line,
                        column=statement.column,
                        source=statement.source,
                    )
                )
                continue

            for term in statement.rhs_terms:
                lhs = _clone_tensor_ref(statement.lhs)
                rhs_value = copy.deepcopy(term)
                equations.append(
                    Equation(
                        lhs=lhs,
                        rhs=rhs_value,
                        projection=projection,
                        line=statement.line,
                        column=statement.column,
                        source=statement.source,
                    )
                )
        elif isinstance(statement, SinkNode):
            equations.append(
                Equation(
                    lhs=TensorRef(name="__sink__", indices=[], dotted_axes=[]),
                    rhs=copy.deepcopy(statement.value),
                    sink_file=statement.target,
                    is_sink=True,
                    line=statement.line,
                    column=statement.column,
                    source=statement.source,
                )
            )
        else:  # pragma: no cover - defensive guard
            raise ParseError("Unknown statement in program")

    exports = [node.name for node in nodes.exports]
    seen_signatures = set()
    deduped: List[Equation] = []
    for eq in equations:
        signature = _equation_signature(eq)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(eq)
    return ProgramIR(equations=deduped, exports=exports)


def parse(program_str: str) -> ProgramIR:
    parser = _build_lark()
    lines = program_str.splitlines()
    try:
        tree = parser.parse(program_str)
    except UnexpectedInput as exc:
        line = exc.line or 1
        column = exc.column or 1
        line_text = ""
        if 1 <= line <= len(lines):
            line_text = lines[line - 1]
        raise ParseError(
            "Syntax error while parsing program",
            line=line,
            column=column,
            line_text=line_text,
        ) from exc
    except LarkError as exc:  # pragma: no cover - defensive
        raise ParseError(str(exc)) from exc

    transformer = FuseTransformer(program_str)
    nodes = transformer.transform(tree)
    return _build_program_ir(nodes)
