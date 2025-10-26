import ast
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from .ir import (
    TensorRef,
    Term,
    FuncCall,
    Equation,
    ProgramIR,
    IndexSpec,
    SliceSpec,
    IndexFunction,
)
from .exceptions import ParseError

# Very small, line-oriented parser. It supports:
# - T[i,j] = A[i,k] B[k,j]
# - Y[i.] = softmax(X[i])          ('.' marks axis on LHS for softmax/lnorm)
# - T[i,j] = "file.npy"            (source)
# - "out.npz" = T[i,j]             (sink)
# - NAME = const(1.0)
# - builtins: step, relu, sig, gelu, softmax, lnorm, rope, concat, causal_mask, topk, const, tucker_dense
#
# Multiple terms added with '+' are turned into separate equations with the same LHS.
# Boolean tensors with parentheses T(i,j) are accepted (treated same as brackets here).

BUILTIN_FUNCS = {
    "step",
    "relu",
    "sig",
    "gelu",
    "softmax",
    "lnorm",
    "layernorm",
    "masked_softmax",
    "attention",
    "rope",
    "concat",
    "causal_mask",
    "topk",
    "const",
    "reduce_max",
    "reduce_mean",
    "sin",
    "cos",
    "case",
    "tucker_dense",
}

INDEX_FUNCTION_NAMES = {
    "even",
    "odd",
}

ASSIGN_RE = re.compile(r"^(?P<lhs>.+?)\s*(?P<op>\+=|max=|avg=|=)\s*(?P<rhs>.+)$")
PROJECTION_MAP = {
    "=": "sum",
    "+=": "sum",
    "max=": "max",
    "avg=": "mean",
}


@dataclass
class Statement:
    text: str
    line: int

    def column_of(self, substring: str, *, start: int = 0) -> int:
        if not substring:
            return 1
        idx = self.text.find(substring, start)
        if idx == -1:
            idx = self.text.find(substring.strip(), start)
        return idx + 1 if idx != -1 else 1


def _raise_parse_error(statement: Statement, message: str, column: int = 1) -> None:
    raise ParseError(message, line=statement.line, column=column, line_text=statement.text)

def parse(program_str: str) -> ProgramIR:
    statements: List[Statement] = []
    pending_parts: List[str] = []
    pending_start_line: Optional[int] = None
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    def _flush_pending() -> None:
        nonlocal pending_start_line, paren_depth, bracket_depth, brace_depth
        if not pending_parts or pending_start_line is None:
            return
        statements.append(Statement(text=" ".join(pending_parts), line=pending_start_line))
        pending_parts.clear()
        pending_start_line = None
        paren_depth = bracket_depth = brace_depth = 0

    eqs: List[Equation] = []
    exports: List[str] = []

    for line_no, raw in enumerate(program_str.splitlines(), start=1):
        stripped = raw.split('#', 1)[0]
        line = stripped.strip()
        if not line:
            continue
        if pending_start_line is None:
            pending_start_line = line_no
        pending_parts.append(line)
        paren_depth += line.count('(') - line.count(')')
        bracket_depth += line.count('[') - line.count(']')
        brace_depth += line.count('{') - line.count('}')

        if paren_depth <= 0 and bracket_depth <= 0 and brace_depth <= 0:
            _flush_pending()

    if pending_parts:
        _flush_pending()

    for statement in statements:
        line = statement.text
        if line.lower().startswith('export '):
            name = line.split(None,1)[1].strip()
            exports.append(name)
            continue

        # sink: "file" = RHS
        if line.startswith('"'):
            if '=' not in line:
                _raise_parse_error(statement, "Sink assignment must include '='", 1)
            file, rhs = line.split('=',1)
            file = file.strip().strip('"')
            rhs_column = statement.column_of(rhs.strip())
            rhs_expr = _parse_expr(rhs.strip(), statement, rhs_column)
            # Create a fake lhs ref with name from RHS if it's a TensorRef
            # We'll store sink on equation for later execution
            dummy_lhs = TensorRef(name="__sink__", indices=[], dotted_axes=[])
            eqs.append(
                Equation(
                    lhs=dummy_lhs,
                    rhs=rhs_expr,
                    sink_file=file,
                    is_sink=True,
                    line=statement.line,
                    column=1,
                    source=statement.text,
                )
            )
            continue

        match = ASSIGN_RE.match(line)
        if not match:
            _raise_parse_error(statement, f"Cannot parse line: {line}", 1)
        lhs_str = match.group("lhs").strip()
        op = match.group("op")
        rhs_str = match.group("rhs").strip()
        projection = PROJECTION_MAP.get(op)
        if projection is None:
            op_column = statement.column_of(op)
            _raise_parse_error(statement, f"Unsupported assignment operator '{op}'", op_column)

        # source: T[idx] = "file"
        if rhs_str.startswith('"') and rhs_str.endswith('"'):
            if op != "=":
                op_column = statement.column_of(op)
                _raise_parse_error(statement, "Sources must use '=' assignment", op_column)
            file = rhs_str.strip().strip('"')
            lhs_col = statement.column_of(lhs_str)
            lhs_ref = _parse_tensor_ref(lhs_str, statement, lhs_col)
            eqs.append(
                Equation(
                    lhs=lhs_ref,
                    rhs=None,
                    src_file=file,
                    is_source=True,
                    line=statement.line,
                    column=lhs_col,
                    source=statement.text,
                )
            )
            continue

        # split '+' into separate equations (implicit add on same LHS)
        rhs_parts = split_top_level_plus(rhs_str)

        search_pos = 0
        for part in rhs_parts:
            part_strip = part.strip()
            part_col = statement.column_of(part_strip, start=search_pos)
            rhs_expr = _parse_expr(part_strip, statement, part_col)
            search_pos = statement.text.find(part_strip, search_pos)
            if search_pos == -1:
                search_pos = len(statement.text)
            else:
                search_pos += len(part_strip)
            lhs_col = statement.column_of(lhs_str)
            lhs_ref = _parse_tensor_ref(lhs_str, statement, lhs_col)
            eqs.append(
                Equation(
                    lhs=lhs_ref,
                    rhs=rhs_expr,
                    projection=projection,
                    line=statement.line,
                    column=lhs_col,
                    source=statement.text,
                )
            )

    return ProgramIR(equations=eqs, exports=exports)

def split_top_level_plus(s: str) -> List[str]:
    parts = []
    depth = 0
    bracket_depth = 0
    brace_depth = 0
    last = 0
    for i,ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
        elif ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
        elif ch == '+' and depth == 0 and bracket_depth == 0 and brace_depth == 0:
            parts.append(s[last:i])
            last = i+1
    parts.append(s[last:])
    return parts

def _parse_tensor_ref(s: str, statement: Statement, base_column: int) -> TensorRef:
    # Accept T[i,j] or T(i,j); allow dotted indices like p'. in LHS
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*([\[\(])\s*([^\]\)]*)[\]\)]$', s)
    if not m:
        # bare name (scalar)
        name = s.strip()
        return TensorRef(name=name, indices=[], dotted_axes=[])
    name, bracket_char, idxs = m.groups()
    raw_indices = [tok for tok in idxs.split(',')] if idxs.strip() else []
    dotted: List[str] = []
    clean: List[str] = []
    rolling: Dict[str, int] = {}
    specs: List[IndexSpec] = []
    search_pos = 0
    for raw_idx in raw_indices:
        idx = raw_idx
        token = idx.strip()
        local_offset = s.find(raw_idx, search_pos)
        if local_offset == -1:
            local_offset = s.find(token, search_pos)
        if local_offset == -1:
            local_offset = search_pos
        search_pos = local_offset + len(raw_idx)
        token_column = base_column + local_offset
        dotted_flag = token.endswith('.')
        if dotted_flag:
            token = token[:-1].strip()
        slice_spec: Optional[SliceSpec] = None
        offset = 0
        if token.startswith('*'):
            m = re.match(r'^\*([A-Za-z_][A-Za-z0-9_]*)([+-]\d+)?$', token)
            if not m:
                _raise_parse_error(
                    statement,
                    f"Invalid streaming index syntax: {idx}",
                    token_column,
                )
            base = m.group(1)
            offset = int(m.group(2) or 0)
            rolling[base] = offset
            token = base
        axis_name = token
        if ':' in token:
            normalized = ":".join(part.strip() for part in token.split(':'))
            bounds = [part.strip() for part in token.split(':')]
            if len(bounds) not in (2, 3):
                _raise_parse_error(statement, f"Invalid slice specifier: {idx}", token_column)
            start = _parse_slice_bound(bounds[0], statement, token_column)
            stop = _parse_slice_bound(bounds[1], statement, token_column)
            step = _parse_slice_bound(bounds[2], statement, token_column) if len(bounds) == 3 else None
            slice_spec = SliceSpec(start=start, stop=stop, step=step)
            axis_name = normalized
            offset = 0
        else:
            match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)([+-]\d+)?$', token)
            if match:
                axis_name = match.group(1)
                offset = int(match.group(2) or 0)
            else:
                axis_name = token
        if dotted_flag:
            dotted.append(axis_name)
        clean.append(axis_name)
        specs.append(IndexSpec(axis=axis_name, offset=offset, slice=slice_spec))
    return TensorRef(
        name=name,
        indices=clean,
        dotted_axes=dotted,
        rolling=rolling,
        index_specs=specs,
        is_paren=(bracket_char == '('),
    )

def _parse_slice_bound(raw: str, statement: Statement, column: int) -> Optional[int]:
    value = raw.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ParseError(
            f"Slice bounds must be integers: {raw}",
            line=statement.line,
            column=column,
            line_text=statement.text,
        ) from exc

def _extract_axis_symbol(expr: Any, statement: Statement, column: int) -> str:
    if isinstance(expr, TensorRef):
        if expr.indices or expr.dotted_axes or expr.rolling:
            _raise_parse_error(statement, "Index functions require a scalar axis symbol", column)
        return expr.name
    if isinstance(expr, str):
        return expr
    _raise_parse_error(statement, f"Unsupported axis specification for index function: {expr!r}", column)

def _parse_product(s: str, statement: Statement, base_column: int):
    # factors separated by spaces, each factor may itself be an expression
    tokens = top_level_tokens(s, sep=' ')
    factors = []
    search_pos = 0
    for tok in tokens:
        raw_tok = tok
        tok_strip = raw_tok.strip()
        if not tok_strip:
            continue
        local_offset = s.find(raw_tok, search_pos)
        if local_offset == -1:
            local_offset = s.find(tok_strip, search_pos)
        if local_offset == -1:
            local_offset = search_pos
        search_pos = local_offset + len(raw_tok)
        column = base_column + local_offset
        factors.append(_parse_expr(tok_strip, statement, column))
    return Term(factors=factors)

def _parse_funccall(s: str, statement: Statement, base_column: int) -> Any:
    name, arg = s.split('(',1)
    name = name.strip()
    arg = arg.rstrip(')').strip()
    lower_name = name.lower()
    if not arg:
        if lower_name in INDEX_FUNCTION_NAMES:
            _raise_parse_error(statement, f"{name} requires an axis argument", base_column)
        return FuncCall(name=name, arg=None, kwargs={})

    tokens = top_level_tokens(arg, sep=',')
    args_info: List[Tuple[Any, int]] = []
    kwargs_info: Dict[str, Tuple[Any, int]] = {}

    search_pos = 0
    for tok in tokens:
        raw_tok = tok
        tok_strip = raw_tok.strip()
        if not tok_strip:
            continue
        local_offset = arg.find(raw_tok, search_pos)
        if local_offset == -1:
            local_offset = arg.find(tok_strip, search_pos)
        if local_offset == -1:
            local_offset = search_pos
        token_column = base_column + s.find('(', 0) + 1 + local_offset
        key, value = _maybe_split_kwarg(tok_strip)
        if key is None:
            parsed_arg = _parse_expr(tok_strip, statement, token_column)
            args_info.append((parsed_arg, token_column))
        else:
            parsed_value = _parse_kwarg_value(value, statement, token_column)
            kwargs_info[key] = (parsed_value, token_column)
        search_pos = local_offset + len(raw_tok)

    args: List[Any] = [value for value, _ in args_info]

    if lower_name in INDEX_FUNCTION_NAMES:
        axis_value: Any = None
        axis_column: int = base_column
        if args:
            if len(args) > 1:
                _raise_parse_error(statement, f"{name} accepts at most one positional argument", base_column)
            axis_value, axis_column = args_info[0]
        if "axis" in kwargs_info:
            if axis_value is not None:
                _raise_parse_error(statement, f"{name} axis specified twice", base_column)
            axis_value, axis_column = kwargs_info.pop("axis")
            kwargs = {key: value for key, (value, _) in kwargs_info.items()}
        if kwargs_info:
            unexpected = ", ".join(sorted(kwargs_info))
            _raise_parse_error(statement, f"Unexpected kwargs for {name}: {unexpected}", base_column)
        if axis_value is None:
            _raise_parse_error(statement, f"{name} requires an axis argument", base_column)
        axis_name = _extract_axis_symbol(axis_value, statement, axis_column)
        return IndexFunction(name=lower_name, axis=axis_name)

    kwargs_clean: Dict[str, Any] = {key: value for key, (value, _) in kwargs_info.items()}

    if len(args) == 0:
        arg_value: Any = None
    elif len(args) == 1:
        arg_value = args[0]
    else:
        arg_value = tuple(args)

    return FuncCall(name=name, arg=arg_value, kwargs=kwargs_clean)

def _parse_expr(s: str, statement: Statement, base_column: int):
    s = s.strip()
    # function?
    func_match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\(', s)
    if func_match:
        name = func_match.group(1)
        lower = name.lower()
        if '[' in s or lower in BUILTIN_FUNCS or lower in INDEX_FUNCTION_NAMES:
            return _parse_funccall(s, statement, base_column)
    # literal number/bool/quoted string
    literal = _parse_literal(s)
    if literal is not None:
        return literal
    # product?
    if ' ' in s:
        tokens = [tok.strip() for tok in top_level_tokens(s, sep=' ') if tok.strip()]
        if len(tokens) > 1:
            return _parse_product(s, statement, base_column)
    # single token -> tensor ref
    return _parse_tensor_ref(s, statement, base_column)

def top_level_tokens(s: str, sep=' ') -> List[str]:
    parts = []
    depth = 0
    bracket_depth = 0
    last = 0
    for i,ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
        elif ch == sep and depth == 0 and bracket_depth == 0:
            parts.append(s[last:i])
            last = i+1
    parts.append(s[last:])
    return parts

def _maybe_split_kwarg(token: str) -> Tuple[Optional[str], str]:
    depth = 0
    for i,ch in enumerate(token):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '=' and depth == 0:
            key = token[:i].strip()
            value = token[i+1:].strip()
            return key, value
    return None, token

def _parse_literal(token: str) -> Optional[Any]:
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        return token[1:-1]
    if token.lower() in ("true", "false"):
        return token.lower() == "true"
    num_match = re.fullmatch(r'[+-]?\d+', token)
    if num_match:
        try:
            return int(token)
        except ValueError:
            pass
    float_match = re.fullmatch(r'[+-]?\d*\.\d+(e[+-]?\d+)?', token, flags=re.IGNORECASE)
    if float_match:
        try:
            return float(token)
        except ValueError:
            pass
    if token.startswith('[') or token.startswith('{'):
        try:
            return ast.literal_eval(token)
        except (SyntaxError, ValueError):
            pass
    return None

def _parse_kwarg_value(raw: str, statement: Statement, column: int) -> Any:
    literal = _parse_literal(raw)
    if literal is not None:
        return literal
    expr = _parse_expr(raw, statement, column)
    if isinstance(expr, TensorRef) and not expr.indices and not expr.dotted_axes:
        return expr.name
    return expr
    if isinstance(expr, TensorRef) and not expr.indices and not expr.dotted_axes:
        # Treat bare identifiers in kwargs as symbols (e.g., axis=k)
        return expr.name
    return expr
