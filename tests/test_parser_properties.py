import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fuse.core.ir import (
    FuncCall,
    IndexFunction,
    ProgramIR,
    TensorRef,
    Term,
    equation_index_summary,
    lhs_indices,
    rhs_indices,
)
from fuse.core.parser import parse


@dataclass
class EquationCase:
    program: str
    expected_equations: int
    projection: str
    lhs_name: str
    lhs_indices: Tuple[str, ...]
    is_source: bool = False
    is_sink: bool = False


_TENSOR_NAMES = [
    "A",
    "B",
    "C",
    "Weights",
    "Activ",
    "Tmp",
    "Proj",
    "Data",
]
_INDICES = ["i", "j", "k", "p", "q"]
_BUILTINS = ["gelu", "relu", "sig", "sin", "cos"]
_NO_NEWLINE_TOKENS = {"=", "+", "+=", "max=", "avg="}


@dataclass
class AxisToken:
    text: str
    axis: str


@dataclass
class Component:
    text: str
    axes: List[str]


_HYPOTHESIS_AXES = list(dict.fromkeys(_INDICES + ["t", "b", "n", "s"]))
_SOURCE_FILES = ["tensor0.npy", "tensor1.npy", "input.npy"]
_SINK_FILES = ["out0.npz", "result0.npz", "writeback.npz"]
_FUNC_NAMES = ["relu", "gelu", "sig", "sin", "cos", "sum", "max", "mean", "concat"]


def _random_name(rng: random.Random) -> str:
    base = rng.choice(_TENSOR_NAMES)
    suffix = rng.randint(0, 3)
    return f"{base}{suffix}" if suffix else base


def _random_indices(rng: random.Random, length: int) -> Tuple[str, ...]:
    return tuple(rng.choice(_INDICES) for _ in range(length))


def _build_matmul_case(rng: random.Random) -> EquationCase:
    lhs_name = _random_name(rng)
    lhs_indices = ("i", "j")
    shared_idx = "k"
    lhs = f"{lhs_name}[{','.join(lhs_indices)}]"
    left = f"{_random_name(rng)}[i,{shared_idx}]"
    right = f"{_random_name(rng)}[{shared_idx},j]"
    maybe_bias = ""
    expected_eqs = 1
    if rng.random() < 0.5:
        bias = f"{_random_name(rng)}[i,j]"
        maybe_bias = f" + {bias}"
        expected_eqs = 2
    program = f"{lhs} = {left} {right}{maybe_bias}"
    return EquationCase(
        program=program,
        expected_equations=expected_eqs,
        projection="sum",
        lhs_name=lhs_name,
        lhs_indices=lhs_indices,
    )


def _build_unary_case(rng: random.Random) -> EquationCase:
    lhs_name = _random_name(rng)
    rank = rng.choice([1, 2])
    lhs_indices = _random_indices(rng, rank)
    lhs_indices_str = ",".join(lhs_indices)
    fn = rng.choice(_BUILTINS)
    arg = f"{_random_name(rng)}[{lhs_indices_str}]"
    if rng.random() < 0.4 and rank > 0:
        axis = rng.choice(lhs_indices)
        program = f'{lhs_name}[{lhs_indices_str}] = {fn}({arg}, axis="{axis}")'
    else:
        program = f"{lhs_name}[{lhs_indices_str}] = {fn}({arg})"
    return EquationCase(
        program=program,
        expected_equations=1,
        projection="sum",
        lhs_name=lhs_name,
        lhs_indices=lhs_indices,
    )


def _build_source_case(rng: random.Random) -> EquationCase:
    lhs_name = _random_name(rng)
    rank = rng.choice([0, 1, 2])
    lhs_indices = _random_indices(rng, rank)
    indices_part = f"[{','.join(lhs_indices)}]" if lhs_indices else ""
    filename = f"tensor_{rng.randint(1, 5)}.npy"
    program = f'{lhs_name}{indices_part} = "{filename}"'
    return EquationCase(
        program=program,
        expected_equations=1,
        projection="sum",
        lhs_name=lhs_name,
        lhs_indices=lhs_indices,
        is_source=True,
    )


def _build_sink_case(rng: random.Random) -> EquationCase:
    tensor_name = _random_name(rng)
    rank = rng.choice([0, 1, 2])
    indices = _random_indices(rng, rank)
    ref = f"{tensor_name}[{','.join(indices)}]" if indices else tensor_name
    filename = f"out_{rng.randint(1, 3)}.npz"
    program = f'"{filename}" = {ref}'
    return EquationCase(
        program=program,
        expected_equations=1,
        projection="sum",
        lhs_name="__sink__",
        lhs_indices=(),
        is_sink=True,
    )


def _random_equation_case(seed: int) -> EquationCase:
    rng = random.Random(seed)
    choice = rng.random()
    if choice < 0.4:
        return _build_matmul_case(rng)
    if choice < 0.7:
        return _build_unary_case(rng)
    if choice < 0.85:
        return _build_source_case(rng)
    return _build_sink_case(rng)


def _normalize_expr(expr) -> Union[str, Tuple]:
    if expr is None:
        return None
    if isinstance(expr, TensorRef):
        return (
            "TensorRef",
            expr.name,
            tuple(expr.indices),
            tuple(expr.dotted_axes),
        )
    if isinstance(expr, Term):
        return ("Term", tuple(_normalize_expr(factor) for factor in expr.factors))
    if isinstance(expr, FuncCall):
        if isinstance(expr.arg, tuple):
            args = tuple(_normalize_expr(arg) for arg in expr.arg)
        elif expr.arg is None:
            args = ()
        else:
            args = (_normalize_expr(expr.arg),)
        kwargs = tuple(sorted((key, _normalize_value(value)) for key, value in expr.kwargs.items()))
        return ("FuncCall", expr.name, args, kwargs)
    if isinstance(expr, IndexFunction):
        return ("IndexFunction", expr.name, expr.axis)
    return _normalize_value(expr)


def _normalize_value(value):
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, list):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _normalize_value(val)) for key, val in value.items()))
    return repr(value)


def _normalize_program(ir: ProgramIR) -> Tuple[Tuple, Tuple[str, ...]]:
    equations: List[Tuple] = []
    for eq in ir.equations:
        lhs = eq.lhs
        entry = (
            eq.is_source,
            eq.is_sink,
            lhs.name,
            tuple(lhs.indices),
            tuple(lhs.dotted_axes),
            eq.projection,
            eq.src_file,
            eq.sink_file,
            _normalize_expr(eq.rhs),
        )
        equations.append(entry)
    return tuple(equations), tuple(sorted(ir.exports))


@pytest.mark.parametrize("seed", range(50))
def test_parser_accepts_random_equations(seed):
    case = _random_equation_case(seed)
    ir = parse(case.program)

    assert isinstance(ir, ProgramIR)
    assert len(ir.equations) == case.expected_equations

    for eq in ir.equations:
        assert eq.projection == case.projection
        if case.is_source:
            assert eq.is_source
            assert eq.src_file is not None
        elif case.is_sink:
            assert eq.is_sink
            assert eq.sink_file is not None
        else:
            assert eq.lhs.name == case.lhs_name
            assert tuple(eq.lhs.indices) == case.lhs_indices
            if case.expected_equations > 1:
                # For bias split we expect Term for the matmul part.
                assert isinstance(eq.rhs, (Term, TensorRef))


def test_parser_supports_trailing_commas():
    program = """
    A[i] = concat(B[i], axis="i",)
    Params = const({"bias": [0, 1,],},)
    """

    ir = parse(program)
    eq_map = {eq.lhs.name: eq for eq in ir.equations}

    concat_call = eq_map["A"].rhs
    assert isinstance(concat_call, FuncCall)
    assert concat_call.kwargs["axis"] == "i"

    const_call = eq_map["Params"].rhs
    assert isinstance(const_call, FuncCall)
    assert const_call.arg == {"bias": [0, 1]}


@pytest.mark.parametrize("seed", range(25))
def test_parser_whitespace_equivalence(seed):
    rng = random.Random(seed)
    case = _random_equation_case(seed + 103)
    tokens = case.program.replace("\n", " ").split()
    if not tokens:
        pytest.skip("Generated empty program string")
    parts: List[str] = []
    for idx, token in enumerate(tokens):
        parts.append(token)
        if idx < len(tokens) - 1:
            spaces = " " * rng.randint(1, 3)
            parts.append(spaces)
    variant = "".join(parts)

    base_ir = parse(case.program)
    variant_ir = parse(variant)

    assert _normalize_program(base_ir) == _normalize_program(variant_ir)


def _axis_token_strategy(
    *,
    allow_stream: bool = True,
    allow_slice: bool = True,
    allow_offset: bool = True,
    allow_dot: bool = True,
):
    @st.composite
    def _axis(draw):
        axis_symbol = draw(st.sampled_from(_HYPOTHESIS_AXES))
        kinds = ["plain"]
        if allow_offset:
            kinds.append("offset")
        if allow_slice:
            kinds.append("slice")
        if allow_stream:
            kinds.append("stream")
        kind = draw(st.sampled_from(kinds))
        dotted = False
        if allow_dot and kind in {"plain", "offset"}:
            dotted = draw(st.booleans())
        if kind == "plain":
            token_text = axis_symbol
            axis_name = axis_symbol
        elif kind == "offset":
            offset = draw(st.integers(-2, 2).filter(lambda value: value != 0))
            sign = "+" if offset > 0 else "-"
            token_text = f"{axis_symbol}{sign}{abs(offset)}"
            axis_name = axis_symbol
        elif kind == "slice":
            start = draw(st.integers(-2, 4))
            stop = draw(st.integers(-2, 4))
            if draw(st.booleans()):
                step = draw(st.integers(-3, 3).filter(lambda value: value != 0))
                axis_name = f"{start}:{stop}:{step}"
            else:
                axis_name = f"{start}:{stop}"
            token_text = axis_name
        else:
            offset = draw(st.integers(-2, 2))
            if offset > 0:
                offset_text = f"+{offset}"
            elif offset < 0:
                offset_text = str(offset)
            else:
                offset_text = ""
            token_text = f"*{axis_symbol}{offset_text}"
            axis_name = axis_symbol
        if dotted:
            token_text = f"{token_text}."
        return AxisToken(text=token_text, axis=axis_name)

    return _axis()


def _tensor_spec_strategy(
    *,
    min_rank: int = 0,
    max_rank: int = 3,
    allow_stream: bool = True,
    allow_slice: bool = True,
    allow_offset: bool = True,
    allow_dot: bool = True,
):
    axis_strategy = _axis_token_strategy(
        allow_stream=allow_stream,
        allow_slice=allow_slice,
        allow_offset=allow_offset,
        allow_dot=allow_dot,
    )

    @st.composite
    def _tensor(draw):
        name = draw(st.sampled_from(_TENSOR_NAMES))
        rank = draw(st.integers(min_rank, max_rank))
        tokens = [draw(axis_strategy) for _ in range(rank)]
        if tokens:
            indices = ",".join(token.text for token in tokens)
            text = f"{name}[{indices}]"
        else:
            text = name
        return Component(text=text, axes=[token.axis for token in tokens])

    return _tensor()


def _index_function_component_strategy():
    @st.composite
    def _index_fn(draw):
        fn_name = draw(st.sampled_from(["Even", "Odd"]))
        axis_symbol = draw(st.sampled_from(_HYPOTHESIS_AXES))
        if draw(st.booleans()):
            text = f'{fn_name}(axis="{axis_symbol}")'
        else:
            text = f"{fn_name}({axis_symbol})"
        return Component(text=text, axes=[axis_symbol])

    return _index_fn()


def _factor_component_strategy():
    return st.one_of(
        _tensor_spec_strategy(),
        _index_function_component_strategy(),
    )


def _term_component_strategy():
    factor_strategy = _factor_component_strategy()

    @st.composite
    def _term(draw):
        factor_count = draw(st.integers(1, 3))
        factors = [draw(factor_strategy) for _ in range(factor_count)]
        text = " ".join(factor.text for factor in factors)
        axes: List[str] = []
        for factor in factors:
            axes.extend(factor.axes)
        return Component(text=text, axes=axes)

    return _term()


def _func_call_component_strategy():
    tensor_arg = _tensor_spec_strategy()
    term_arg = _term_component_strategy()
    concat_arg = _tensor_spec_strategy()

    @st.composite
    def _func(draw):
        fn_name = draw(st.sampled_from(_FUNC_NAMES))
        args: List[Component]
        if fn_name == "concat":
            count = draw(st.integers(1, 3))
            args = [draw(concat_arg) for _ in range(count)]
        else:
            args = [draw(st.one_of(tensor_arg, term_arg))]
        axes: List[str] = []
        arg_texts: List[str] = []
        for arg in args:
            axes.extend(arg.axes)
            arg_texts.append(arg.text)
        kwargs: List[str] = []
        axis_choice: Optional[str] = None
        if fn_name in {"sum", "max", "mean"}:
            if draw(st.booleans()):
                axis_pool = axes or _HYPOTHESIS_AXES
                axis_choice = draw(st.sampled_from(axis_pool))
                kwargs.append(f'axis="{axis_choice}"')
        elif fn_name == "concat":
            if draw(st.booleans()):
                axis_choice = draw(st.sampled_from(_HYPOTHESIS_AXES))
                kwargs.append(f'axis="{axis_choice}"')
        args_payload = ", ".join(arg_texts)
        if kwargs:
            kw_text = ", ".join(kwargs)
            if args_payload:
                args_payload = f"{args_payload}, {kw_text}"
            else:
                args_payload = kw_text
        text = f"{fn_name}({args_payload})"
        axes_out = list(axes)
        if axis_choice is not None:
            axes_out.append(axis_choice)
        return Component(text=text, axes=axes_out)

    return _func()


def _expression_component_strategy():
    atom_strategy = st.one_of(_term_component_strategy(), _func_call_component_strategy())

    @st.composite
    def _expr(draw):
        count = draw(st.integers(1, 3))
        parts = [draw(atom_strategy) for _ in range(count)]
        text = " + ".join(part.text for part in parts)
        axes: List[str] = []
        for part in parts:
            axes.extend(part.axes)
        return Component(text=text, axes=axes)

    return _expr()


def _assignment_statement_strategy():
    lhs_strategy = _tensor_spec_strategy(
        min_rank=0,
        max_rank=3,
        allow_stream=True,
        allow_slice=True,
        allow_offset=True,
        allow_dot=True,
    )
    rhs_strategy = _expression_component_strategy()

    @st.composite
    def _stmt(draw):
        lhs = draw(lhs_strategy)
        rhs = draw(rhs_strategy)
        op = draw(st.sampled_from(["=", "+=", "max=", "avg="]))
        left_gap = " " * draw(st.integers(1, 3))
        right_gap = " " * draw(st.integers(1, 3))
        indent = draw(st.text(alphabet=" \t", min_size=0, max_size=2))
        text = f"{lhs.text}{left_gap}{op}{right_gap}{rhs.text}"
        return f"{indent}{text}"

    return _stmt()


def _source_statement_strategy():
    lhs_strategy = _tensor_spec_strategy(
        min_rank=0,
        max_rank=2,
        allow_stream=False,
        allow_slice=False,
        allow_offset=False,
        allow_dot=False,
    )

    @st.composite
    def _stmt(draw):
        lhs = draw(lhs_strategy)
        filename = draw(st.sampled_from(_SOURCE_FILES))
        indent = draw(st.text(alphabet=" \t", min_size=0, max_size=2))
        return f'{indent}{lhs.text} = "{filename}"'

    return _stmt()


def _sink_statement_strategy():
    value_strategy = _tensor_spec_strategy(
        min_rank=0,
        max_rank=3,
        allow_stream=True,
        allow_slice=True,
        allow_offset=True,
        allow_dot=False,
    )

    @st.composite
    def _stmt(draw):
        value = draw(value_strategy)
        filename = draw(st.sampled_from(_SINK_FILES))
        indent = draw(st.text(alphabet=" \t", min_size=0, max_size=2))
        return f'{indent}"{filename}" = {value.text}'

    return _stmt()


def _program_strategy():
    assignment_strategy = _assignment_statement_strategy()
    source_strategy = _source_statement_strategy()
    sink_strategy = _sink_statement_strategy()

    @st.composite
    def _program(draw):
        assignment_count = draw(st.integers(1, 4))
        statements = [draw(assignment_strategy) for _ in range(assignment_count)]
        if draw(st.booleans()):
            statements.insert(draw(st.integers(0, len(statements))), draw(source_strategy))
        if draw(st.booleans()):
            statements.insert(draw(st.integers(0, len(statements))), draw(sink_strategy))
        separator = draw(st.sampled_from(["\n", "\n\n"]))
        program = separator.join(statements)
        if draw(st.booleans()):
            program = program + "\n"
        if draw(st.booleans()):
            program = "\n" + program
        return program

    return _program()


@settings(max_examples=50, deadline=None)
@given(_program_strategy())
def test_parser_roundtrip_property(program_src):
    ir = parse(program_src)
    seen_sources = set()
    statements: List[str] = []
    for eq in ir.equations:
        if eq.source and eq.source not in seen_sources:
            seen_sources.add(eq.source)
            statements.append(eq.source)
    roundtrip_src = "\n".join(statements)
    roundtrip_ir = parse(roundtrip_src)
    assert _normalize_program(ir) == _normalize_program(roundtrip_ir)


@settings(max_examples=50, deadline=None)
@given(_program_strategy())
def test_equation_index_summary_consistency(program_src):
    ir = parse(program_src)
    for eq in ir.equations:
        if eq.is_source or eq.is_sink or eq.rhs is None:
            continue
        lhs = lhs_indices(eq)
        rhs = rhs_indices(eq)
        projected: List[str] = []
        for axis in rhs:
            if axis not in lhs and axis not in projected:
                projected.append(axis)
        summary = equation_index_summary(eq, projected)
        assert summary["lhs"] == lhs
        assert set(summary["projected"]) == set(projected)
        rhs_only = set(rhs) - set(lhs)
        assert set(summary["rhs_only"]) == rhs_only
        assert set(summary["dangling"]) == rhs_only - set(summary["projected"])
