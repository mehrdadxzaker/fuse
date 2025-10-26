import random
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pytest

from fuse.core.ir import FuncCall, IndexFunction, ProgramIR, TensorRef, Term
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
    rng = np.random.default_rng(seed)
    case = _random_equation_case(seed + 103)
    tokens = case.program.replace("\n", " ").split()
    if not tokens:
        pytest.skip("Generated empty program string")
    parts: List[str] = []
    for idx, token in enumerate(tokens):
        parts.append(token)
        if idx < len(tokens) - 1:
            spaces = " " * int(rng.integers(1, 4))
            parts.append(spaces)
    variant = "".join(parts)

    base_ir = parse(case.program)
    variant_ir = parse(variant)

    assert _normalize_program(base_ir) == _normalize_program(variant_ir)
