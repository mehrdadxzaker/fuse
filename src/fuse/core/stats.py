from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


def _prod(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(max(1, value))
    return int(result)


def _parse_einsum_equation(
    equation: str,
    operand_shapes: Sequence[Sequence[int]],
) -> Tuple[Dict[str, int], List[str], List[str]]:
    if not equation:
        return {}, [], []
    if "->" in equation:
        input_part, output_part = equation.split("->", 1)
    else:
        input_part, output_part = equation, None
    inputs = input_part.split(",") if input_part else []
    if len(inputs) != len(operand_shapes):
        raise ValueError("Operand shape list does not match einsum inputs")

    counts: Counter[str] = Counter()
    dims: Dict[str, int] = {}

    for labels, shape in zip(inputs, operand_shapes):
        if len(labels) != len(shape):
            raise ValueError(f"Einsum label '{labels}' incompatible with shape {shape}")
        for label, dim in zip(labels, shape):
            dim = int(dim)
            if label in dims and dims[label] != dim:
                raise ValueError(f"Conflicting dimension for index '{label}'")
            dims[label] = dim
            counts[label] += 1

    if output_part is not None and len(output_part) > 0:
        output_labels = list(output_part)
    elif output_part == "":
        output_labels = []
    else:
        output_labels = []
        seen = set()
        for labels in inputs:
            for label in labels:
                if counts[label] == 1 and label not in seen:
                    seen.add(label)
                    output_labels.append(label)

    contracted = [label for label in dims.keys() if label not in output_labels]
    return dims, output_labels, contracted


def compute_einsum_stats(
    equation: str,
    operand_shapes: Sequence[Sequence[int]],
    operand_itemsizes: Sequence[int],
    result_shape: Sequence[int],
    result_itemsize: int,
) -> Dict[str, float]:
    if not equation:
        return {}

    dims, output_labels, contracted_labels = _parse_einsum_equation(
        equation,
        operand_shapes,
    )

    output_size = (
        _prod(dims[label] for label in output_labels)
        if output_labels
        else max(_prod(result_shape), 1)
    )
    contract_size = _prod(dims[label] for label in contracted_labels) if contracted_labels else 1

    if contracted_labels:
        flops = float(2 * output_size * contract_size)
        reductions = int(max(contract_size - 1, 0) * output_size)
    else:
        flops = float(output_size)
        reductions = 0

    bytes_in = 0
    for shape, itemsize in zip(operand_shapes, operand_itemsizes):
        bytes_in += _prod(shape) * int(itemsize)
    bytes_out = _prod(result_shape) * int(result_itemsize) if result_shape else int(result_itemsize)

    return {
        "flops": flops,
        "bytes_in": int(bytes_in),
        "bytes_out": int(bytes_out),
        "bytes_total": int(bytes_in + bytes_out),
        "contracted": contracted_labels,
        "output_indices": output_labels,
        "reductions": reductions,
    }
