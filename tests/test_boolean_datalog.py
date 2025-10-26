import textwrap

import numpy as np
import pytest

from fuse import ExecutionConfig, Program
from tests._torch_utils import require_torch


def _write_facts(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(str(int(x)) for x in row) + "\n")


def _reference_closure(parent_edges, sister_edges):
    parents = set(parent_edges)
    ancestors = set(parent_edges)
    changed = True
    while changed:
        changed = False
        new_edges = set()
        for (x, y) in parent_edges:
            for (mid, z) in ancestors:
                if y == mid and (x, z) not in ancestors:
                    new_edges.add((x, z))
        if new_edges:
            ancestors.update(new_edges)
            changed = True
    aunts = {
        (sister, child)
        for (sister, sibling) in sister_edges
        for (parent, child) in parent_edges
        if sibling == parent
    }
    return ancestors, aunts


def _array_pairs(arr):
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        arr = arr.detach().cpu()
        try:
            import torch

            arr = arr.to(dtype=torch.float32)
        except Exception:
            pass
        arr = np.array(arr.tolist())
    else:
        arr = np.asarray(arr)
    return {tuple(map(int, coord)) for coord in np.argwhere(arr > 0)}


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_parent_aunt_closure_matches_reference(tmp_path, backend):
    if backend == "torch":
        require_torch()
    if backend == "jax":
        pytest.importorskip("jax")
    parent_edges = [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (4, 4),
    ]
    sister_edges = [
        (5, 1),
        (6, 2),
        (0, 4),
    ]

    parent_path = tmp_path / "parent.tsv"
    sister_path = tmp_path / "sister.tsv"

    _write_facts(parent_path, parent_edges)
    _write_facts(sister_path, sister_edges)

    prog_src = textwrap.dedent(
        f"""
        Parent(x, y) = "{parent_path.as_posix()}"
        Sister(x, y) = "{sister_path.as_posix()}"
        Ancestor(x, z) = Parent(x, z)
        Ancestor(x, z) = Parent(x, y) Ancestor(y, z)
        Aunt(x, z) = Sister(x, y) Parent(y, z)
        export Ancestor
        export Aunt
        """
    ).strip()

    program = Program(prog_src)
    cfg = ExecutionConfig(mode="fixpoint", max_iters=32)
    runner = program.compile(backend=backend, config=cfg)
    outputs = runner()

    ancestor_pairs = _array_pairs(outputs["Ancestor"])
    aunt_pairs = _array_pairs(outputs["Aunt"])

    expected_ancestors, expected_aunts = _reference_closure(parent_edges, sister_edges)

    assert ancestor_pairs == expected_ancestors
    assert aunt_pairs == expected_aunts


def test_demand_query_aunt_row_materializes_minimum(tmp_path):
    parent_edges = [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (4, 4),
    ]
    sister_edges = [
        (5, 1),
        (6, 2),
        (0, 4),
    ]

    parent_path = tmp_path / "parent.tsv"
    sister_path = tmp_path / "sister.tsv"

    _write_facts(parent_path, parent_edges)
    _write_facts(sister_path, sister_edges)

    prog_src = textwrap.dedent(
        f"""
        Parent(x, y) = "{parent_path.as_posix()}"
        Sister(x, y) = "{sister_path.as_posix()}"
        Ancestor(x, z) = Parent(x, z)
        Ancestor(x, z) = Parent(x, y) Ancestor(y, z)
        Aunt(x, z) = Sister(x, y) Parent(y, z)
        export Ancestor
        export Aunt
        """
    ).strip()

    program = Program(prog_src)
    forward_cfg = ExecutionConfig(mode="fixpoint", max_iters=32)
    forward_runner = program.compile(config=forward_cfg)
    expected_aunt = forward_runner()["Aunt"]

    demand_cfg = ExecutionConfig(mode="demand", max_iters=32)
    demand_runner = program.compile(config=demand_cfg)

    demand_runner.logs.clear()
    target_row = 5
    demand_row = demand_runner.query("Aunt", {"x": target_row})
    np.testing.assert_array_equal(demand_row, expected_aunt[target_row])

    ancestor_logs = [
        entry
        for entry in demand_runner.logs
        if entry.get("kind") == "equation" and entry["equation"]["name"] == "Ancestor"
    ]
    assert not ancestor_logs

    demand_runner.logs.clear()
    cached_row = demand_runner.query("Aunt", {"x": target_row})
    np.testing.assert_array_equal(cached_row, expected_aunt[target_row])
    assert not demand_runner.logs

    full_aunt = demand_runner.query("Aunt")
    np.testing.assert_array_equal(full_aunt, expected_aunt)
