import numpy as np

from fuse import (
    ExecutionConfig,
    TreeFactorGraph,
    Variable,
    Factor,
    conditional_probability,
)
from fuse.inference.tree_program import brute_force_joint


def test_tree_program_matches_bruteforce_conditional():
    variables = [
        Variable("A", 2),
        Variable("B", 2),
        Variable("C", 2),
    ]

    rng = np.random.default_rng(0)
    factors = [
        Factor("f_ab", ("A", "B"), rng.random((2, 2), dtype=np.float64)),
        Factor("f_bc", ("B", "C"), rng.random((2, 2), dtype=np.float64)),
    ]

    graph = TreeFactorGraph(variables, factors)
    tree_prog = graph.build_program(query_vars=("A",), evidence={"C": 1})
    runner = tree_prog.compile(config=ExecutionConfig(mode="single", chaining="backward"))
    outputs = runner()

    # Normalize the full query distribution
    qe = np.asarray(outputs["ProgQE"], dtype=np.float64)
    pe = float(np.asarray(outputs["ProgE"], dtype=np.float64))
    conditional = qe / pe
    brute_joint = brute_force_joint(factors, variables, evidence={"C": 1})
    var_index = {var.name: idx for idx, var in enumerate(variables)}
    slice_c = brute_joint[..., 1]
    marginal_a = slice_c.sum(axis=var_index["B"])
    marginal_total = marginal_a.sum()
    brute = marginal_a / marginal_total
    np.testing.assert_allclose(conditional, brute, rtol=1e-6, atol=1e-6)

    # helper-based computation
    prob = conditional_probability(outputs, tree_prog.query_vars, {"A": 0})
    assert np.isclose(prob, brute[0])


def test_tree_program_multi_variable_query():
    variables = [
        Variable("A", 2),
        Variable("B", 3),
        Variable("C", 2),
        Variable("D", 2),
    ]
    rng = np.random.default_rng(42)
    factors = [
        Factor("f_ab", ("A", "B"), rng.random((2, 3), dtype=np.float64)),
        Factor("f_bc", ("B", "C"), rng.random((3, 2), dtype=np.float64)),
        Factor("f_cd", ("C", "D"), rng.random((2, 2), dtype=np.float64)),
    ]

    graph = TreeFactorGraph(variables, factors)
    tree_prog = graph.build_program(query_vars=("A", "B"), evidence={"D": 1})
    runner = tree_prog.compile(config=ExecutionConfig(mode="single", chaining="backward"))
    outputs = runner()

    qe = np.asarray(outputs["ProgQE"], dtype=np.float64)
    pe = float(np.asarray(outputs["ProgE"], dtype=np.float64))
    cond = qe / pe

    brute_joint = brute_force_joint(factors, variables, evidence={"D": 1})
    idx = {var.name: pos for pos, var in enumerate(variables)}
    # Conditioning on D = 1
    slice_d = brute_joint.take(indices=1, axis=idx["D"])
    # Arrange axes to (A,B,C) -> sum out C
    brute = slice_d.sum(axis=idx["C"])
    brute = brute / brute.sum()
    np.testing.assert_allclose(cond, brute, rtol=1e-6, atol=1e-6)


def test_tree_program_rejects_cycle():
    variables = [
        Variable("A", 2),
        Variable("B", 2),
        Variable("C", 2),
    ]
    rng = np.random.default_rng(1)
    factors = [
        Factor("f_ab", ("A", "B"), rng.random((2, 2))),
        Factor("f_bc", ("B", "C"), rng.random((2, 2))),
        Factor("f_ca", ("C", "A"), rng.random((2, 2))),
    ]
    try:
        TreeFactorGraph(variables, factors)
    except ValueError as exc:
        assert "tree" in str(exc).lower()
    else:
        raise AssertionError("Expected TreeFactorGraph to reject cyclic graph")
