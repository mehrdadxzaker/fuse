import numpy as np

from fuse import ExecutionConfig, InMemoryWeightStore, Program, RuntimePolicies


GRID_MRF_PROGRAM = """
Grid[r, c, u, d] = InputGrid[r, c, u, d]
PairWeight[u, d] = InputWeight[u, d]
Marginal[r, c] avg= Grid[r, c, u, d] PairWeight[u, d]
export Marginal
""".strip()


def _compile_with_grid(grid, config: ExecutionConfig):
    grid = np.asarray(grid)
    weight = np.ones(grid.shape[2:], dtype=grid.dtype)
    policies = RuntimePolicies(
        weight_store=InMemoryWeightStore(
            {"InputGrid": grid, "InputWeight": weight}
        )
    )
    program = Program(GRID_MRF_PROGRAM)
    return program.compile(config=config, policies=policies)


def test_monte_carlo_projection_matches_exact_when_sampling_all():
    rng = np.random.default_rng(0)
    grid = rng.random((3, 3, 4, 4), dtype=np.float32)
    exact_cfg = ExecutionConfig(mode="single")
    exact_runner = _compile_with_grid(grid, exact_cfg)
    exact = exact_runner()["Marginal"]

    total_states = grid.shape[2] * grid.shape[3]
    mc_cfg = ExecutionConfig(
        mode="single",
        projection_strategy="monte_carlo",
        projection_samples=total_states,
        projection_seed=123,
    )
    mc_runner = _compile_with_grid(grid, mc_cfg)
    approx = mc_runner()["Marginal"]
    np.testing.assert_allclose(approx, exact)


def test_monte_carlo_projection_converges_on_grid_mrf():
    rng = np.random.default_rng(13)
    grid = rng.random((4, 4, 20, 20), dtype=np.float32)

    exact_cfg = ExecutionConfig(mode="single")
    exact_runner = _compile_with_grid(grid, exact_cfg)
    exact = exact_runner()["Marginal"]

    sample_counts = [10, 80, 320]
    errors = []
    for count in sample_counts:
        cfg = ExecutionConfig(
            mode="single",
            projection_strategy="monte_carlo",
            projection_samples=count,
            projection_seed=2024,
        )
        runner = _compile_with_grid(grid, cfg)
        approx = runner()["Marginal"]
        errors.append(np.max(np.abs(approx - exact)))

    assert errors[-1] < 5e-2
    assert errors[0] >= errors[1] >= errors[2]
