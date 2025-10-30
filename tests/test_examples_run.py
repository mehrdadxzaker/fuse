from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import pytest

from fuse import Program

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Examples that currently exercise unsupported parser or shape features. We keep
# them in the parametrized list as expected failures so the suite tracks when
# those features land without causing CI regressions today.
KNOWN_FAILURES: dict[str, str] = {
    "01_attention_block.fuse": "prime-axis masking syntax not yet supported",
    "02_rules_and_embeddings.fuse": "demo assets use conflicting entity dimensions",
    "04_mlp.fuse": "bias broadcast rules need shape-checker update",
    "05_transformer_block.fuse": "feed-forward rotation lacks positional axis",
    "06_gnn_message_passing.fuse": "graph shift requires neighbourhood axis support",
    "08_rnn_unrolled.fuse": "prime-axis recurrence parsing incomplete",
    "03_zero_input_lm.fuse": "demo tensors mix incompatible vocabulary sizes",
    "09_logistic_regression.fuse": "shape checker rejects class-axis reduction",
    "10_hmm_forward.fuse": "parser lacks support for quote-style axes",
    "14_datalog_logic.fuse": "projection semantics pending shape-checker support",
    "15_kg_embeddings.fuse": "shape checker rejects broadcasted embedding axes",
    "17_neural_theorem_prover.fuse": "shape checker missing support for goal axis",
    "19_variational_autoencoder.fuse": "latent broadcast semantics not implemented",
    "26_physics_informed_nn.fuse": "parser lacks support for prime time-axis notation",
    "27_policy_reasoning.fuse": "shape checker rejects policy aggregation axes",
}


def _example_params(paths: Iterable[Path]) -> List[pytest.ParameterSet]:
    params: List[pytest.ParameterSet] = []
    for path in sorted(paths, key=lambda p: p.name):
        if path.name in KNOWN_FAILURES:
            params.append(
                pytest.param(
                    path,
                    marks=pytest.mark.xfail(
                        reason=KNOWN_FAILURES[path.name],
                        strict=False,
                    ),
                )
            )
        else:
            params.append(pytest.param(path))
    return params


@pytest.mark.parametrize(
    "program_path",
    _example_params(EXAMPLES_DIR.glob("*.fuse")),
)
def test_fuse_examples_compile_and_run(program_path: Path) -> None:
    # Execute from the examples directory so relative asset paths resolve
    cwd = os.getcwd()
    try:
        os.chdir(EXAMPLES_DIR)
        src = program_path.read_text(encoding="utf-8")
        prog = Program(src)
        runner = prog.compile(backend="numpy")
        outputs = runner()
        assert isinstance(outputs, dict)
    finally:
        os.chdir(cwd)
