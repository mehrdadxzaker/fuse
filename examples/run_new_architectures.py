import argparse
import os
from pathlib import Path
from typing import Optional

from fuse import Program

ARCHITECTURES = [
    ("04_mlp", ["Probs"]),
    ("05_transformer_block", ["Final"]),
    ("06_gnn_message_passing", ["Emb"]),
    ("07_cnn_1d", ["Act"]),
    ("08_rnn_unrolled", ["Hidden"]),
    ("09_logistic_regression", ["NBScore", "LRProb"]),
    ("10_hmm_forward", ["Normalized"]),
    ("11_factor_graph", ["ProbX", "ProbY", "ProbZ"]),
    ("12_gaussian_process", ["PredMean", "PredCov"]),
    ("13_svm_kernel", ["Decision", "Margin"]),
    ("14_datalog_logic", ["Ancestor", "Aunt"]),
    ("15_kg_embeddings", ["ScoreTrans", "ScoreDist"]),
    ("16_differentiable_logic", ["Conclusion"]),
    ("17_neural_theorem_prover", ["Confidence"]),
    ("18_policy_guard", ["Compliant", "BlockMask"]),
    ("19_variational_autoencoder", ["ReconProb", "KL"]),
    ("20_diffusion_process", ["Estimate"]),
    ("21_energy_based_model", ["Energy", "Reconstruct"]),
    ("22_rbm_energy", ["HiddenProb", "FreeEnergy"]),
    ("23_neuro_symbolic", ["Decision"]),
    ("24_prob_neural_fusion", ["NeuralRecon", "Normalized", "Consistency"]),
    ("25_graph_transformer_hybrid", ["Context"]),
    ("26_physics_informed_nn", ["Pred", "Residual"]),
    ("27_policy_reasoning", ["Distribution"]),
    ("28_tensorized_planner", ["State1"]),
    ("29_explainable_pipeline", ["Contribution", "Proba"]),
    ("30_llm_guardrail", ["Reasoned"]),
]


def run_example(name: str, exports, backend: str, cache_dir: Optional[str]):
    base = Path(__file__).resolve().parent
    os.chdir(base)
    eqs_path = base / f"{name}.fuse"
    program = Program(eqs_path.read_text())
    runner = program.compile(backend=backend, cache_dir=cache_dir)
    outputs = runner()
    summary = {
        key: None
        if outputs.get(key) is None
        else tuple(outputs[key].shape)
        if hasattr(outputs[key], "shape")
        else "scalar"
        for key in exports
    }
    print(f"{name}: {summary}")


def main():
    parser = argparse.ArgumentParser(description="Run the DSL architecture examples.")
    parser.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy", "torch", "jax"],
        help="Backend to use when compiling programs.",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Optional cache directory for compiled artifacts."
    )
    parser.add_argument(
        "--only", nargs="*", help="Subset of example prefixes to run (e.g. 04_mlp)."
    )
    args = parser.parse_args()

    selected = ARCHITECTURES
    if args.only:
        wanted = set(args.only)
        selected = [entry for entry in ARCHITECTURES if entry[0] in wanted]

    for name, exports in selected:
        run_example(name, exports, backend=args.backend, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
