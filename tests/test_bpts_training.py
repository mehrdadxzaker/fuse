import numpy as np
import pytest

from fuse import Program, gradients_for_program
from tests._torch_utils import require_torch

torch = require_torch()
import torch.nn.functional as F  # noqa: E402


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _const_literal(array: np.ndarray) -> str:
    return repr(array.tolist())


def _build_program(example: dict, weights: np.ndarray) -> Program:
    features = np.asarray(example["features"], dtype=np.float32)
    edges = np.asarray(example["edges"], dtype=np.float32)
    target = float(example["target"])
    root = int(example["root"])
    w = np.asarray(weights, dtype=np.float32)

    root_mask = np.zeros(features.shape[0], dtype=np.float32)
    root_mask[root] = 1.0

    lines = [
        f"Feat[n,f] = const({_const_literal(features)})",
        f"Edge[n,m] = const({_const_literal(edges)})",
        f"W[f] = const({_const_literal(w)})",
        f"RootMask[n] = const({_const_literal(root_mask)})",
        f"Target = const({target})",
        "NodeLinear[n] = W[f] Feat[n,f]",
        "NodeScore[n] = gelu(NodeLinear[n])",
        "Message[n] = Edge[n,m] NodeScore[m]",
        "Total[n] = NodeScore[n]",
        "Total[n] = Message[n]",
        "RootScore = RootMask[n] Total[n]",
        "Error = RootScore",
        "ErrorSq = Error Error",
        f"Cross = Error const({-2.0 * target})",
        f"TargetSq = const({target * target})",
        "Loss = ErrorSq",
        "Loss = Cross",
        "Loss = TargetSq",
        "export Loss",
    ]
    return Program("\n".join(lines))


def _bpts_training(dataset, initial_weights, lr, epochs):
    weights = np.asarray(initial_weights, dtype=np.float32).copy()
    for _ in range(epochs):
        for example in dataset:
            program = _build_program(example, weights)
            grads, _ = gradients_for_program(
                program,
                seeds={"Loss": "const(1.0)"},
                grad_tensors=["W"],
            )
            weights = weights - lr * grads["W"]
    return weights


def _torch_training(dataset, initial_weights, lr, epochs):
    w = torch.tensor(initial_weights.tolist(), dtype=torch.float32, requires_grad=True)
    for _ in range(epochs):
        for example in dataset:
            features = torch.tensor(example["features"].tolist(), dtype=torch.float32)
            edges = torch.tensor(example["edges"].tolist(), dtype=torch.float32)
            target = torch.tensor(float(example["target"]), dtype=torch.float32)
            root = int(example["root"])

            node_linear = torch.matmul(features, w)
            node_score = _gelu_tanh(node_linear)
            message = torch.matmul(edges, node_score)
            total = node_score + message
            root_value = total[root]
            loss = (root_value - target) ** 2
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
            w.grad.zero_()
    return np.array(w.detach().cpu().to(torch.float32).tolist(), dtype=np.float32)


def test_bpts_matches_torch_baseline():
    dataset = [
        {
            "features": np.array(
                [[0.2, -0.1, 0.4], [0.1, 0.3, -0.4], [-0.1, 0.2, 0.1]],
                dtype=np.float32,
            ),
            "edges": np.array(
                [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
            ),
            "target": 0.5,
            "root": 0,
        },
        {
            "features": np.array(
                [[-0.3, 0.5, -0.2], [0.4, -0.1, 0.2]], dtype=np.float32
            ),
            "edges": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            "target": -0.1,
            "root": 0,
        },
        {
            "features": np.array(
                [[0.1, 0.3, -0.2], [0.2, -0.4, 0.1], [-0.3, 0.2, 0.4], [0.0, -0.1, 0.3]],
                dtype=np.float32,
            ),
            "edges": np.array(
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            "target": 0.2,
            "root": 0,
        },
    ]

    initial_weights = np.array([0.05, -0.1, 0.02], dtype=np.float32)
    learning_rate = 0.05
    epochs = 40

    bpts_weights = _bpts_training(dataset, initial_weights, learning_rate, epochs)
    torch_weights = _torch_training(dataset, initial_weights, learning_rate, epochs)

    np.testing.assert_allclose(bpts_weights, torch_weights, rtol=1e-4, atol=1e-5)
