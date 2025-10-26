import numpy as np

from fuse import logic, nn, pgm
from fuse.core.policies import InMemoryWeightStore, RuntimePolicies


def _run(program, weights):
    policies = RuntimePolicies(weight_store=InMemoryWeightStore(weights))
    runner = program.compile(policies=policies)
    return runner()


def test_transformer_encoder_template_numpy():
    prog = nn.transformer_encoder()
    batch, positions, dim = 1, 3, 4
    heads, dk, dv, ff, classes = 2, 2, 3, 5, 3

    weights = {
        "InputStream": np.random.randn(batch, positions, dim).astype(np.float32),
        "PositionalEncoding": np.zeros((batch, positions, dim), dtype=np.float32),
        "QueryWeight": np.random.randn(heads, dk, dim).astype(np.float32),
        "KeyWeight": np.random.randn(heads, dk, dim).astype(np.float32),
        "ValueWeight": np.random.randn(heads, dv, dim).astype(np.float32),
        "AttentionScale": np.array(1.0 / np.sqrt(dk), dtype=np.float32),
        "AttentionMask": np.zeros((batch, heads, positions, positions), dtype=np.float32),
        "OutputProj": np.random.randn(dim, heads, dv).astype(np.float32),
        "MlpWeight1": np.random.randn(ff, dim).astype(np.float32),
        "MlpWeight2": np.random.randn(dim, ff).astype(np.float32),
        "ReadoutWeight": np.random.randn(classes, dim).astype(np.float32),
    }

    outputs = _run(prog, weights)
    assert "StreamOut" in outputs
    assert "Y" in outputs
    stream = outputs["StreamOut"]
    logits = outputs["Y"]
    assert stream.shape == (batch, positions, dim)
    assert logits.shape == (batch, positions, classes)


def test_graph_message_passing_template_numpy():
    layers, nodes, dim, classes = 2, 4, 3, 2
    prog = nn.graph_message_passing(num_layers=layers)

    weights = {
        "Adjacency": np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
        "NodeFeatures": np.random.randn(nodes, dim).astype(np.float32),
        "NodeClassifier": np.random.randn(classes, dim).astype(np.float32),
        "GraphClassifier": np.random.randn(classes, dim).astype(np.float32),
        "GraphPooling": np.ones((nodes,), dtype=np.float32),
    }

    for layer in range(layers):
        weights[f"MessageWeight{layer}"] = np.random.randn(dim, dim).astype(np.float32)
        weights[f"AggWeight{layer}"] = np.random.randn(dim, dim).astype(np.float32)
        weights[f"SelfWeight{layer}"] = np.random.randn(dim, dim).astype(np.float32)

    outputs = _run(prog, weights)
    assert outputs["Embeddings"].shape == (nodes, dim)
    assert outputs["NodeProbs"].shape == (nodes, classes)
    assert outputs["EdgeProb"].shape == (nodes, nodes)
    assert outputs["GraphProb"].shape == (classes,)


def test_aunt_analogy_logic_template():
    prog = logic.aunt_analogy()
    num_entities, emb_dim = 3, 4

    weights = {
        "SisterFacts": np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "ParentFacts": np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "ObjectEmbeddings": np.random.randn(num_entities, emb_dim).astype(np.float32),
    }

    outputs = _run(prog, weights)
    assert outputs["Aunt"].shape == (num_entities, num_entities)
    assert outputs["Analogical"].shape == (num_entities, num_entities)


def test_factor_graph_template():
    prog = pgm.factor_graph_triplet()

    weights = {
        "UnaryXPotential": np.array([0.6, 0.4], dtype=np.float32),
        "UnaryYPotential": np.array([0.5, 0.5], dtype=np.float32),
        "UnaryZPotential": np.array([0.7, 0.3], dtype=np.float32),
        "PairXYPotential": np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        "PairYZPotential": np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32),
    }

    outputs = _run(prog, weights)
    assert outputs["Joint"].shape == (2, 2, 2)
    assert outputs["ProbX"].shape == (2,)
    assert outputs["ProbY"].shape == (2,)
    assert outputs["ProbZ"].shape == (2,)
