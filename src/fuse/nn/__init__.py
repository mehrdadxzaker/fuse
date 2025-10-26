"""
Neural network oriented program templates.

This module provides ready-to-use :class:`~fuse.core.program.Program` factories
implementing common building blocks such as attention layers, transformer
encoders, and graph neural network message passing. Each factory returns a
parsed ``Program`` whose sources are expected to be materialised via the
runtime weight store (for example using :class:`~fuse.core.policies.InMemoryWeightStore`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.program import Program

__all__ = [
    "attention_block",
    "transformer_encoder",
    "graph_message_passing",
    "ATTENTION_SOURCES",
    "TRANSFORMER_SOURCES",
    "graph_message_passing_sources",
]

# ---------------------------------------------------------------------------
# Attention blocks

ATTENTION_SOURCES: Tuple[str, ...] = (
    "InputStream[b,p,d]",
    "PositionalEncoding[b,p,d]",
    "QueryWeight[h,dk,d]",
    "KeyWeight[h,dk,d]",
    "ValueWeight[h,dv,d]",
    "AttentionScale",
    "AttentionMask[b,h,p,p']",
    "OutputProj[d,h,dv]",
)

_ATTENTION_EQUATIONS = """
# Multi-head attention context block with optional mask
Stream[b,p,d] = InputStream[b,p,d]
PosEnc[b,p,d] = PositionalEncoding[b,p,d]
StreamPos[b,p,d] = Stream[b,p,d]
StreamPos[b,p,d] += PosEnc[b,p,d]

Query[b,h,p,dk] = QueryWeight[h,dk,d] StreamPos[b,p,d]
Key[b,h,p,dk]   = KeyWeight[h,dk,d] StreamPos[b,p,d]
Value[b,h,p,dv] = ValueWeight[h,dv,d] StreamPos[b,p,d]

Score[b,h,p,p']  = Query[b,h,p,dk] Key[b,h,p',dk]
Scaled[b,h,p,p'] = Score[b,h,p,p'] AttentionScale
Masked[b,h,p,p'] = Scaled[b,h,p,p']
Masked[b,h,p,p'] += AttentionMask[b,h,p,p']
AttnWeights[b,h,p,p'.] = softmax(Masked[b,h,p,p'])
Attn[b,h,p,dv] = AttnWeights[b,h,p,p'] Value[b,h,p',dv]

Context[b,p,d] = OutputProj[d,h,dv] Attn[b,h,p,dv]

export Context
"""


def attention_block() -> Program:
    """
    Build a multi-head attention program.

    The program expects the following tensors to be resolvable via the weight
    store (names match :data:`ATTENTION_SOURCES`):

    ``InputStream[b,p,d]`` – source activations
    ``PositionalEncoding[b,p,d]`` – additive positional signal (set to zeros when unused)
    ``QueryWeight[h,dk,d]``, ``KeyWeight[h,dk,d]``, ``ValueWeight[h,dv,d]`` – head projections
    ``AttentionScale`` – scalar (typically ``1/sqrt(dk)``)
    ``AttentionMask[b,h,p,p']`` – additive mask per head; supply zeros to disable masking
    ``OutputProj[d,h,dv]`` – projection for concatenated head outputs

    Returns
    -------
    Program
        Ready-to-run attention block exporting ``Context``.
    """

    return Program(_ATTENTION_EQUATIONS)


# ---------------------------------------------------------------------------
# Transformer encoder

TRANSFORMER_SOURCES: Tuple[str, ...] = (
    *ATTENTION_SOURCES,
    "MlpWeight1[ff,d]",
    "MlpWeight2[d,ff]",
    "ReadoutWeight[t,d]",
)

_TRANSFORMER_EQUATIONS = """
# Transformer encoder block with residual connections and feed-forward MLP
Stream[b,p,d] = InputStream[b,p,d]
PosEnc[b,p,d] = PositionalEncoding[b,p,d]
StreamPos[b,p,d] = Stream[b,p,d]
StreamPos[b,p,d] += PosEnc[b,p,d]

Query[b,h,p,dk] = QueryWeight[h,dk,d] StreamPos[b,p,d]
Key[b,h,p,dk]   = KeyWeight[h,dk,d] StreamPos[b,p,d]
Value[b,h,p,dv] = ValueWeight[h,dv,d] StreamPos[b,p,d]

Score[b,h,p,p']  = Query[b,h,p,dk] Key[b,h,p',dk]
Scaled[b,h,p,p'] = Score[b,h,p,p'] AttentionScale
Masked[b,h,p,p'] = Scaled[b,h,p,p']
Masked[b,h,p,p'] += AttentionMask[b,h,p,p']
AttnWeights[b,h,p,p'.] = softmax(Masked[b,h,p,p'])
Attn[b,h,p,dv] = AttnWeights[b,h,p,p'] Value[b,h,p',dv]

Attended[b,p,d] = OutputProj[d,h,dv] Attn[b,h,p,dv]
Resid1[b,p,d] = Attended[b,p,d]
Resid1[b,p,d] += Stream[b,p,d]
Norm1[b,p,d.] = lnorm(Resid1[b,p,d])

MlpHidden[b,p,ff] = MlpWeight1[ff,d] Norm1[b,p,d]
MlpAct[b,p,ff] = relu(MlpHidden[b,p,ff])
MlpOut[b,p,d] = MlpWeight2[d,ff] MlpAct[b,p,ff]

Resid2[b,p,d] = Norm1[b,p,d]
Resid2[b,p,d] += MlpOut[b,p,d]
StreamOut[b,p,d.] = lnorm(Resid2[b,p,d])

Y[b,p,t.] = softmax(ReadoutWeight[t,d] StreamOut[b,p,d])

export StreamOut
export Y
"""


def transformer_encoder() -> Program:
    """
    Build a single transformer encoder layer (self-attention + MLP).

    Required tensor names are listed in :data:`TRANSFORMER_SOURCES`. Provide an
    attention scale scalar and a per-head additive mask compatible with the
    batch/sequence layout. The program exports both the post-MLP stream
    (``StreamOut``) and the token logits (``Y``).
    """

    return Program(_TRANSFORMER_EQUATIONS)


# ---------------------------------------------------------------------------
# Graph neural network message passing


@dataclass(frozen=True)
class _GNNSpec:
    num_layers: int
    feature_name: str = "NodeFeatures"
    adjacency_name: str = "Adjacency"
    message_prefix: str = "MessageWeight"
    agg_prefix: str = "AggWeight"
    self_prefix: str = "SelfWeight"

    def required_sources(self) -> Tuple[str, ...]:
        base: Tuple[str, ...] = (
            f"{self.adjacency_name}[n,n']",
            f"{self.feature_name}[n,d]",
            "NodeClassifier[c,d]",
            "GraphClassifier[c,d]",
            "GraphPooling[n]",
        )
        per_layer: Tuple[str, ...] = tuple(
            f"{prefix}{layer}[d,d]"
            for layer in range(self.num_layers)
            for prefix in (self.message_prefix, self.agg_prefix, self.self_prefix)
        )
        return base + per_layer

    def build_equations(self) -> str:
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        lines = [
            "# Graph neural network message passing with shared hidden size",
            f"Adj[n,n'] = {self.adjacency_name}[n,n']",
            f"EmbLayer0[n,d] = {self.feature_name}[n,d]",
        ]
        for layer in range(self.num_layers):
            next_layer = layer + 1
            lines.extend(
                [
                    f"Message{layer}[n,d] = {self.message_prefix}{layer}[d,d] EmbLayer{layer}[n,d]",
                    f"MessageRelu{layer}[n,d] = relu(Message{layer}[n,d])",
                    f"Agg{layer}[n,d] = Adj[n,n'] MessageRelu{layer}[n',d]",
                    f"AggMix{layer}[n,d] = {self.agg_prefix}{layer}[d,d] Agg{layer}[n,d]",
                    f"SelfMix{layer}[n,d] = {self.self_prefix}{layer}[d,d] EmbLayer{layer}[n,d]",
                    f"Update{layer}[n,d] = AggMix{layer}[n,d]",
                    f"Update{layer}[n,d] += SelfMix{layer}[n,d]",
                    f"EmbLayer{next_layer}[n,d] = relu(Update{layer}[n,d])",
                ]
            )
        final_layer = self.num_layers
        lines.extend(
            [
                f"NodeLogits[n,c] = NodeClassifier[c,d] EmbLayer{final_layer}[n,d]",
                "NodeProbs[n,c.] = softmax(NodeLogits[n,c])",
                f"NeighborEmb{final_layer}[n',d] = EmbLayer{final_layer}[n',d]",
                f"EdgeScore[n,n'] = EmbLayer{final_layer}[n,d] NeighborEmb{final_layer}[n',d]",
                "EdgeProb[n,n'.] = sig(EdgeScore[n,n'])",
                f"GraphReadout[d] = GraphPooling[n] EmbLayer{final_layer}[n,d]",
                "GraphLogits[c] = GraphClassifier[c,d] GraphReadout[d]",
                "GraphProb[c.] = softmax(GraphLogits[c])",
                f"Embeddings[n,d] = EmbLayer{final_layer}[n,d]",
                "export Embeddings",
                "export NodeProbs",
                "export EdgeProb",
                "export GraphProb",
            ]
        )
        return "\n".join(lines)


def graph_message_passing(num_layers: int = 2) -> Program:
    """
    Build a stacked message passing GNN with linear self/neighbor mixing.

    Parameters
    ----------
    num_layers:
        How many message passing iterations to include. Each layer expects
        per-layer weights ``MessageWeight{l}``, ``AggWeight{l}``, and
        ``SelfWeight{l}`` with square ``[d,d]`` layout, plus a pooling vector
        ``GraphPooling[n]`` to aggregate node embeddings.

    Returns
    -------
    Program
        Program exporting ``Embeddings``, node probabilities, edge logits, and
        graph-level probabilities.
    """

    spec = _GNNSpec(num_layers=num_layers)
    return Program(spec.build_equations())


def graph_message_passing_sources(num_layers: int = 2) -> Tuple[str, ...]:
    """
    List the tensor names the :func:`graph_message_passing` program expects.
    """

    spec = _GNNSpec(num_layers=num_layers)
    return spec.required_sources()
