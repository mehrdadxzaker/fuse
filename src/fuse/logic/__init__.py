"""
Logic program templates for symbolic and neuro-symbolic workloads.

These helpers expose ready-to-parse :class:`~fuse.core.program.Program`
instances so downstream users can assemble reasoning pipelines without
rewriting recurring equation sets.
"""

from __future__ import annotations

from typing import Tuple

from ..core.program import Program

__all__ = [
    "aunt_analogy",
    "AUNT_ANALOGY_SOURCES",
]

AUNT_ANALOGY_SOURCES: Tuple[str, ...] = (
    "SisterFacts[x,y]",
    "ParentFacts[y,z]",
    "ObjectEmbeddings[obj,d]",
)

_AUNT_ANALOGY_EQUATIONS = """
# Deductive aunt rule plus embedding-based analogy scoring
Sister(x, y) = SisterFacts[x, y]
Parent(y, z) = ParentFacts[y, z]

Aunt[x, z] = step(Sister[x, y] Parent[y, z])

Emb[obj,d] = ObjectEmbeddings[obj,d]
EmbAunt[i,j] = Aunt[x, y] Emb[x,i] Emb[y,j]
Analogical[a,b] = sig(EmbAunt[i,j] Emb[a,i] Emb[b,j])

export Aunt
export Analogical
"""


def aunt_analogy() -> Program:
    """
    Build a neuro-symbolic aunt inference template.

    Supply boolean relation tensors for ``SisterFacts`` and ``ParentFacts``,
    along with dense ``ObjectEmbeddings`` aligned on the same entity axis.
    The program exports both the inferred ``Aunt`` relation and an analogical
    score tensor ``Analogical[a,b]`` suitable for ranking candidate pairs.
    """

    return Program(_AUNT_ANALOGY_EQUATIONS)
