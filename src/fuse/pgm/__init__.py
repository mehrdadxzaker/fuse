"""
Probabilistic graphical model program templates.

The helpers here provide junction-tree friendly equation sets for factor
graphs and small hidden Markov models that can be compiled directly with the
Fuse runtime.
"""

from __future__ import annotations

from typing import Tuple

from ..core.program import Program

__all__ = [
    "factor_graph_triplet",
    "hmm_forward_two_step",
    "FACTOR_GRAPH_SOURCES",
    "HMM_FORWARD_SOURCES",
]

FACTOR_GRAPH_SOURCES: Tuple[str, ...] = (
    "UnaryXPotential[x]",
    "UnaryYPotential[y]",
    "UnaryZPotential[z]",
    "PairXYPotential[x,y]",
    "PairYZPotential[y,z]",
)

_FACTOR_GRAPH_EQUATIONS = """
# Pairwise factor graph with three variables and junction tree style marginals
UnaryX[x] = UnaryXPotential[x]
UnaryY[y] = UnaryYPotential[y]
UnaryZ[z] = UnaryZPotential[z]
FactorXY[x,y] = PairXYPotential[x,y]
FactorYZ[y,z] = PairYZPotential[y,z]

Joint[x,y,z] = UnaryX[x] UnaryY[y] UnaryZ[z] FactorXY[x,y] FactorYZ[y,z]

MarginalX[x] = Joint[x,y,z] UnaryY[y] UnaryZ[z]
MarginalY[y] = Joint[x,y,z] UnaryX[x] UnaryZ[z]
MarginalZ[z] = Joint[x,y,z] UnaryX[x] UnaryY[y]

ProbX[x.] = softmax(MarginalX[x])
ProbY[y.] = softmax(MarginalY[y])
ProbZ[z.] = softmax(MarginalZ[z])

export Joint
export ProbX
export ProbY
export ProbZ
"""

HMM_FORWARD_SOURCES: Tuple[str, ...] = (
    "PriorProb[s]",
    "TransitionMatrix[s,s']",
    "EmissionMatrix[s,o]",
    "ObservationOneHot[t,o]",
)

_HMM_FORWARD_EQUATIONS = """
# Two-step Hidden Markov Model forward pass
Prior[s] = PriorProb[s]
Trans[s,s'] = TransitionMatrix[s,s']
Emit[s,o] = EmissionMatrix[s,o]
Obs[t,o] = ObservationOneHot[t,o]

Emit0[s] = Emit[s,o] Obs[0,o]
Alpha0[s] = Prior[s] Emit0[s]

Emit1[s] = Emit[s,o] Obs[1,o]
Message1[s] = Alpha0[s'] Trans[s',s]
Alpha1[s] = Message1[s] Emit1[s]

Select0[t] = const([1,0])
Select1[t] = const([0,1])

Alpha[t,s] = Select0[t] Alpha0[s] + Select1[t] Alpha1[s]
Normalized[t,s.] = softmax(Alpha[t,s])

export Alpha
export Normalized
"""


def factor_graph_triplet() -> Program:
    """
    Build a simple pairwise factor graph over three variables.

    Provide unary and pairwise potentials matching
    :data:`FACTOR_GRAPH_SOURCES`. The program exports the joint tensor and the
    normalized unary marginals.
    """

    return Program(_FACTOR_GRAPH_EQUATIONS)


def hmm_forward_two_step() -> Program:
    """
    Build a two-observation Hidden Markov Model forward recursion.

    The observation selector constants can be replaced or extended downstream
    by modifying the returned program source if more timesteps are required.
    """

    return Program(_HMM_FORWARD_EQUATIONS)
