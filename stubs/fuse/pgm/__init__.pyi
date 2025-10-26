from ..core.program import Program

FACTOR_GRAPH_SOURCES: tuple[str, ...]
HMM_FORWARD_SOURCES: tuple[str, ...]

def factor_graph_triplet() -> Program: ...
def hmm_forward_two_step() -> Program: ...
