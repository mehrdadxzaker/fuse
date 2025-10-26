import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, Any

import numpy as np

from ..core.program import Program
from ..core.policies import InMemoryWeightStore, RuntimePolicies
from ..core.evaluator_numpy import ExecutionConfig


@dataclass(frozen=True)
class Variable:
    name: str
    cardinality: int


@dataclass(frozen=True)
class Factor:
    name: str
    scope: Tuple[str, ...]
    table: np.ndarray

    def normalized(self) -> "Factor":
        table = np.asarray(self.table, dtype=np.float64)
        if table.ndim != len(self.scope):
            raise ValueError(
                f"Factor '{self.name}' table rank {table.ndim} "
                f"does not match scope {self.scope}"
            )
        if np.any(table < 0):
            raise ValueError(f"Factor '{self.name}' contains negative entries")
        return Factor(name=self.name, scope=self.scope, table=table)


@dataclass
class TreeProgram:
    program: Program
    weights: Dict[str, np.ndarray]
    query_vars: Tuple[str, ...]
    variables: Dict[str, Variable]

    def compile(
        self,
        *,
        config: Optional[ExecutionConfig] = None,
    ):
        policies = RuntimePolicies(
            weight_store=InMemoryWeightStore(self.weights.copy())
        )
        return self.program.compile(config=config, policies=policies)

    @property
    def source(self) -> str:
        return self.program.src


class TreeFactorGraph:
    def __init__(
        self,
        variables: Sequence[Variable],
        factors: Sequence[Factor],
    ):
        if not variables:
            raise ValueError("TreeFactorGraph requires at least one variable")
        if not factors:
            raise ValueError("TreeFactorGraph requires at least one factor")
        self.variables: Dict[str, Variable] = {}
        for var in variables:
            if var.name in self.variables:
                raise ValueError(f"Duplicate variable '{var.name}'")
            if var.cardinality <= 0:
                raise ValueError(f"Variable '{var.name}' must have positive cardinality")
            self.variables[var.name] = var

        self.factors: Dict[str, Factor] = {}
        for factor in factors:
            factor = factor.normalized()
            if factor.name in self.factors:
                raise ValueError(f"Duplicate factor '{factor.name}'")
            for axis in factor.scope:
                if axis not in self.variables:
                    raise ValueError(
                        f"Factor '{factor.name}' references unknown variable '{axis}'"
                    )
            expected_shape = tuple(self.variables[axis].cardinality for axis in factor.scope)
            if factor.table.shape != expected_shape:
                raise ValueError(
                    f"Factor '{factor.name}' table shape {factor.table.shape} "
                    f"does not match variable cardinalities {expected_shape}"
                )
            self.factors[factor.name] = factor

        self._check_tree_structure()

    def build_program(
        self,
        *,
        query_vars: Sequence[str],
        evidence: Optional[Dict[str, Any]] = None,
    ) -> TreeProgram:
        if not query_vars:
            raise ValueError("query_vars must be non-empty")
        for name in query_vars:
            if name not in self.variables:
                raise ValueError(f"Unknown query variable '{name}'")
        query_tuple = tuple(query_vars)
        # Pick a root factor that covers all query variables.
        root_factor = None
        query_set = set(query_tuple)
        for factor in self.factors.values():
            if query_set.issubset(factor.scope):
                root_factor = factor
                break
        if root_factor is None:
            raise ValueError(
                "No factor contains all query variables; choose a different query set "
                "or provide a join tree factor covering it."
            )

        weights = self._build_weight_store(evidence or {})
        message_lines: List[str] = []

        adjacency = self._build_adjacency()
        root_node = ("factor", root_factor.name)
        visited: Set[Tuple[str, str]] = set()

        def dfs(node: Tuple[str, str], parent: Optional[Tuple[str, str]]):
            visited.add(node)
            neighbors = sorted(
                adjacency[node],
                key=lambda item: (0 if item[0] == "factor" else 1, item[1]),
            )
            for child in neighbors:
                if parent is not None and child == parent:
                    continue
                if child not in visited:
                    dfs(child, node)
            if parent is not None:
                message_lines.append(self._emit_message(node, parent))

        dfs(root_node, None)

        root_terms = [self._factor_tensor_name(root_factor.name, root_factor.scope)]
        for var_name in root_factor.scope:
            root_terms.append(
                self._msg_var_to_factor_name(var_name, root_factor.name, index=var_name)
            )
        prog_qe_lhs = self._format_indices("ProgQE", query_tuple)
        prog_qe_rhs = " ".join(root_terms)
        final_lines: List[str] = []
        final_lines.append(f"{prog_qe_lhs} = {prog_qe_rhs}")
        prog_e_lhs = self._format_indices("ProgE", ())
        prog_e_terms = list(root_terms)
        for var_name in query_tuple:
            prog_e_terms.append(
                self._evidence_tensor_name(var_name, index=var_name)
            )
        final_lines.append(f"{prog_e_lhs} = {' '.join(prog_e_terms)}")
        final_lines.append("export ProgQE")
        final_lines.append("export ProgE")

        message_lines = list(reversed(message_lines))
        program_src = "\n".join(final_lines + message_lines)
        program = Program(program_src)
        return TreeProgram(
            program=program,
            weights=weights,
            query_vars=query_tuple,
            variables=self.variables,
        )

    def _check_tree_structure(self):
        adjacency = self._build_adjacency()
        nodes = list(adjacency.keys())
        edges = sum(len(neigh) for neigh in adjacency.values()) // 2
        if edges != len(nodes) - 1:
            raise ValueError(
                "Factor graph must be a tree (edges != nodes - 1)"
            )
        # Ensure connectivity
        start = nodes[0]
        stack = [start]
        seen: Set[Tuple[str, str]] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    stack.append(neighbor)
        if len(seen) != len(nodes):
            raise ValueError("Factor graph must be connected")

    def _build_adjacency(self) -> Dict[Tuple[str, str], Set[Tuple[str, str]]]:
        adjacency: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        for factor in self.factors.values():
            factor_node = ("factor", factor.name)
            adjacency.setdefault(factor_node, set())
            for var_name in factor.scope:
                var_node = ("var", var_name)
                adjacency.setdefault(var_node, set())
                adjacency[factor_node].add(var_node)
                adjacency[var_node].add(factor_node)
        return adjacency

    def _emit_message(
        self,
        node: Tuple[str, str],
        parent: Tuple[str, str],
    ) -> str:
        if node[0] == "factor" and parent[0] == "var":
            return self._emit_factor_to_var(node[1], parent[1])
        if node[0] == "var" and parent[0] == "factor":
            return self._emit_var_to_factor(node[1], parent[1])
        raise ValueError(f"Unexpected edge types {node} -> {parent}")

    def _emit_factor_to_var(self, factor_name: str, var_name: str) -> str:
        factor = self.factors[factor_name]
        neighbors = [
            axis for axis in factor.scope if axis != var_name
        ]
        terms = [self._factor_tensor_name(factor.name, factor.scope)]
        for child_var in neighbors:
            terms.append(
                self._msg_var_to_factor_name(child_var, factor.name, index=child_var)
            )
        lhs = self._msg_factor_to_var_name(factor.name, var_name, index=var_name)
        return f"{lhs} = {' '.join(terms)}"

    def _emit_var_to_factor(self, var_name: str, factor_name: str) -> str:
        terms = [self._evidence_tensor_name(var_name, index=var_name)]
        for neighbor in self._variable_neighbors(var_name):
            if neighbor == factor_name:
                continue
            terms.append(
                self._msg_factor_to_var_name(neighbor, var_name, index=var_name)
            )
        lhs = self._msg_var_to_factor_name(var_name, factor_name, index=var_name)
        return f"{lhs} = {' '.join(terms)}"

    def _variable_neighbors(self, var_name: str) -> Iterable[str]:
        factor_names = []
        for factor in self.factors.values():
            if var_name in factor.scope:
                factor_names.append(factor.name)
        return factor_names

    def _build_weight_store(self, evidence: Dict[str, Any]) -> Dict[str, np.ndarray]:
        weights: Dict[str, np.ndarray] = {}
        for factor in self.factors.values():
            weights[self._factor_tensor_identifier(factor.name)] = np.asarray(
                factor.table, dtype=np.float64
            )
        for var_name, var in self.variables.items():
            evid_value = evidence.get(var_name)
            if evid_value is None:
                arr = np.ones(var.cardinality, dtype=np.float64)
            else:
                arr = self._normalize_evidence(var, evid_value)
            weights[self._evidence_tensor_identifier(var_name)] = arr
        return weights

    def _normalize_evidence(self, var: Variable, value: Any) -> np.ndarray:
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != (var.cardinality,):
                raise ValueError(
                    f"Evidence for '{var.name}' must have shape {(var.cardinality,)}"
                )
            return arr
        index = int(value)
        if index < 0 or index >= var.cardinality:
            raise ValueError(
                f"Evidence value {index} out of range for variable '{var.name}'"
            )
        arr = np.zeros(var.cardinality, dtype=np.float64)
        arr[index] = 1.0
        return arr

    @staticmethod
    def _format_indices(name: str, indices: Sequence[str]) -> str:
        if not indices:
            return name
        joined = ", ".join(indices)
        return f"{name}[{joined}]"

    @staticmethod
    def _factor_tensor_identifier(factor_name: str) -> str:
        return f"Factor_{factor_name}"

    @classmethod
    def _factor_tensor_name(cls, factor_name: str, scope: Sequence[str]) -> str:
        base = cls._factor_tensor_identifier(factor_name)
        return cls._format_indices(base, scope)

    @staticmethod
    def _evidence_tensor_identifier(var_name: str) -> str:
        return f"Evidence_{var_name}"

    @classmethod
    def _evidence_tensor_name(cls, var_name: str, *, index: str) -> str:
        return cls._format_indices(cls._evidence_tensor_identifier(var_name), (index,))

    @staticmethod
    def _msg_factor_to_var_identifier(factor_name: str, var_name: str) -> str:
        return f"MsgF_{factor_name}_{var_name}"

    @classmethod
    def _msg_factor_to_var_name(cls, factor_name: str, var_name: str, *, index: str) -> str:
        return cls._format_indices(
            cls._msg_factor_to_var_identifier(factor_name, var_name),
            (index,),
        )

    @staticmethod
    def _msg_var_to_factor_identifier(var_name: str, factor_name: str) -> str:
        return f"MsgV_{var_name}_{factor_name}"

    @classmethod
    def _msg_var_to_factor_name(cls, var_name: str, factor_name: str, *, index: str) -> str:
        return cls._format_indices(
            cls._msg_var_to_factor_identifier(var_name, factor_name),
            (index,),
        )


def conditional_probability(
    outputs: Dict[str, Any],
    query_vars: Sequence[str],
    assignment: Dict[str, int],
) -> float:
    joint = np.asarray(outputs["ProgQE"], dtype=np.float64)
    evidence_total = float(np.asarray(outputs["ProgE"], dtype=np.float64))
    if evidence_total == 0.0:
        raise ValueError("Evidence has zero probability; cannot normalize.")
    indices = tuple(int(assignment[var]) for var in query_vars)
    value = float(joint[indices])
    return value / evidence_total


def brute_force_joint(
    factors: Sequence[Factor],
    variables: Sequence[Variable],
    evidence: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    evidence = evidence or {}
    var_order = [var.name for var in variables]
    cardinalities = [var.cardinality for var in variables]
    total_states = int(np.prod(cardinalities)) if cardinalities else 1
    joint = np.zeros(total_states, dtype=np.float64)
    var_to_index = {name: idx for idx, name in enumerate(var_order)}

    for idx, assignment in enumerate(itertools.product(*[range(c) for c in cardinalities])):
        weight = 1.0
        valid = True
        for var_name, value in evidence.items():
            if assignment[var_to_index[var_name]] != value:
                valid = False
                break
        if not valid:
            continue
        for factor in factors:
            coords = tuple(assignment[var_to_index[name]] for name in factor.scope)
            weight *= factor.table[coords]
        joint[idx] = weight
    return joint.reshape(tuple(cardinalities))
