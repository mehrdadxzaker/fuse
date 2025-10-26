from dataclasses import replace
from typing import Any, Dict, List, Optional

from .cache import CacheManager, build_cache_key, compute_program_hash
from .evaluator_numpy import DemandNumpyRunner, ExecutionConfig, NumpyRunner
from .exceptions import BackendError
from .ir import (
    ProgramIR,
    TensorRef,
    equation_index_summary,
    format_index_summary,
    json_ready,
    lhs_indices,
    rhs_indices,
)
from .parser import parse
from .policies import RuntimePolicies
from .shape_checker import validate_program_shapes


class Program:
    def __init__(self, eqs: str):
        self.src = eqs
        self.ir: ProgramIR = parse(eqs)
        validate_program_shapes(self.ir)
        self.digest = compute_program_hash(self.src)

    def compile(
        self,
        backend: str = "numpy",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        config: Optional[ExecutionConfig] = None,
        execution: Optional[str] = None,
        policies: Optional[RuntimePolicies] = None,
        **backend_kwargs,
    ):
        cfg = config or ExecutionConfig()
        if execution is not None:
            cfg = replace(cfg, mode=execution)
        if device != "auto":
            cfg = replace(cfg, device=device)
        cfg = cfg.normalized()
        target_device = cfg.device

        policy_obj = policies or RuntimePolicies()
        cache_manager = CacheManager(cache_dir) if cache_dir else None

        if backend == "numpy":
            if cfg.device not in {"auto", "cpu"}:
                raise ValueError(
                    f"NumPy backend only supports CPU execution; received device='{cfg.device}'"
                )
            if cfg.precision == "auto":
                cfg = replace(cfg, precision="fp32")
            elif cfg.precision != "fp32":
                raise ValueError(
                    f"NumPy backend only supports fp32 precision; received precision='{cfg.precision}'"
                )
            if cfg.mode == "demand":
                runner = DemandNumpyRunner(self.ir, config=cfg, policies=policy_obj)
            else:
                runner = NumpyRunner(self.ir, config=cfg, policies=policy_obj)
            if cache_manager is not None:
                cache_key = build_cache_key(
                    program_src=self.src,
                    backend="numpy",
                    artifact="metadata",
                    device=target_device,
                    execution_config=cfg,
                    policies=policy_obj,
                )
                cache_manager.write_metadata(
                    "numpy",
                    cache_key,
                    {
                        "device": target_device,
                        "execution": cfg.mode,
                        "digest": self.digest,
                    },
                )
            return runner

        compiler = self._resolve_backend(backend)
        return compiler(
            self,
            device=target_device,
            cache_manager=cache_manager,
            execution_config=cfg,
            policies=policy_obj,
            **backend_kwargs,
        )

    def explain(self, *, json: bool = False) -> Any:
        sources: List[Dict[str, Any]] = []
        sinks: List[Dict[str, Any]] = []
        equations: List[Dict[str, Any]] = []

        for idx, eq in enumerate(self.ir.equations):
            if eq.is_source:
                sources.append(
                    {
                        "name": eq.lhs.name,
                        "path": eq.src_file,
                        "indices": lhs_indices(eq),
                        "line": eq.line,
                        "column": eq.column,
                    }
                )
                continue
            if eq.is_sink:
                sinks.append(
                    {
                        "name": eq.rhs.name if isinstance(eq.rhs, TensorRef) else eq.lhs.name,
                        "path": eq.sink_file,
                        "indices": lhs_indices(eq),
                        "line": eq.line,
                        "column": eq.column,
                    }
                )
                continue

            lhs = lhs_indices(eq)
            rhs = rhs_indices(eq)
            projected = [axis for axis in rhs if axis not in lhs]
            summary = equation_index_summary(eq, projected)
            equations.append(
                {
                    "id": idx,
                    "name": eq.lhs.name,
                    "projection": eq.projection,
                    "equation": eq.source,
                    "index_summary": summary,
                    "index_table": format_index_summary(summary),
                    "line": eq.line,
                    "column": eq.column,
                }
            )

        payload = {
            "digest": self.digest,
            "sources": sources,
            "equations": equations,
            "sinks": sinks,
        }

        if json:
            return json_ready(payload)

        lines: List[str] = []
        for src in sources:
            indices = ",".join(src["indices"]) if src["indices"] else "-"
            lines.append(f"[src] {src['name']} <- {src.get('path') or '<memory>'} idx[{indices}]")
        for entry in equations:
            table = entry["index_table"]
            proj = entry["projection"]
            lines.append(f"[eq] {entry['name']} {table} proj={proj}")
        for sink in sinks:
            indices = ",".join(sink["indices"]) if sink["indices"] else "-"
            lines.append(
                f"[sink] {sink.get('path') or '<memory>'} <- {sink['name']} idx[{indices}]"
            )
        return "\n".join(lines)

    def _resolve_backend(self, backend: str):
        if backend == "torch":
            from ..torch_backend.compile import compile as torch_compile

            return torch_compile
        if backend == "jax":
            from ..jax_backend.compile import compile as jax_compile

            return jax_compile
        raise BackendError(f"Unknown backend '{backend}'")
