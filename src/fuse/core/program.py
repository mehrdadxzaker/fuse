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
        backend: str = "auto",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        config: Optional[ExecutionConfig] = None,
        execution: Optional[str] = None,
        policies: Optional[RuntimePolicies] = None,
        **backend_kwargs,
    ):
        """Compile the program to a runnable for the selected backend.

        When ``backend`` is ``"auto"`` (default), choose a backend based on:
        - Execution mode and projection strategy (NumPy for demand/Monte Carlo).
        - Streaming usage (NumPy when streaming is present).
        - Hardware availability and requested device.
        - A light heuristic over the IR to detect DL-like workloads (einsum-heavy,
          attention/MLP ops) where Torch/JAX are preferable.

        The selection prefers Torch on CUDA/MPS for attention/MLP-style programs,
        otherwise tries JAX when JIT cost can amortize. Small/streaming workloads
        use NumPy to avoid dispatch/JIT overheads.
        """
        cfg = config or ExecutionConfig()
        if execution is not None:
            cfg = replace(cfg, mode=execution)
        if device != "auto":
            cfg = replace(cfg, device=device)
        cfg = cfg.normalized()
        target_device = cfg.device

        policy_obj = policies or RuntimePolicies()
        cache_manager = CacheManager(cache_dir) if cache_dir else None

        # Auto backend selection -------------------------------------------------
        if backend == "auto":
            backend = self._choose_backend_auto(cfg)

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

    # Heuristics ---------------------------------------------------------------
    def _choose_backend_auto(self, cfg: ExecutionConfig) -> str:
        """Pick a backend based on IR, execution config, and hardware.

        Rules of thumb:
        - Demand mode and Monte Carlo projections use NumPy.
        - Streaming programs use NumPy.
        - Prefer Torch on CUDA/MPS for attention/MLP-like workloads.
        - Otherwise try JAX if available for heavier, batched workloads.
        - Fall back to NumPy for small programs and when no accel is present.
        """
        # Fast exits where Torch/JAX intentionally fall back to NumPy
        if cfg.mode == "demand" or cfg.projection_strategy == "monte_carlo":
            return "numpy"
        if self.ir.has_streaming():
            return "numpy"

        # Inspect IR for workload hints
        features = self._program_features()
        equations = features.get("equations", 0)
        einsum_terms = features.get("einsum_terms", 0)
        has_attention = bool(features.get("has_attention", False))
        op_count = features.get("op_count", 0)

        # A simple workload score: einsums and attention weigh more
        score = einsum_terms + op_count + (3 if has_attention else 0)
        small_workload = equations <= 3 and score <= 2

        target = (cfg.device or "auto").lower()

        # Probe availability lazily
        torch_available = False
        torch_gpu_or_mps = False
        try:  # pragma: no cover - import/availability depends on environment
            import torch as _torch  # type: ignore

            torch_available = _torch is not None
            torch_gpu_or_mps = (
                _torch.cuda.is_available() if hasattr(_torch, "cuda") else False
            ) or (
                hasattr(_torch, "backends")
                and hasattr(_torch.backends, "mps")
                and bool(_torch.backends.mps.is_available())
            )
        except Exception:  # pragma: no cover - torch optional
            torch_available = False
            torch_gpu_or_mps = False

        jax_available = False
        jax_has_gpu = False
        try:  # pragma: no cover - import/availability depends on environment
            import jax as _jax  # type: ignore

            jax_available = _jax is not None
            # If jax is present, ask for gpu devices
            try:
                jax_has_gpu = bool(_jax.devices("gpu"))
            except Exception:
                jax_has_gpu = False
        except Exception:  # pragma: no cover - jax optional
            jax_available = False
            jax_has_gpu = False

        # GPU/MPS targets ------------------------------------------------------
        wants_gpu_like = (
            target.startswith("cuda")
            or target == "mps"
            or (target == "auto" and (torch_gpu_or_mps or jax_has_gpu))
        )
        if not small_workload and wants_gpu_like:
            # Prefer Torch for attention/DL-like programs when available
            if torch_available and (
                torch_gpu_or_mps or target.startswith("cuda") or target == "mps"
            ):
                return "torch"
            if jax_available and (jax_has_gpu or target == "auto"):
                return "jax"
            # No accel but heavy: fall through to CPU choices below

        # CPU targets or small workloads --------------------------------------
        if small_workload:
            return "numpy"

        # Heavier CPU workloads: prefer Torch if available, else JAX.
        if torch_available:
            return "torch"
        if jax_available:
            return "jax"
        return "numpy"

    def _program_features(self) -> Dict[str, int]:
        """Collect lightweight structural features from the IR for heuristics."""
        equations = 0
        einsum_terms = 0
        op_count = 0
        has_attention = 0

        from .ir import FuncCall, Term  # local import to avoid cycles

        for eq in self.ir.equations:
            if eq.is_source or eq.is_sink:
                continue
            equations += 1
            rhs = eq.rhs
            if isinstance(rhs, Term):
                if len(getattr(rhs, "factors", []) or []) >= 2:
                    einsum_terms += 1
            if isinstance(rhs, FuncCall):
                op_count += 1
                name = (rhs.name or "").lower()
                if name in {"attention"}:
                    has_attention = 1
        return {
            "equations": equations,
            "einsum_terms": einsum_terms,
            "op_count": op_count,
            "has_attention": has_attention,
        }
