from __future__ import annotations

import json
import shutil
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .core.ir import FuncCall, IndexFunction, ProgramIR, TensorRef, Term
from .core.policies import ManifestWeightStore, RuntimePolicies
from .core.program import Program


def _serialize_slice(slice_spec: Optional[Any]) -> Optional[Dict[str, Any]]:
    if slice_spec is None:
        return None
    return {
        "start": slice_spec.start,
        "stop": slice_spec.stop,
        "step": slice_spec.step,
    }


def _serialize_index_spec(spec: Any) -> Dict[str, Any]:
    return {
        "axis": spec.axis,
        "offset": spec.offset,
        "slice": _serialize_slice(spec.slice),
    }


def _serialize_tensor_ref(ref: TensorRef) -> Dict[str, Any]:
    return {
        "name": ref.name,
        "indices": list(ref.indices),
        "dotted_axes": list(ref.dotted_axes),
        "rolling": dict(ref.rolling),
        "index_specs": [_serialize_index_spec(spec) for spec in ref.index_specs],
        "is_paren": ref.is_paren,
    }


def _serialize_expr(expr: Any) -> Any:
    if isinstance(expr, TensorRef):
        return {"type": "tensor_ref", **_serialize_tensor_ref(expr)}
    if isinstance(expr, Term):
        return {
            "type": "term",
            "factors": [_serialize_expr(factor) for factor in expr.factors],
        }
    if isinstance(expr, FuncCall):
        if isinstance(expr.arg, tuple):
            arg = [_serialize_expr(item) for item in expr.arg]
        elif expr.arg is None:
            arg = None
        else:
            arg = _serialize_expr(expr.arg)
        kwargs: Dict[str, Any] = {}
        for key, value in expr.kwargs.items():
            if isinstance(value, (TensorRef, Term, FuncCall, IndexFunction, tuple, list)):
                kwargs[key] = _serialize_expr(value)
            else:
                kwargs[key] = value
        return {
            "type": "func",
            "name": expr.name,
            "arg": arg,
            "kwargs": kwargs,
        }
    if isinstance(expr, IndexFunction):
        return {
            "type": "index_fn",
            "name": expr.name,
            "axis": expr.axis,
        }
    if isinstance(expr, (tuple, list)):
        return [_serialize_expr(item) for item in expr]
    return expr


def serialize_program_ir(ir: ProgramIR) -> Dict[str, Any]:
    equations: List[Dict[str, Any]] = []
    for eq in ir.equations:
        equations.append(
            {
                "lhs": _serialize_tensor_ref(eq.lhs),
                "rhs": _serialize_expr(eq.rhs),
                "projection": eq.projection,
                "src_file": eq.src_file,
                "sink_file": eq.sink_file,
                "is_source": eq.is_source,
                "is_sink": eq.is_sink,
                "export": eq.export,
            }
        )
    return {
        "equations": equations,
        "exports": list(ir.exports),
    }


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _sanitize_logs(logs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return json.loads(json.dumps(list(logs), default=_json_default))


def _extract_kernel_stats(logs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    kernels: List[Dict[str, Any]] = []
    for entry in logs:
        if entry.get("kind") != "equation":
            continue
        data = dict(entry["equation"])
        data["projected"] = list(data.get("projected", []))
        data["contracted"] = list(data.get("contracted") or [])
        data["output_indices"] = list(data.get("output_indices") or [])
        kernels.append(data)
    return kernels


def _index_kernels(
    kernels: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[Any, ...], Mapping[str, Any]]:
    counter: Dict[Tuple[Any, ...], int] = defaultdict(int)
    mapping: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    for kernel in kernels:
        key_base = (
            kernel.get("iteration"),
            kernel.get("name"),
            kernel.get("einsum"),
            tuple(kernel.get("projected", [])),
            kernel.get("status"),
        )
        idx = counter[key_base]
        counter[key_base] = idx + 1
        mapping[key_base + (idx,)] = kernel
    return mapping


def _merge_kernel_runs(
    cold: Sequence[Mapping[str, Any]],
    warm: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    warm_index = _index_kernels(warm)
    counter: Dict[Tuple[Any, ...], int] = defaultdict(int)
    for kernel in cold:
        key_base = (
            kernel.get("iteration"),
            kernel.get("name"),
            kernel.get("einsum"),
            tuple(kernel.get("projected", [])),
            kernel.get("status"),
        )
        idx = counter[key_base]
        counter[key_base] = idx + 1
        warm_kernel = warm_index.get(key_base + (idx,))
        merged.append(
            {
                "name": kernel.get("name"),
                "iteration": kernel.get("iteration"),
                "status": kernel.get("status"),
                "einsum": kernel.get("einsum"),
                "projected": list(kernel.get("projected", [])),
                "contracted": list(kernel.get("contracted", [])),
                "output_indices": list(kernel.get("output_indices", [])),
                "projection": kernel.get("projection"),
                "strategy": kernel.get("strategy"),
                "flops": kernel.get("flops"),
                "bytes_total": kernel.get("bytes_total"),
                "bytes_in": kernel.get("bytes_in"),
                "bytes_out": kernel.get("bytes_out"),
                "cold": {
                    "duration_ms": kernel.get("duration_ms"),
                    "flops": kernel.get("flops"),
                    "bytes_total": kernel.get("bytes_total"),
                },
                "warm": None
                if warm_kernel is None
                else {
                    "duration_ms": warm_kernel.get("duration_ms"),
                    "flops": warm_kernel.get("flops"),
                    "bytes_total": warm_kernel.get("bytes_total"),
                },
            }
        )
    return merged


def _summarize_kernels(kernels: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_flops = float(sum(k.get("flops") or 0.0 for k in kernels))
    total_bytes = int(sum(k.get("bytes_total") or 0 for k in kernels))
    total_duration = float(
        sum((k.get("duration_ms") or 0.0) for k in kernels if k.get("duration_ms") is not None)
    )
    durations_reported = sum(1 for k in kernels if k.get("duration_ms") is not None)
    return {
        "kernel_count": len(kernels),
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "total_duration_ms": total_duration,
        "timed_kernels": durations_reported,
    }


def _format_duration(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _format_engineering(value: Optional[Union[int, float]], unit: str) -> str:
    if value is None:
        return "-"
    magnitude = float(value)
    if magnitude == 0:
        return f"0 {unit}"
    prefixes = ["", "K", "M", "G", "T", "P"]
    idx = 0
    while abs(magnitude) >= 1000.0 and idx < len(prefixes) - 1:
        magnitude /= 1000.0
        idx += 1
    return f"{magnitude:.3g} {prefixes[idx]}{unit}"


def _render_explain_md(
    backend: str,
    device: str,
    digest: str,
    cold_time_ms: float,
    warm_time_ms: Optional[float],
    plan_entries: Sequence[Mapping[str, Any]],
) -> str:
    lines = []
    lines.append("# Fuse Explain")
    lines.append("")
    lines.append(f"- Backend: `{backend}`")
    lines.append(f"- Device: `{device}`")
    lines.append(f"- Program digest: `{digest}`")
    lines.append(f"- Cold run: {cold_time_ms:.3f} ms")
    if warm_time_ms is not None:
        lines.append(f"- Warm run: {warm_time_ms:.3f} ms")
    lines.append("")
    lines.append("## Kernels")
    lines.append("")
    header = "| Name | Iter | Status | Cold (ms) | Warm (ms) | FLOPs | Bytes | Einsum | Projected | Contracted |"
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines.append(header)
    lines.append(divider)
    for entry in plan_entries:
        projected = ",".join(entry.get("projected") or [])
        contracted = ",".join(entry.get("contracted") or [])
        lines.append(
            "| {name} | {iter} | {status} | {cold} | {warm} | {flops} | {bytes} | `{einsum}` | {proj} | {contr} |".format(
                name=entry.get("name"),
                iter=entry.get("iteration"),
                status=entry.get("status"),
                cold=_format_duration(
                    entry["cold"].get("duration_ms") if entry.get("cold") else None
                ),
                warm=_format_duration(
                    entry["warm"].get("duration_ms") if entry.get("warm") else None
                ),
                flops=_format_engineering(entry.get("flops"), "FLOP"),
                bytes=_format_engineering(entry.get("bytes_total"), "B"),
                einsum=entry.get("einsum") or "",
                proj=projected or "-",
                contr=contracted or "-",
            )
        )
    lines.append("")
    lines.append("### Projection Sets")
    lines.append("")
    for entry in plan_entries:
        proj = entry.get("projected") or []
        projection = entry.get("projection", "sum")
        lines.append(
            f"- `{entry.get('name')}` (iter {entry.get('iteration')}, projection={projection}): [{', '.join(proj)}]"
            if proj
            else f"- `{entry.get('name')}` (iter {entry.get('iteration')}, projection={projection}): []"
        )
    lines.append("")
    lines.append("### Normalized Einsums")
    lines.append("")
    seen = set()
    for entry in plan_entries:
        key = (entry.get("name"), entry.get("iteration"), entry.get("einsum"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- `{entry.get('name')}` iter {entry.get('iteration')}: `{entry.get('einsum') or ''}`"
        )
    lines.append("")
    return "\n".join(lines)


def _ensure_runtime_policies(policies: Optional[RuntimePolicies]) -> RuntimePolicies:
    return policies or RuntimePolicies()


def build_package(
    program: Program,
    *,
    package_path: Union[str, Path],
    backend: str = "numpy",
    device: str = "auto",
    inputs: Optional[Mapping[str, Any]] = None,
    config: Optional[Any] = None,
    policies: Optional[RuntimePolicies] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    warm_run: bool = True,
) -> Path:
    package_path = Path(package_path)
    input_map = dict(inputs or {})

    temp_cache_dir: Optional[Path] = None
    if cache_dir is None:
        temp_cache_dir = Path(tempfile.mkdtemp(prefix="fuse-cache-"))
        cache_path = temp_cache_dir
    else:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

    runtime_policies = _ensure_runtime_policies(policies)
    runner = program.compile(
        backend=backend,
        device=device,
        cache_dir=str(cache_path),
        config=config,
        policies=runtime_policies,
    )

    def _run_once() -> Tuple[float, List[Dict[str, Any]]]:
        start = time.perf_counter()
        runner(inputs=input_map)
        duration_ms = (time.perf_counter() - start) * 1000.0
        logs = _sanitize_logs(runner.logs)
        return duration_ms, logs

    cold_time_ms, cold_logs = _run_once()
    warm_time_ms: Optional[float] = None
    warm_logs: List[Dict[str, Any]] = []
    if warm_run:
        warm_time_ms, warm_logs = _run_once()

    cold_kernels = _extract_kernel_stats(cold_logs)
    warm_kernels = _extract_kernel_stats(warm_logs)
    plan_entries = _merge_kernel_runs(cold_kernels, warm_kernels)

    profile = {
        "backend": backend,
        "device": device,
        "digest": program.digest,
        "runs": [
            {
                "kind": "cold",
                "duration_ms": cold_time_ms,
                "summary": _summarize_kernels(cold_kernels),
                "kernels": cold_kernels,
            }
        ],
    }
    if warm_run:
        profile["runs"].append(
            {
                "kind": "warm",
                "duration_ms": warm_time_ms,
                "summary": _summarize_kernels(warm_kernels),
                "kernels": warm_kernels,
            }
        )

    plan = {
        "backend": backend,
        "device": device,
        "digest": program.digest,
        "cold_time_ms": cold_time_ms,
        "warm_time_ms": warm_time_ms,
        "kernels": plan_entries,
    }

    manifest_state: Optional[Dict[str, Any]] = None
    runner_policies = getattr(runner, "policies", None)
    active_store = getattr(runner_policies, "weight_store", None)
    if isinstance(active_store, ManifestWeightStore):
        manifest_state = active_store.export_state()
    elif isinstance(runtime_policies.weight_store, ManifestWeightStore):
        manifest_state = runtime_policies.weight_store.export_state()

    explain_md = _render_explain_md(
        backend,
        device,
        program.digest,
        cold_time_ms,
        warm_time_ms,
        plan_entries,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "source").mkdir(parents=True, exist_ok=True)
        (root / "ir").mkdir(parents=True, exist_ok=True)
        (root / "plans").mkdir(parents=True, exist_ok=True)
        (root / "traces").mkdir(parents=True, exist_ok=True)
        (root / "caches").mkdir(parents=True, exist_ok=True)
        (root / "manifests").mkdir(parents=True, exist_ok=True)

        (root / "source" / "program.fuse").write_text(program.src, encoding="utf-8")

        ir_data = serialize_program_ir(program.ir)
        with (root / "ir" / "program.json").open("w", encoding="utf-8") as fh:
            json.dump(ir_data, fh, indent=2, default=_json_default)

        with (root / "plans" / "plan.json").open("w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=2, default=_json_default)

        with (root / "profile.json").open("w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, default=_json_default)

        (root / "explain.md").write_text(explain_md, encoding="utf-8")

        with (root / "traces" / "run_cold.json").open("w", encoding="utf-8") as fh:
            json.dump(cold_logs, fh, indent=2, default=_json_default)
        if warm_run:
            with (root / "traces" / "run_warm.json").open("w", encoding="utf-8") as fh:
                json.dump(warm_logs, fh, indent=2, default=_json_default)

        if manifest_state is not None:
            with (root / "manifests" / "weights.json").open("w", encoding="utf-8") as fh:
                json.dump(manifest_state, fh, indent=2, default=_json_default)

        cache_target = root / "caches"
        if cache_path.exists():
            for item in cache_path.iterdir():
                dest = cache_target / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in root.rglob("*"):
                zf.write(file_path, file_path.relative_to(root))

    if temp_cache_dir is not None:
        shutil.rmtree(temp_cache_dir, ignore_errors=True)

    return package_path


__all__ = ["build_package", "serialize_program_ir"]
