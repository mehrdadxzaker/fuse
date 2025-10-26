#!/usr/bin/env python3
"""
Attention fast-path benchmark for Torch/JAX backends.

This script exercises the masked softmax, layernorm, and attention kernels
added to the Fuse Torch and JAX runners. It is intended for GPU machines with
PyTorch + Inductor/Triton and/or JAX + XLA builds installed.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from fuse import Program
from fuse import jax as fuse_jax
from fuse import torch as fuse_torch

EQS = """
export Prob
export Norm
export Attn
Scale = const(0.5)
Prob[p,q] = masked_softmax(Logits[p,q], mask=Mask[p,q])
Norm[p,d] = layernorm(X[p,d])
Attn[p,v] = attention(Q[p,d], K[m,d], V[m,v], mask=Mask[p,m], scale=Scale)
"""


@dataclass
class BenchmarkResult:
    backend: str
    device: str
    min_s: float
    mean_s: float
    iterations: int
    tokens_per_s: Optional[float]


def build_program() -> Program:
    return Program(EQS)


def build_inputs(
    *,
    seq: int,
    mem: int,
    d_model: int,
    value_dim: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(seq, mem)).astype(np.float32)
    mask = rng.random(size=(seq, mem)) > 0.1
    np.fill_diagonal(mask[: min(seq, mem), : min(seq, mem)], True)
    x = rng.normal(size=(seq, d_model)).astype(np.float32)
    q = rng.normal(size=(seq, d_model)).astype(np.float32)
    k = rng.normal(size=(mem, d_model)).astype(np.float32)
    v = rng.normal(size=(mem, value_dim)).astype(np.float32)
    return {
        "Logits": logits,
        "Mask": mask,
        "X": x,
        "Q": q,
        "K": k,
        "V": v,
    }


def bench(
    fn: Callable[[], Any],
    sync: Callable[[Any], None],
    *,
    iterations: int,
    warmup: int,
) -> Iterable[float]:
    timings = []
    for step in range(iterations + warmup):
        start = time.perf_counter()
        result = fn()
        sync(result)
        elapsed = time.perf_counter() - start
        if step >= warmup:
            timings.append(elapsed)
    return timings


def _torch_device_name(device) -> str:
    if device.index is not None:
        return f"{device.type}:{device.index}"
    return device.type


def run_torch(
    program: Program,
    base_inputs: Dict[str, Any],
    *,
    device_spec: str,
    iterations: int,
    warmup: int,
    seq: int,
) -> BenchmarkResult:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"torch unavailable: {exc}") from exc

    if device_spec == "auto":
        if torch.cuda.is_available():
            chosen = torch.device("cuda")
        elif torch.backends.mps.is_available():
            chosen = torch.device("mps")
        else:
            chosen = torch.device("cpu")
    else:
        chosen = torch.device(device_spec)

    if chosen.type == "cpu":
        raise RuntimeError("Torch benchmark requires a CUDA or MPS device for meaningful results")

    runner = fuse_torch.compile(program, device=_torch_device_name(chosen))

    def to_torch(value: Any):
        if isinstance(value, np.ndarray):
            return torch.tensor(value, device=chosen)
        return value

    inputs = {name: to_torch(value) for name, value in base_inputs.items()}

    def invoke():
        return runner(inputs=inputs)

    def sync(result: Dict[str, Any]):
        if chosen.type == "cuda":
            torch.cuda.synchronize(chosen)
        elif chosen.type == "mps":
            try:
                torch.mps.synchronize()
            except AttributeError:
                pass
        # Touch one tensor to ensure materialization in eager execution.
        _ = result["Attn"]

    timings = list(bench(invoke, sync, iterations=iterations, warmup=warmup))
    min_s = min(timings)
    mean_s = sum(timings) / len(timings)
    tokens = seq
    tokens_per_s = tokens / min_s if min_s > 0 else None
    return BenchmarkResult(
        backend="torch",
        device=_torch_device_name(chosen),
        min_s=min_s,
        mean_s=mean_s,
        iterations=iterations,
        tokens_per_s=tokens_per_s,
    )


def run_jax(
    program: Program,
    base_inputs: Dict[str, Any],
    *,
    device_spec: str,
    iterations: int,
    warmup: int,
    seq: int,
) -> BenchmarkResult:
    try:
        import jax
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"jax unavailable: {exc}") from exc

    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")

    if device_spec == "auto":
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        tpu_devices = [d for d in devices if d.platform == "tpu"]
        if gpu_devices:
            target = gpu_devices[0]
        elif tpu_devices:
            target = tpu_devices[0]
        else:
            target = devices[0]
    else:
        matches = [d for d in devices if d.platform == device_spec]
        if not matches:
            raise RuntimeError(f"No JAX devices found for platform '{device_spec}'")
        target = matches[0]

    if target.platform not in ("gpu", "tpu"):
        raise RuntimeError("JAX benchmark requires a GPU/TPU device for meaningful results")

    runner = fuse_jax.compile(program)

    def to_jax(value: Any):
        return jax.device_put(value, device=target)

    inputs = {name: to_jax(value) for name, value in base_inputs.items()}

    def invoke():
        return runner(inputs=inputs)

    def sync(result: Dict[str, Any]):
        for value in result.values():
            if hasattr(value, "block_until_ready"):
                value.block_until_ready()
            else:
                jax.block_until_ready(value)

    timings = list(bench(invoke, sync, iterations=iterations, warmup=warmup))
    min_s = min(timings)
    mean_s = sum(timings) / len(timings)
    tokens = seq
    tokens_per_s = tokens / min_s if min_s > 0 else None
    device_name = f"{target.platform}:{target.id}"
    return BenchmarkResult(
        backend="jax",
        device=device_name,
        min_s=min_s,
        mean_s=mean_s,
        iterations=iterations,
        tokens_per_s=tokens_per_s,
    )


def format_results(results: Iterable[BenchmarkResult]) -> str:
    header = f"{'backend':<8} {'device':<12} {'min (ms)':>12} {'mean (ms)':>12} {'iters':>8} {'tokens/s':>12}"
    rows = [header]
    for result in results:
        min_ms = result.min_s * 1e3
        mean_ms = result.mean_s * 1e3
        tokens_per_s = result.tokens_per_s or math.nan
        rows.append(
            f"{result.backend:<8} {result.device:<12} {min_ms:12.3f} {mean_ms:12.3f} {result.iterations:8d} {tokens_per_s:12.2f}"
        )
    return "\n".join(rows)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Fuse attention fast paths on Torch/JAX GPU backends."
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "jax", "all"),
        default="all",
        help="Backend(s) to benchmark (default: all).",
    )
    parser.add_argument(
        "--device", default="auto", help="Device platform to use (e.g., cuda, gpu, mps, auto)."
    )
    parser.add_argument(
        "--seq", type=int, default=1024, help="Sequence length for queries (default: 1024)."
    )
    parser.add_argument(
        "--mem", type=int, default=None, help="Memory length for keys/values (default: seq)."
    )
    parser.add_argument(
        "--d-model", type=int, default=128, help="Model dimension / attention width (default: 128)."
    )
    parser.add_argument(
        "--value-dim", type=int, default=128, help="Value dimension (default: 128)."
    )
    parser.add_argument(
        "--seed", type=int, default=2024, help="Random seed for inputs (default: 2024)."
    )
    parser.add_argument(
        "--iterations", type=int, default=30, help="Timed iterations per backend (default: 30)."
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations to discard (default: 5)."
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    mem = args.mem if args.mem is not None else args.seq
    program = build_program()
    base_inputs = build_inputs(
        seq=args.seq,
        mem=mem,
        d_model=args.d_model,
        value_dim=args.value_dim,
        seed=args.seed,
    )

    results = []
    requested = ("torch", "jax") if args.backend == "all" else (args.backend,)
    for backend in requested:
        try:
            if backend == "torch":
                results.append(
                    run_torch(
                        program,
                        base_inputs,
                        device_spec=args.device,
                        iterations=args.iterations,
                        warmup=args.warmup,
                        seq=args.seq,
                    )
                )
            elif backend == "jax":
                results.append(
                    run_jax(
                        program,
                        base_inputs,
                        device_spec=args.device,
                        iterations=args.iterations,
                        warmup=args.warmup,
                        seq=args.seq,
                    )
                )
        except RuntimeError as exc:
            print(f"[skip] {backend}: {exc}", file=sys.stderr)

    if not results:
        print(
            "No backends were benchmarked. Ensure Torch/JAX are installed and a GPU device is available.",
            file=sys.stderr,
        )
        return 1

    print(format_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
