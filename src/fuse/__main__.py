from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .core.evaluator_numpy import ExecutionConfig
from .core.program import Program


def _load_program(path: Path) -> Program:
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Program file not found: {path}") from exc
    return Program(source)


def _write_output(path: Path, tensor: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).lower().endswith(".npy"):
        np.save(path, np.asarray(tensor))
    elif str(path).lower().endswith(".npz"):
        np.savez(path, np.asarray(tensor))
    elif str(path).lower().endswith(".json"):
        path.write_text(json.dumps(np.asarray(tensor).tolist(), indent=2), encoding="utf-8")
    elif str(path).lower().endswith(".jsonl"):
        arr = np.asarray(tensor)
        with path.open("w", encoding="utf-8") as handle:
            for row in arr.reshape(arr.shape[0], -1):
                handle.write(json.dumps(row.tolist()) + "\n")
    else:
        np.save(path, np.asarray(tensor))


def _run(program_path: Path, backend: str, out: Optional[Path]) -> None:
    program = _load_program(program_path)
    runner = program.compile(backend=backend, config=ExecutionConfig())
    outputs: Dict[str, Any] = runner()
    if not outputs:
        raise SystemExit("Program produced no outputs; ensure an `export` is declared.")
    if out is None:
        first_name, tensor = next(iter(outputs.items()))
        np.set_printoptions(suppress=True)
        print(f"# {first_name}")
        print(np.asarray(tensor))
        return
    if len(outputs) > 1:
        out_data = {name: np.asarray(value).tolist() for name, value in outputs.items()}
        out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    else:
        tensor = next(iter(outputs.values()))
        _write_output(out, tensor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse command line utilities")
    subparsers = parser.add_subparsers(dest="cmd")

    run_parser = subparsers.add_parser("run", help="Execute a .fuse program")
    run_parser.add_argument("program", type=Path, help="Path to .fuse program file")
    run_parser.add_argument(
        "--backend",
        default="numpy",
        choices=["numpy", "torch", "jax"],
        help="Execution backend to use (default: numpy)",
    )
    run_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path (.npy/.npz/.json/.jsonl). If omitted, prints first export",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        _run(args.program, backend=args.backend, out=args.out)
        return

    parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
