from __future__ import annotations

import pytest


def require_torch(min_version: str = "2.8.0"):
    torch = pytest.importorskip("torch", minversion=min_version)
    if getattr(torch, "__config__", None) is None:
        return torch
    try:
        torch.tensor([0], device="cpu")
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Torch backend unavailable: {exc}", allow_module_level=True)
    return torch
