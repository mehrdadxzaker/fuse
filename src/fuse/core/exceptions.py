from __future__ import annotations

from typing import Optional


class FuseError(Exception):
    """Base class for Fuse-specific exceptions."""


class ParseError(FuseError, ValueError):
    def __init__(
        self,
        message: str,
        *,
        line: Optional[int] = None,
        column: Optional[int] = None,
        line_text: Optional[str] = None,
    ):
        detail = _format_location(line, column, line_text)
        super().__init__(f"{message}{detail}")
        self.line = line
        self.column = column
        self.line_text = line_text


class ShapeError(FuseError, ValueError):
    def __init__(
        self,
        message: str,
        *,
        line: Optional[int] = None,
        column: Optional[int] = None,
        line_text: Optional[str] = None,
    ):
        detail = _format_location(line, column, line_text)
        super().__init__(f"{message}{detail}")
        self.line = line
        self.column = column
        self.line_text = line_text


class BackendError(FuseError, RuntimeError):
    pass


class CacheError(FuseError, RuntimeError):
    pass


def _format_location(
    line: Optional[int],
    column: Optional[int],
    line_text: Optional[str],
) -> str:
    if line is None and column is None:
        return ""
    location = []
    if line is not None:
        location.append(f"line {line}")
    if column is not None:
        location.append(f"col {column}")
    location_str = f" ({', '.join(location)})"
    if line_text is None or column is None or column < 1:
        return location_str
    caret = " " * (column - 1) + "^"
    return f"{location_str}\n  {line_text}\n  {caret}"
