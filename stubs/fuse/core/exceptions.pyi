from _typeshed import Incomplete

class FuseError(Exception): ...

class ParseError(FuseError, ValueError):
    line: Incomplete
    column: Incomplete
    line_text: Incomplete
    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
        line_text: str | None = None,
    ) -> None: ...

class ShapeError(FuseError, ValueError):
    line: Incomplete
    column: Incomplete
    line_text: Incomplete
    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
        line_text: str | None = None,
    ) -> None: ...

class BackendError(FuseError, RuntimeError): ...
class CacheError(FuseError, RuntimeError): ...
