from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _typeshed import Incomplete

from .evaluator_numpy import ExecutionConfig as ExecutionConfig
from .exceptions import CacheError as CacheError
from .policies import RuntimePolicies as RuntimePolicies

CACHE_VERSION: str

def compute_program_hash(src: str) -> str: ...

@dataclass
class CacheRecord:
    payload: Any
    metadata: dict[str, Any]

class CacheManager:
    root: Incomplete
    def __init__(self, cache_dir: str) -> None: ...
    def path_for(self, backend: str, key: str) -> Path: ...
    def load(self, backend: str, key: str) -> CacheRecord | None: ...
    def store(
        self, backend: str, key: str, payload: Any, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def write_metadata(self, backend: str, key: str, metadata: dict[str, Any]) -> None: ...

def cache_fingerprint(*, program_src: str, backend: str, artifact: str | None = None, device: str | None = None, execution_config: ExecutionConfig | None = None, policies: RuntimePolicies | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]: ...
def cache_key_from_fingerprint(fingerprint: dict[str, Any]) -> str: ...
def build_cache_key(*, program_src: str, backend: str, artifact: str | None = None, device: str | None = None, execution_config: ExecutionConfig | None = None, policies: RuntimePolicies | None = None, extra: dict[str, Any] | None = None) -> str: ...
