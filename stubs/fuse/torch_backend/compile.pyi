from typing import Any

from fuse.core.cache import CacheManager
from fuse.core.evaluator_numpy import ExecutionConfig
from fuse.core.policies import RuntimePolicies

def compile(
    program: Any,
    device: str = ...,
    cache_manager: CacheManager | None = None,
    execution_config: ExecutionConfig | None = None,
    policies: RuntimePolicies | None = None,
    **_: Any,
) -> Any: ...


__all__ = ["compile"]
