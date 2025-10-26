try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _load_version
except ImportError:  # pragma: no cover
    from importlib_metadata import (  # type: ignore
        PackageNotFoundError,
    )
    from importlib_metadata import (
        version as _load_version,
    )

from . import logic, nn, pgm
from .core.cache import CacheManager
from .core.evaluator_numpy import ExecutionConfig
from .core.policies import (
    InMemoryWeightStore,
    LoRAPolicy,
    ManifestWeightStore,
    QuantizationPolicy,
    RuntimePolicies,
    ShardingPolicy,
)
from .core.program import Program
from .core.temperature import (
    ConstantSchedule,
    LinearRampSchedule,
    PiecewiseSchedule,
    make_schedule,
)
from .inference.grad_builder import GradientProgram, generate_gradient_program
from .inference.tree_program import (
    Factor,
    TreeFactorGraph,
    TreeProgram,
    Variable,
    conditional_probability,
)
from .interop import (
    from_pytorch,
    from_safetensors,
    to_onnx,
    to_torchscript,
)
from .package import build_package
from .training import gradients_for_program

try:
    __version__ = _load_version("fuse-ai")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Program",
    "ExecutionConfig",
    "RuntimePolicies",
    "InMemoryWeightStore",
    "ManifestWeightStore",
    "from_pytorch",
    "from_safetensors",
    "to_torchscript",
    "to_onnx",
    "build_package",
    "ShardingPolicy",
    "QuantizationPolicy",
    "LoRAPolicy",
    "ConstantSchedule",
    "LinearRampSchedule",
    "PiecewiseSchedule",
    "make_schedule",
    "CacheManager",
    "TreeFactorGraph",
    "TreeProgram",
    "Variable",
    "Factor",
    "conditional_probability",
    "generate_gradient_program",
    "GradientProgram",
    "gradients_for_program",
    "nn",
    "logic",
    "pgm",
    "torch",
    "jax",
    "__version__",
]


class torch:
    @staticmethod
    def compile(*args, **kwargs):
        from .torch_backend.compile import compile as torch_compile

        return torch_compile(*args, **kwargs)


class jax:
    @staticmethod
    def compile(*args, **kwargs):
        from .jax_backend.compile import compile as jax_compile

        return jax_compile(*args, **kwargs)
