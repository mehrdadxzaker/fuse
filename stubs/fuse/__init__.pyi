from _typeshed import Incomplete

from . import logic as logic
from . import nn as nn
from . import pgm as pgm
from .core.cache import CacheManager as CacheManager
from .core.evaluator_numpy import ExecutionConfig as ExecutionConfig
from .core.policies import InMemoryWeightStore as InMemoryWeightStore
from .core.policies import LoRAPolicy as LoRAPolicy
from .core.policies import ManifestWeightStore as ManifestWeightStore
from .core.policies import QuantizationPolicy as QuantizationPolicy
from .core.policies import RuntimePolicies as RuntimePolicies
from .core.policies import ShardingPolicy as ShardingPolicy
from .core.program import Program as Program
from .core.temperature import ConstantSchedule as ConstantSchedule
from .core.temperature import LinearRampSchedule as LinearRampSchedule
from .core.temperature import PiecewiseSchedule as PiecewiseSchedule
from .core.temperature import make_schedule as make_schedule
from .inference.grad_builder import GradientProgram as GradientProgram
from .inference.grad_builder import generate_gradient_program as generate_gradient_program
from .inference.tree_program import Factor as Factor
from .inference.tree_program import TreeFactorGraph as TreeFactorGraph
from .inference.tree_program import TreeProgram as TreeProgram
from .inference.tree_program import Variable as Variable
from .inference.tree_program import conditional_probability as conditional_probability
from .interop import from_pytorch as from_pytorch
from .interop import from_safetensors as from_safetensors
from .interop import to_onnx as to_onnx
from .interop import to_torchscript as to_torchscript
from .package import build_package as build_package
from .training import gradients_for_program as gradients_for_program

__version__: Incomplete

class torch:
    @staticmethod
    def compile(*args: object, **kwargs: object) -> object: ...

class jax:
    @staticmethod
    def compile(*args: object, **kwargs: object) -> object: ...
