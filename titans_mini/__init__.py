"""Titans-Mini: a streaming engine over a swappable test-time memory core."""

from .config import TitansConfig
from .core import MemoryCore, build_memory_core, memory_core_names, register_memory_core
from .engine import ModularTitansEngine

# Importing the cores is what registers them with the factory.
from .cores.mlp_core import MLPMemoryCore
from .cores.vector_core import VectorMemoryCore

__all__ = [
    "TitansConfig",
    "MemoryCore",
    "MLPMemoryCore",
    "VectorMemoryCore",
    "ModularTitansEngine",
    "build_memory_core",
    "memory_core_names",
    "register_memory_core",
]
