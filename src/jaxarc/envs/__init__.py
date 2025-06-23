"""
JaxARC environments module.

This module provides reinforcement learning environments for solving
ARC (Abstraction and Reasoning Corpus) tasks, including both multi-agent
primitive environments and ARCLE-based environments.
"""

from __future__ import annotations

from .arcle_env import ARCLEEnvironment
from .arcle_operations import execute_arcle_operation
from .primitive_env import MultiAgentPrimitiveArcEnv, batched_env_step, load_config

__all__ = [
    # Primitive environment
    "MultiAgentPrimitiveArcEnv",
    "batched_env_step",
    "load_config",
    # ARCLE environment
    "ARCLEEnvironment",
    "execute_arcle_operation",
]
