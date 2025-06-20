"""
JaxARC environments module.

This module provides multi-agent reinforcement learning environments for solving
ARC (Abstraction and Reasoning Corpus) tasks.
"""

from __future__ import annotations

from .primitive_env import MultiAgentPrimitiveArcEnv, batched_env_step, load_config

__all__ = [
    "MultiAgentPrimitiveArcEnv",
    "batched_env_step",
    "load_config",
]
