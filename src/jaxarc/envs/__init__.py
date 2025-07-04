"""
JaxARC environments module.

This module provides reinforcement learning environments for solving
ARC (Abstraction and Reasoning Corpus) tasks using JAX-compatible
single-agent environments with grid operations.
"""

from __future__ import annotations

from .arc_base import ArcEnvironment, ArcEnvState
from .grid_operations import execute_grid_operation

__all__ = [
    # ARC environments
    "ArcEnvironment",
    "ArcEnvState",
    "execute_grid_operation",
]
