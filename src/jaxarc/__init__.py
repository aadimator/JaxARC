"""
Copyright (c) 2025 Aadam. All rights reserved.

JaxARC: MARL environment for ARC dataset in JAX
"""

from __future__ import annotations

from ._version import version as __version__
from .envs import ArcEnvironment

__all__ = [
    "ArcEnvironment",
    "__version__",
]
