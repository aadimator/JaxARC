"""JAX-compatible action and observation spaces for ARC (Stoa-based).
"""

from __future__ import annotations

from typing import TypeVar

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jax import Array
from stoa.spaces import (
    BoundedArraySpace,
    DictSpace,
    DiscreteSpace,
    Space,
)

from jaxarc.types import NUM_COLORS, NUM_OPERATIONS

T = TypeVar("T")


# === ARC-SPECIFIC SPACES ===

class GridSpace(BoundedArraySpace):
    """ARC grid space with proper color bounds."""
    
    def __init__(self, max_height: int = 30, max_width: int = 30):
        super().__init__(
            shape=(max_height, max_width),
            dtype=jnp.int32,
            minimum=-1,  # Background/padding
            maximum=NUM_COLORS - 1,  # ARC colors 0-9
            name="arc_grid"
        )


class SelectionSpace(BoundedArraySpace):
    """Binary selection mask space."""
    
    def __init__(self, max_height: int = 30, max_width: int = 30):
        super().__init__(
            shape=(max_height, max_width),
            dtype=jnp.bool_,
            minimum=False,
            maximum=True,
            name="selection_mask"
        )


class ARCActionSpace(DictSpace):
    """Complete ARC action space (operation + selection)."""
    
    def __init__(self, max_height: int = 30, max_width: int = 30):
        spaces = {
            "operation": DiscreteSpace(NUM_OPERATIONS, dtype=jnp.int32, name="operation"),
            "selection": SelectionSpace(max_height, max_width)
        }
        super().__init__(spaces, "arc_action")


# === LEGACY COMPATIBILITY (minimal) ===

class MultiBinary(BoundedArraySpace):
    """Legacy compatibility for existing MultiBinary usage."""
    
    def __init__(self, n: int | tuple[int, ...]):
        shape = (n,) if isinstance(n, int) else tuple(n)
        super().__init__(shape=shape, dtype=jnp.int32, minimum=0, maximum=1, name="multibinary")
        self.n = shape
    
    def sample(self, key: PRNGKey) -> Array:
        return jax.random.bernoulli(key, 0.5, shape=self.shape).astype(jnp.int32)





__all__ = [
    "ARCActionSpace",
    "BoundedArraySpace", 
    "DictSpace",
    "DiscreteSpace", 
    "GridSpace", 
    "MultiBinary",  # Legacy compatibility
    "SelectionSpace", 
    "Space", 
]
