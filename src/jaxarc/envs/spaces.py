"""JAX-compatible action and observation spaces for ARC (Stoa-based)."""

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
    MultiDiscreteSpace,
    Space,
)

from jaxarc.types import NUM_COLORS, NUM_OPERATIONS

T = TypeVar("T")


# === ARC-SPECIFIC SPACES ===


class GridSpace(BoundedArraySpace):
    """
    ARC grid space with proper color bounds and correct integer sampling.
    """

    def __init__(self, max_height: int = 30, max_width: int = 30):
        super().__init__(
            shape=(max_height, max_width),
            dtype=jnp.int32,
            minimum=-1,  # Background/padding
            maximum=NUM_COLORS - 1,  # ARC colors 0-9
            name="arc_grid",
        )

        # Check that the dtype is an integer type
        if not jnp.issubdtype(self.dtype, jnp.integer):
            raise ValueError(
                f"GridSpace requires an integer dtype, but got {self.dtype}"
            )

    def sample(self, rng_key: PRNGKey) -> Array:
        """
        Sample a random grid of integers within the specified bounds.
        Overrides the parent method which samples floats.
        """
        return jax.random.randint(
            rng_key,
            shape=self.shape,
            minval=self.minimum,
            # maxval is exclusive for randint, so we add 1
            maxval=self.maximum + 1,
            dtype=self.dtype,
        )


class SelectionSpace(MultiDiscreteSpace):
    """Binary selection mask space"""

    def __init__(self, max_height: int = 30, max_width: int = 30):
        grid_shape = (max_height, max_width)

        # Each cell has 2 possible values (True/False)
        num_values_per_cell = jnp.full(grid_shape, 2)

        # Initialize the MultiDiscreteSpace parent
        super().__init__(
            num_values=num_values_per_cell, dtype=jnp.int32, name="selection_mask"
        )


class ARCActionSpace(DictSpace):
    """Complete ARC action space (operation + selection)."""

    def __init__(self, max_height: int = 30, max_width: int = 30):
        spaces = {
            "operation": DiscreteSpace(
                NUM_OPERATIONS, dtype=jnp.int32, name="operation"
            ),
            "selection": SelectionSpace(max_height, max_width),
        }
        super().__init__(spaces, "arc_action")


__all__ = [
    "ARCActionSpace",
    "BoundedArraySpace",
    "DictSpace",
    "DiscreteSpace",
    "GridSpace",
    "SelectionSpace",
    "Space",
]
