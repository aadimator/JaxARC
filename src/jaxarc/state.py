"""
Centralized environment state definition using Equinox.

This module defines the simplified, generic `State` used throughout JaxARC.
Static configuration has been removed from state and moved to EnvParams.
The legacy ArcEnvState has been fully removed. Use `State` (or the `ArcState`
alias) instead.

Key properties:
- Equinox Module for automatic PyTree registration
- JAXTyping annotations for precise type safety
- Purely dynamic fields that change during episodes
- JAX transformation compatibility (jit, vmap, pmap)
- Optional `carry` following the XLand-Minigrid pattern for extensions
"""

from __future__ import annotations

from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp

from jaxarc.utils.jax_types import (
    GridArray,
    MaskArray,
    OperationMask,
    PRNGKey,
    SelectionArray,
    SimilarityScore,
    StepCount,
)


EnvCarryT = TypeVar("EnvCarryT")


class State(eqx.Module, Generic[EnvCarryT]):
    """Environment state with optional carry (XLand-Minigrid pattern).

    Contains only truly dynamic variables that change during episodes.
    Static configuration is moved to EnvParams.
    """

    # Core dynamic grid state
    working_grid: GridArray            # Current grid being modified
    working_grid_mask: MaskArray       # Valid cells mask
    input_grid: GridArray              # Original input grid for current pair
    input_grid_mask: MaskArray         # Valid cells mask for input grid
    target_grid: GridArray             # Goal grid for current example
    target_grid_mask: MaskArray        # Valid cells mask for target grid

    # Grid operations state
    selected: SelectionArray           # Selection mask for operations
    clipboard: GridArray               # For copy/paste operations

    # Episode progress tracking
    step_count: StepCount              # Current step number

    # Dynamic control state
    allowed_operations_mask: OperationMask  # Dynamic operation filtering

    # Similarity tracking score (required array)
    similarity_score: SimilarityScore

    # PRNG key for environment randomness (auto-reset compatibility)
    key: PRNGKey

    # Optional carry for extensions (proper XLand-Minigrid pattern)
    carry: EnvCarryT | None = None

    def __check_init__(self) -> None:
        """Validate dynamic state structure."""
        # During tracing, attributes may be placeholders; guard accordingly
        if not hasattr(self.working_grid, "shape"):
            return

        try:
            import chex

            # Validate grid ranks
            chex.assert_rank(self.working_grid, 2)
            chex.assert_rank(self.working_grid_mask, 2)
            chex.assert_rank(self.input_grid, 2)
            chex.assert_rank(self.input_grid_mask, 2)
            chex.assert_rank(self.target_grid, 2)
            chex.assert_rank(self.target_grid_mask, 2)
            chex.assert_rank(self.selected, 2)
            chex.assert_rank(self.clipboard, 2)
            chex.assert_rank(self.allowed_operations_mask, 1)

            # Validate consistent shapes
            chex.assert_shape(self.working_grid_mask, self.working_grid.shape)
            chex.assert_shape(self.input_grid, self.working_grid.shape)
            chex.assert_shape(self.input_grid_mask, self.working_grid.shape)
            chex.assert_shape(self.target_grid, self.working_grid.shape)
            chex.assert_shape(self.target_grid_mask, self.working_grid.shape)
            chex.assert_shape(self.selected, self.working_grid.shape)
            chex.assert_shape(self.clipboard, self.working_grid.shape)

            # Validate scalars/types
            chex.assert_type(self.step_count, jnp.integer)

        except (AttributeError, TypeError):
            # Gracefully skip during tracing
            pass


# Type aliases for convenience
BaseState = State[None]   # No carry
ArcState = BaseState      # Our standard simplified state
