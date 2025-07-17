"""
Centralized ARC environment state definition.

This module contains the single, canonical definition of ArcEnvState using JAXTyping
for better type safety and JAX compatibility. This eliminates code duplication and
provides a single source of truth for state management.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp

from .types import JaxArcTask
from .utils.jax_types import (
    EpisodeDone,
    EpisodeIndex,
    GridArray,
    MaskArray,
    SelectionArray,
    SimilarityScore,
    StepCount,
)


@chex.dataclass
class ArcEnvState:
    """ARC environment state with full grid operations compatibility.
    
    This is the canonical definition of ArcEnvState using JAXTyping annotations
    for better type safety and documentation. All other modules should import
    this definition rather than defining their own.
    
    Attributes:
        task_data: The current ARC task data
        working_grid: Current grid being modified
        working_grid_mask: Valid cells mask for the working grid
        target_grid: Goal grid for current example
        step_count: Number of steps taken in current episode
        episode_done: Whether the current episode is complete
        current_example_idx: Which training example we're working on
        selected: Selection mask for operations
        clipboard: Grid data for copy/paste operations
        similarity_score: Grid similarity to target (0.0 to 1.0)
    """

    # Core ARC state
    task_data: JaxArcTask
    working_grid: GridArray  # Current grid being modified
    working_grid_mask: MaskArray  # Valid cells mask
    target_grid: GridArray  # Goal grid for current example

    # Episode management
    step_count: StepCount
    episode_done: EpisodeDone
    current_example_idx: EpisodeIndex  # Which training example we're working on

    # Grid operations fields
    selected: SelectionArray  # Selection mask for operations
    clipboard: GridArray  # For copy/paste operations
    similarity_score: SimilarityScore  # Grid similarity to target

    def __post_init__(self) -> None:
        """Validate ARC environment state structure.
        
        This validation ensures that all arrays have the correct shapes and types.
        It's designed to work with JAX transformations by gracefully handling
        cases where arrays don't have concrete shapes during tracing.
        """
        # Skip validation during JAX transformations
        if not hasattr(self.working_grid, "shape"):
            return

        try:
            # Validate grid shapes and types
            chex.assert_rank(self.working_grid, 2)
            chex.assert_rank(self.working_grid_mask, 2)
            chex.assert_rank(self.target_grid, 2)
            chex.assert_rank(self.selected, 2)
            chex.assert_rank(self.clipboard, 2)

            chex.assert_type(self.working_grid, jnp.integer)
            chex.assert_type(self.working_grid_mask, jnp.bool_)
            chex.assert_type(self.target_grid, jnp.integer)
            chex.assert_type(self.selected, jnp.bool_)
            chex.assert_type(self.clipboard, jnp.integer)
            chex.assert_type(self.similarity_score, jnp.floating)

            # Check consistent shapes
            chex.assert_shape(self.working_grid_mask, self.working_grid.shape)
            chex.assert_shape(self.target_grid, self.working_grid.shape)
            chex.assert_shape(self.selected, self.working_grid.shape)
            chex.assert_shape(self.clipboard, self.working_grid.shape)

            # Validate scalar types
            chex.assert_type(self.step_count, jnp.integer)
            chex.assert_type(self.episode_done, jnp.bool_)
            chex.assert_type(self.current_example_idx, jnp.integer)
            
            # Validate scalar shapes
            chex.assert_shape(self.step_count, ())
            chex.assert_shape(self.episode_done, ())
            chex.assert_shape(self.current_example_idx, ())
            chex.assert_shape(self.similarity_score, ())

        except (AttributeError, TypeError):
            # Skip validation during JAX transformations
            pass