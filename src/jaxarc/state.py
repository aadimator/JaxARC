"""
Centralized ARC environment state definition using Equinox.

This module contains the single, canonical definition of ArcEnvState using Equinox Module
for better JAX integration, automatic PyTree registration, and improved type safety.
This eliminates code duplication and provides a single source of truth for state management.

Key Features:
- Equinox Module for automatic PyTree registration
- JAXTyping annotations for precise type safety
- Automatic validation through Equinox patterns
- Better JAX transformation compatibility
- Cleaner functional patterns for state updates

Examples:
    ```python
    import jax
    from jaxarc.state import ArcEnvState
    
    # Create state (typically done by environment)
    state = ArcEnvState(
        task_data=task,
        working_grid=grid,
        working_grid_mask=mask,
        # ... other fields
    )
    
    # Update state using Equinox patterns
    new_state = eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)
    
    # Or use replace method for multiple updates
    new_state = state.replace(
        step_count=state.step_count + 1,
        episode_done=True
    )
    ```
"""

from __future__ import annotations

import equinox as eqx
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


class ArcEnvState(eqx.Module):
    """ARC environment state with Equinox Module for better JAX integration.
    
    This is the canonical definition of ArcEnvState using Equinox Module for automatic
    PyTree registration and JAXTyping annotations for better type safety and documentation.
    All other modules should import this definition rather than defining their own.
    
    Equinox provides several advantages over chex dataclass:
    - Automatic PyTree registration for JAX transformations
    - Better error messages for shape mismatches
    - Cleaner functional patterns for state updates
    - Improved compatibility with JAX transformations (jit, vmap, pmap)
    - Built-in validation through __check_init__
    
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
        
    Examples:
        ```python
        # Create new state
        state = ArcEnvState(
            task_data=task,
            working_grid=grid,
            working_grid_mask=mask,
            target_grid=target,
            step_count=jnp.array(0),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0),
            selected=jnp.zeros_like(grid, dtype=bool),
            clipboard=jnp.zeros_like(grid),
            similarity_score=jnp.array(0.0)
        )
        
        # Update state using Equinox tree_at
        new_state = eqx.tree_at(
            lambda s: s.step_count, 
            state, 
            state.step_count + 1
        )
        
        # Update multiple fields
        new_state = eqx.tree_at(
            lambda s: (s.step_count, s.episode_done),
            state,
            (state.step_count + 1, jnp.array(True))
        )
        ```
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

    def __check_init__(self) -> None:
        """Equinox validation method for state structure.
        
        This validation ensures that all arrays have the correct shapes and types.
        It's designed to work with JAX transformations by gracefully handling
        cases where arrays don't have concrete shapes during tracing.
        
        Equinox automatically calls this method during module creation, providing
        better error messages and validation than manual __post_init__ methods.
        """
        # Skip validation during JAX transformations
        if not hasattr(self.working_grid, "shape"):
            return

        try:
            # Import chex here to avoid circular imports
            import chex
            
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
    
    def replace(self, **kwargs) -> 'ArcEnvState':
        """Create a new state with updated fields.
        
        This method provides a convenient way to update multiple fields at once,
        similar to the dataclass replace method but using Equinox patterns.
        
        Args:
            **kwargs: Fields to update with their new values
            
        Returns:
            New ArcEnvState with updated fields
            
        Examples:
            ```python
            new_state = state.replace(
                step_count=state.step_count + 1,
                episode_done=True
            )
            ```
        """
        # Get current field values
        current_values = {}
        for field_name in self.__dataclass_fields__:
            current_values[field_name] = getattr(self, field_name)
        
        # Update with provided kwargs
        current_values.update(kwargs)
        
        # Create new instance
        return ArcEnvState(**current_values)
    
    @property
    def __dataclass_fields__(self) -> dict:
        """Property to mimic dataclass fields for replace method."""
        return {
            'task_data': None,
            'working_grid': None,
            'working_grid_mask': None,
            'target_grid': None,
            'step_count': None,
            'episode_done': None,
            'current_example_idx': None,
            'selected': None,
            'clipboard': None,
            'similarity_score': None,
        }