"""
Essential state manipulation utilities for JaxARC.

This module provides core functions for updating State objects using
efficient PyTree operations.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ..state import State
from .jax_types import GridArray, SelectionArray, SimilarityScore
from .pytree import update_multiple_fields


@eqx.filter_jit
def validate_state_consistency(state: State) -> State:
    """Perform essential validation of the environment state.

    This function performs key consistency checks on the state that are
    critical for correct environment operation.

    Args:
        state: Environment state to validate

    Returns:
        The validated state (unchanged if valid)

    Raises:
        Runtime errors via equinox.error_if if validation fails
    """
    # Check that working and target grids have matching shapes
    working_shape = state.working_grid.shape
    target_shape = state.target_grid.shape
    shapes_match = jnp.array_equal(jnp.array(working_shape), jnp.array(target_shape))
    state = eqx.error_if(
        state, ~shapes_match, "Working and target grid shapes must match"
    )

    # Check that working grid mask matches working grid shape
    mask_shape = state.working_grid_mask.shape
    grid_mask_matches = jnp.array_equal(jnp.array(working_shape), jnp.array(mask_shape))
    state = eqx.error_if(
        state, ~grid_mask_matches, "Working grid mask shape must match working grid"
    )

    # Check basic constraints
    state = eqx.error_if(state, state.step_count < 0, "Step count cannot be negative")

    state = eqx.error_if(
        state,
        (state.similarity_score < 0.0) | (state.similarity_score > 1.0),
        "Similarity score must be in [0.0, 1.0]",
    )

    return state


def update_working_grid(state: State, new_grid: GridArray) -> State:
    """Update the working grid efficiently.

    Args:
        state: Current state
        new_grid: New working grid array

    Returns:
        Updated state with new working grid
    """
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


def update_selection(state: State, new_selection: SelectionArray) -> State:
    """Update the selection mask efficiently.

    Args:
        state: Current state
        new_selection: New selection mask array

    Returns:
        Updated state with new selection
    """
    return eqx.tree_at(lambda s: s.selected, state, new_selection)


def increment_step_count(state: State) -> State:
    """Increment the step count efficiently.

    Args:
        state: Current state

    Returns:
        Updated state with incremented step count
    """
    return eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)


def update_similarity_score(state: State, score: SimilarityScore) -> State:
    """Update the similarity score efficiently.

    Args:
        state: Current state
        score: New similarity score

    Returns:
        Updated state with new similarity score
    """
    return eqx.tree_at(lambda s: s.similarity_score, state, score)


def update_grid_and_similarity(
    state: State, new_grid: GridArray, new_score: SimilarityScore
) -> State:
    """Update both working grid and similarity score in one operation.

    Args:
        state: Current state
        new_grid: New working grid array
        new_score: New similarity score

    Returns:
        Updated state with new grid and score
    """
    return update_multiple_fields(
        state, working_grid=new_grid, similarity_score=new_score
    )
