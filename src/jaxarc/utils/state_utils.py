"""
State manipulation utilities for ArcEnvState.

This module provides functions for updating and transforming ArcEnvState
objects, using the core PyTree utilities where possible.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import equinox as eqx
import jax.numpy as jnp

from ..state import ArcEnvState
from .jax_types import (
    NUM_OPERATIONS,
    GridArray,
    SelectionArray,
    SimilarityScore,
)
from .pytree import update_multiple_fields


def apply_grid_operation(
    state: ArcEnvState,
    operation_fn: Callable[[GridArray], GridArray],
    update_similarity: bool = True,
    update_step_count: bool = True,
) -> ArcEnvState:
    """Apply a grid operation with automatic state updates."""
    from ..envs.grid_operations import compute_grid_similarity

    new_grid = operation_fn(state.working_grid)
    updates = {"working_grid": new_grid}

    if update_similarity:
        similarity = compute_grid_similarity(
            new_grid,
            state.working_grid_mask,
            state.target_grid,
            state.target_grid_mask,
        )
        updates["similarity_score"] = similarity

    if update_step_count:
        updates["step_count"] = state.step_count + 1

    return update_multiple_fields(state, **updates)


@eqx.filter_jit
def jit_apply_grid_operation(
    state: ArcEnvState, operation_fn: Callable[[GridArray], GridArray]
) -> ArcEnvState:
    """JIT-compiled version of apply_grid_operation for performance."""
    return apply_grid_operation(state, operation_fn)


def update_working_grid(state: ArcEnvState, new_grid: GridArray) -> ArcEnvState:
    """Update the working grid using optimized PyTree surgery."""
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


def update_selection(state: ArcEnvState, new_selection: SelectionArray) -> ArcEnvState:
    """Update the selection mask using optimized PyTree surgery."""
    return eqx.tree_at(lambda s: s.selected, state, new_selection)


def increment_step_count(state: ArcEnvState) -> ArcEnvState:
    """Increment the step count efficiently."""
    return eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)


# set_episode_done has been removed in the simplified state model.
# Episode termination is represented via TimeStep.step_type; state carries no done flag.


def update_similarity_score(state: ArcEnvState, score: SimilarityScore) -> ArcEnvState:
    """Update the similarity score efficiently."""
    return eqx.tree_at(lambda s: s.similarity_score, state, score)


def create_state_template(
    grid_shape: Tuple[int, int],
) -> ArcEnvState:
    """Create a template ArcEnvState aligned with simplified state layout."""
    height, width = grid_shape
    return ArcEnvState(
        working_grid=jnp.zeros((height, width), dtype=jnp.int32),
        working_grid_mask=jnp.ones((height, width), dtype=jnp.bool_),
        target_grid=jnp.zeros((height, width), dtype=jnp.int32),
        target_grid_mask=jnp.ones((height, width), dtype=jnp.bool_),
        selected=jnp.zeros((height, width), dtype=jnp.bool_),
        clipboard=jnp.zeros((height, width), dtype=jnp.int32),
        step_count=jnp.array(0, dtype=jnp.int32),
        allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
        similarity_score=jnp.array(0.0, dtype=jnp.float32),
        key=jnp.array([0, 0], dtype=jnp.int32),
    )


def extract_grid_components(state: ArcEnvState) -> Dict[str, GridArray]:
    """Extract all grid components from the state for analysis."""
    return {
        "working_grid": state.working_grid,
        "working_grid_mask": state.working_grid_mask,
        "target_grid": state.target_grid,
        "target_grid_mask": state.target_grid_mask,
        "selected": state.selected,
        "clipboard": state.clipboard,
    }


def update_grid_components(
    state: ArcEnvState, grid_updates: Dict[str, GridArray]
) -> ArcEnvState:
    """Update multiple grid components efficiently."""
    return update_multiple_fields(state, **grid_updates)
