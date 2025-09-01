"""
State manipulation utilities for ArcEnvState.

This module provides functions for updating and transforming ArcEnvState
objects, using the core PyTree utilities where possible.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from ..state import State
from .jax_types import (
    NUM_OPERATIONS,
    GridArray,
    SelectionArray,
    SimilarityScore,
)
from .pytree import update_multiple_fields


def apply_grid_operation(
    state: State,
    operation_fn: Callable[[GridArray], GridArray],
    update_similarity: bool = True,
    update_step_count: bool = True,
) -> State:
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
    state: State, operation_fn: Callable[[GridArray], GridArray]
) -> State:
    """JIT-compiled version of apply_grid_operation for performance."""
    return apply_grid_operation(state, operation_fn)


def update_working_grid(state: State, new_grid: GridArray) -> State:
    """Update the working grid using optimized PyTree surgery."""
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


def update_selection(state: State, new_selection: SelectionArray) -> State:
    """Update the selection mask using optimized PyTree surgery."""
    return eqx.tree_at(lambda s: s.selected, state, new_selection)


def increment_step_count(state: State) -> State:
    """Increment the step count efficiently."""
    return eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)


# set_episode_done has been removed in the simplified state model.
# Episode termination is represented via TimeStep.step_type; state carries no done flag.


def update_similarity_score(state: State, score: SimilarityScore) -> State:
    """Update the similarity score efficiently."""
    return eqx.tree_at(lambda s: s.similarity_score, state, score)


def create_state_template(
    grid_shape: Tuple[int, int],
) -> State:
    """Create a template ArcEnvState aligned with simplified state layout.

    This template now populates the explicit `input_grid`/`input_grid_mask`
    fields as well as `task_idx`/`pair_idx` so that created states match the
    updated `State` signature.
    """
    height, width = grid_shape
    return State(
        # Core grids
        working_grid=jnp.zeros((height, width), dtype=jnp.int32),
        working_grid_mask=jnp.ones((height, width), dtype=jnp.bool_),
        # Provide sensible defaults for input/target (same shape as working grid)
        input_grid=jnp.zeros((height, width), dtype=jnp.int32),
        input_grid_mask=jnp.ones((height, width), dtype=jnp.bool_),
        target_grid=jnp.zeros((height, width), dtype=jnp.int32),
        target_grid_mask=jnp.ones((height, width), dtype=jnp.bool_),
        # Grid operation helpers
        selected=jnp.zeros((height, width), dtype=jnp.bool_),
        clipboard=jnp.zeros((height, width), dtype=jnp.int32),
        # Episode progress and tracking
        step_count=jnp.array(0, dtype=jnp.int32),
        # Task/pair tracking (defaults: unknown task, pair 0)
        task_idx=jnp.array(-1, dtype=jnp.int32),
        pair_idx=jnp.array(0, dtype=jnp.int32),
        # Dynamic action mask and scoring
        allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
        similarity_score=jnp.array(0.0, dtype=jnp.float32),
        # PRNG key placeholder
        key=jnp.array([0, 0], dtype=jnp.int32),
    )


# -------------------------------------------------------------------------
# Helpers for extracting task / pair data from an EnvParams buffer
# -------------------------------------------------------------------------


def get_pair_from_task(
    task_data: Any, pair_idx: jnp.ndarray, episode_mode: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract input and target grids (and their masks) for the given pair index.

    Supports both train (demo) and test modes. In test mode the returned target is
    masked (background) consistent with evaluation behavior.

    Args:
        task_data: A JaxArcTask-like pytree (with fields for train/test arrays).
        pair_idx: Scalar JAX integer selecting the pair within the task.
        episode_mode: 0 for training (use demo outputs), 1 for test (mask target).

    Returns:
        (input_grid, input_mask, target_grid, target_mask)
    """

    # Train-mode accessors
    def train_pair():
        input_grid = task_data.input_grids_examples[pair_idx]
        input_mask = task_data.input_masks_examples[pair_idx]
        target_grid = task_data.output_grids_examples[pair_idx]
        target_mask = task_data.output_masks_examples[pair_idx]
        return input_grid, input_mask, target_grid, target_mask

    # Test-mode accessors
    def test_pair():
        input_grid = task_data.test_input_grids[pair_idx]
        input_mask = task_data.test_input_masks[pair_idx]
        # Masked target: fill with background and zero mask
        background = getattr(task_data, "background_color", 0)
        target_grid = jnp.full_like(input_grid, background)
        target_mask = jnp.zeros_like(input_mask)
        return input_grid, input_mask, target_grid, target_mask

    return jax.lax.cond(
        jnp.asarray(episode_mode, dtype=jnp.int32) == jnp.asarray(0, dtype=jnp.int32),
        train_pair,
        test_pair,
    )


def extract_grid_components(state: State) -> Dict[str, GridArray]:
    """Extract all grid components from the state for analysis."""
    return {
        "working_grid": state.working_grid,
        "working_grid_mask": state.working_grid_mask,
        "target_grid": state.target_grid,
        "target_grid_mask": state.target_grid_mask,
        "selected": state.selected,
        "clipboard": state.clipboard,
    }


def update_grid_components(state: State, grid_updates: Dict[str, GridArray]) -> State:
    """Update multiple grid components efficiently."""
    return update_multiple_fields(state, **grid_updates)
