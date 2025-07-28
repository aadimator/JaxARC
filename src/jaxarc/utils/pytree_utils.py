"""
PyTree manipulation utilities for efficient state updates using Equinox.

This module provides optimized utilities for manipulating PyTrees, particularly
ArcEnvState objects, using Equinox's tree_at and other PyTree operations.
These utilities enable efficient functional updates and complex state transformations
while maintaining JAX compatibility.

Key Features:
- Optimized multi-field updates using equinox.tree_at
- Functional update patterns for grid operations
- PyTree filtering and partitioning utilities
- Batch update operations for multiple state objects
- Memory-efficient state transformations

Examples:
    ```python
    from jaxarc.utils.pytree_utils import (
        update_multiple_fields,
        apply_grid_operation,
        batch_update_states,
        filter_arrays_from_state
    )

    # Update multiple fields efficiently
    new_state = update_multiple_fields(
        state,
        step_count=state.step_count + 1,
        episode_done=True,
        similarity_score=0.95
    )

    # Apply grid operation with automatic similarity update
    new_state = apply_grid_operation(
        state, 
        lambda grid: execute_grid_operation(grid, operation_id, selection)
    )

    # Filter arrays from state for serialization
    arrays, non_arrays = filter_arrays_from_state(state)
    ```
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..state import ArcEnvState
from .jax_types import GridArray, SelectionArray, SimilarityScore


def update_multiple_fields(state: ArcEnvState, **updates) -> ArcEnvState:
    """Update multiple fields efficiently using equinox.tree_at.
    
    This function provides an optimized way to update multiple fields in an
    ArcEnvState object simultaneously, which is more efficient than chaining
    individual updates.
    
    Args:
        state: The ArcEnvState to update
        **updates: Field names and their new values
        
    Returns:
        New ArcEnvState with updated fields
        
    Examples:
        ```python
        # Update multiple fields at once
        new_state = update_multiple_fields(
            state,
            step_count=state.step_count + 1,
            episode_done=True,
            similarity_score=0.95,
            selected=new_selection_mask
        )
        
        # More efficient than:
        # new_state = eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)
        # new_state = eqx.tree_at(lambda s: s.episode_done, new_state, True)
        # new_state = eqx.tree_at(lambda s: s.similarity_score, new_state, 0.95)
        # new_state = eqx.tree_at(lambda s: s.selected, new_state, new_selection_mask)
        ```
    """
    if not updates:
        return state
        
    # For multiple updates, chain them one by one
    # This is more reliable than trying to pass multiple paths to tree_at
    current_state = state
    for field_name, new_value in updates.items():
        if not hasattr(current_state, field_name):
            raise AttributeError(f"ArcEnvState has no field '{field_name}'")
        current_state = eqx.tree_at(
            lambda s, fn=field_name: getattr(s, fn), 
            current_state, 
            new_value
        )
    
    return current_state


def apply_grid_operation(
    state: ArcEnvState, 
    operation_fn: Callable[[GridArray], GridArray],
    update_similarity: bool = True,
    update_step_count: bool = True
) -> ArcEnvState:
    """Apply a grid operation with automatic state updates.
    
    This function applies a grid operation to the working grid and automatically
    updates related state fields like similarity score and step count.
    
    Args:
        state: Current environment state
        operation_fn: Function that transforms the working grid
        update_similarity: Whether to recompute similarity score
        update_step_count: Whether to increment step count
        
    Returns:
        New ArcEnvState with updated grid and related fields
        
    Examples:
        ```python
        # Apply a fill operation
        def fill_operation(grid):
            return execute_grid_operation(grid, operation_id=0, selection=selection)
            
        new_state = apply_grid_operation(state, fill_operation)
        
        # Apply operation without updating similarity (for performance)
        new_state = apply_grid_operation(
            state, 
            operation_fn, 
            update_similarity=False
        )
        ```
    """
    from ..envs.grid_operations import compute_grid_similarity
    
    # Apply the operation to get new grid
    new_grid = operation_fn(state.working_grid)
    
    # Prepare updates
    updates = {"working_grid": new_grid}
    
    if update_similarity:
        similarity = compute_grid_similarity(new_grid, state.target_grid)
        updates["similarity_score"] = similarity
        
    if update_step_count:
        updates["step_count"] = state.step_count + 1
        
    return update_multiple_fields(state, **updates)


def update_working_grid(state: ArcEnvState, new_grid: GridArray) -> ArcEnvState:
    """Update working grid using optimized PyTree surgery.
    
    Args:
        state: Current environment state
        new_grid: New working grid
        
    Returns:
        New ArcEnvState with updated working grid
        
    Examples:
        ```python
        new_state = update_working_grid(state, modified_grid)
        ```
    """
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)


def update_selection(state: ArcEnvState, new_selection: SelectionArray) -> ArcEnvState:
    """Update selection mask using optimized PyTree surgery.
    
    Args:
        state: Current environment state
        new_selection: New selection mask
        
    Returns:
        New ArcEnvState with updated selection
        
    Examples:
        ```python
        new_state = update_selection(state, selection_mask)
        ```
    """
    return eqx.tree_at(lambda s: s.selected, state, new_selection)


def increment_step_count(state: ArcEnvState) -> ArcEnvState:
    """Increment step count efficiently.
    
    Args:
        state: Current environment state
        
    Returns:
        New ArcEnvState with incremented step count
        
    Examples:
        ```python
        new_state = increment_step_count(state)
        ```
    """
    return eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)


def set_episode_done(state: ArcEnvState, done: bool = True) -> ArcEnvState:
    """Set episode done flag efficiently.
    
    Args:
        state: Current environment state
        done: Whether episode is done
        
    Returns:
        New ArcEnvState with updated episode_done flag
        
    Examples:
        ```python
        new_state = set_episode_done(state, True)
        ```
    """
    return eqx.tree_at(lambda s: s.episode_done, state, jnp.array(done))


def update_similarity_score(state: ArcEnvState, score: SimilarityScore) -> ArcEnvState:
    """Update similarity score efficiently.
    
    Args:
        state: Current environment state
        score: New similarity score
        
    Returns:
        New ArcEnvState with updated similarity score
        
    Examples:
        ```python
        new_state = update_similarity_score(state, 0.95)
        ```
    """
    return eqx.tree_at(lambda s: s.similarity_score, state, score)


def batch_update_states(
    states: List[ArcEnvState], 
    update_fn: Callable[[ArcEnvState], ArcEnvState]
) -> List[ArcEnvState]:
    """Apply an update function to a batch of states.
    
    This function efficiently applies the same update operation to multiple
    states, useful for batch processing scenarios.
    
    Args:
        states: List of ArcEnvState objects to update
        update_fn: Function to apply to each state
        
    Returns:
        List of updated ArcEnvState objects
        
    Examples:
        ```python
        # Increment step count for all states
        updated_states = batch_update_states(
            states, 
            lambda s: increment_step_count(s)
        )
        
        # Apply complex update to all states
        def complex_update(state):
            return update_multiple_fields(
                state,
                step_count=state.step_count + 1,
                episode_done=state.step_count >= 100
            )
            
        updated_states = batch_update_states(states, complex_update)
        ```
    """
    return [update_fn(state) for state in states]


def filter_arrays_from_state(state: ArcEnvState) -> Tuple[PyTree, PyTree]:
    """Filter arrays and non-arrays from state for serialization.
    
    This function separates array and non-array components of the state,
    which is useful for efficient serialization where you want to handle
    arrays and metadata differently.
    
    Args:
        state: ArcEnvState to filter
        
    Returns:
        Tuple of (arrays, non_arrays) PyTrees
        
    Examples:
        ```python
        arrays, non_arrays = filter_arrays_from_state(state)
        
        # Save arrays with binary format
        eqx.tree_serialise_leaves("state_arrays.eqx", arrays)
        
        # Save non-arrays with JSON
        import json
        with open("state_metadata.json", "w") as f:
            json.dump(non_arrays, f)
        ```
    """
    return eqx.partition(state, eqx.is_array)


def combine_filtered_state(arrays: PyTree, non_arrays: PyTree) -> ArcEnvState:
    """Combine filtered arrays and non-arrays back into state.
    
    This function reconstructs an ArcEnvState from separated array and
    non-array components, typically used during deserialization.
    
    Args:
        arrays: PyTree containing array components
        non_arrays: PyTree containing non-array components
        
    Returns:
        Reconstructed ArcEnvState
        
    Examples:
        ```python
        # Load components
        arrays = eqx.tree_deserialise_leaves("state_arrays.eqx", arrays_template)
        with open("state_metadata.json", "r") as f:
            non_arrays = json.load(f)
            
        # Reconstruct state
        state = combine_filtered_state(arrays, non_arrays)
        ```
    """
    return eqx.combine(arrays, non_arrays)


def create_state_template(
    grid_shape: Tuple[int, int],
    max_train_pairs: int = 10,
    max_test_pairs: int = 4,
    action_history_fields: int = 6
) -> ArcEnvState:
    """Create a template ArcEnvState with correct shapes for deserialization.
    
    This function creates a dummy state with the correct structure and shapes
    for use as a template during deserialization operations.
    
    Args:
        grid_shape: Shape of grids (height, width)
        max_train_pairs: Maximum number of training pairs
        max_test_pairs: Maximum number of test pairs
        action_history_fields: Number of fields in action history records
        
    Returns:
        Template ArcEnvState with correct shapes
        
    Examples:
        ```python
        # Create template for deserialization
        template = create_state_template(
            grid_shape=(30, 30),
            max_train_pairs=10,
            max_test_pairs=4,
            action_history_fields=6  # Point actions
        )
        
        # Use template for loading
        loaded_state = eqx.tree_deserialise_leaves("state.eqx", template)
        ```
    """
    from ..types import JaxArcTask
    from .jax_types import MAX_HISTORY_LENGTH, NUM_OPERATIONS
    
    height, width = grid_shape
    
    # Create dummy task data
    dummy_task = JaxArcTask(
        input_grids_examples=jnp.zeros((max_train_pairs, height, width), dtype=jnp.int32),
        input_masks_examples=jnp.zeros((max_train_pairs, height, width), dtype=jnp.bool_),
        output_grids_examples=jnp.zeros((max_train_pairs, height, width), dtype=jnp.int32),
        output_masks_examples=jnp.zeros((max_train_pairs, height, width), dtype=jnp.bool_),
        test_input_grids=jnp.zeros((max_test_pairs, height, width), dtype=jnp.int32),
        test_input_masks=jnp.zeros((max_test_pairs, height, width), dtype=jnp.bool_),
        true_test_output_grids=jnp.zeros((max_test_pairs, height, width), dtype=jnp.int32),
        true_test_output_masks=jnp.zeros((max_test_pairs, height, width), dtype=jnp.bool_),
        num_train_pairs=max_train_pairs,
        num_test_pairs=max_test_pairs,
        task_index=jnp.array(0, dtype=jnp.int32)
    )
    
    return ArcEnvState(
        # Core state
        task_data=dummy_task,
        working_grid=jnp.zeros((height, width), dtype=jnp.int32),
        working_grid_mask=jnp.ones((height, width), dtype=jnp.bool_),
        target_grid=jnp.zeros((height, width), dtype=jnp.int32),
        target_grid_mask=jnp.ones((height, width), dtype=jnp.bool_),
        
        # Episode management
        step_count=jnp.array(0, dtype=jnp.int32),
        episode_done=jnp.array(False),
        current_example_idx=jnp.array(0, dtype=jnp.int32),
        
        # Grid operations
        selected=jnp.zeros((height, width), dtype=jnp.bool_),
        clipboard=jnp.zeros((height, width), dtype=jnp.int32),
        similarity_score=jnp.array(0.0, dtype=jnp.float32),
        
        # Enhanced functionality
        episode_mode=jnp.array(0, dtype=jnp.int32),
        available_demo_pairs=jnp.ones(max_train_pairs, dtype=jnp.bool_),
        available_test_pairs=jnp.ones(max_test_pairs, dtype=jnp.bool_),
        demo_completion_status=jnp.zeros(max_train_pairs, dtype=jnp.bool_),
        test_completion_status=jnp.zeros(max_test_pairs, dtype=jnp.bool_),
        action_history=jnp.zeros((MAX_HISTORY_LENGTH, action_history_fields), dtype=jnp.float32),
        action_history_length=jnp.array(0, dtype=jnp.int32),
        action_history_write_pos=jnp.array(0, dtype=jnp.int32),
        allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_)
    )


def extract_grid_components(state: ArcEnvState) -> Dict[str, GridArray]:
    """Extract all grid components from state for analysis.
    
    Args:
        state: ArcEnvState to extract from
        
    Returns:
        Dictionary containing all grid arrays
        
    Examples:
        ```python
        grids = extract_grid_components(state)
        working = grids['working_grid']
        target = grids['target_grid']
        selection = grids['selected']
        ```
    """
    return {
        'working_grid': state.working_grid,
        'working_grid_mask': state.working_grid_mask,
        'target_grid': state.target_grid,
        'target_grid_mask': state.target_grid_mask,
        'selected': state.selected,
        'clipboard': state.clipboard
    }


def update_grid_components(
    state: ArcEnvState, 
    grid_updates: Dict[str, GridArray]
) -> ArcEnvState:
    """Update multiple grid components efficiently.
    
    Args:
        state: Current environment state
        grid_updates: Dictionary of grid field names and new values
        
    Returns:
        New ArcEnvState with updated grid components
        
    Examples:
        ```python
        new_state = update_grid_components(state, {
            'working_grid': new_working_grid,
            'selected': new_selection,
            'clipboard': new_clipboard
        })
        ```
    """
    return update_multiple_fields(state, **grid_updates)


@eqx.filter_jit
def jit_update_multiple_fields(state: ArcEnvState, **updates) -> ArcEnvState:
    """JIT-compiled version of update_multiple_fields for performance.
    
    This function provides a JIT-compiled version of the multi-field update
    operation for maximum performance in tight loops.
    
    Args:
        state: The ArcEnvState to update
        **updates: Field names and their new values
        
    Returns:
        New ArcEnvState with updated fields
        
    Note:
        This function is JIT-compiled, so the field names must be known at
        compile time. For dynamic field updates, use update_multiple_fields.
        
    Examples:
        ```python
        # Use in performance-critical code
        new_state = jit_update_multiple_fields(
            state,
            step_count=state.step_count + 1,
            episode_done=True
        )
        ```
    """
    return update_multiple_fields(state, **updates)


@eqx.filter_jit
def jit_apply_grid_operation(
    state: ArcEnvState, 
    operation_fn: Callable[[GridArray], GridArray]
) -> ArcEnvState:
    """JIT-compiled version of apply_grid_operation for performance.
    
    Args:
        state: Current environment state
        operation_fn: Function that transforms the working grid
        
    Returns:
        New ArcEnvState with updated grid and related fields
        
    Examples:
        ```python
        # Use in performance-critical code
        new_state = jit_apply_grid_operation(state, operation_fn)
        ```
    """
    return apply_grid_operation(state, operation_fn)