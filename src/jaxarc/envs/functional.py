"""
Functional API for JaxARC environments.

This module provides pure functional implementations of the ARC environment
using EnvParams (static environment parameters) decoupled from framework configs.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxarc.types import EPISODE_MODE_TRAIN

from ..state import State
from ..types import (
    NUM_OPERATIONS,
    EnvParams,
    JaxArcTask,
    ObservationArray,
    PRNGKey,
    RewardValue,
    StepType,
    TimeStep,
)
from ..utils.state_utils import (
    increment_step_count,
    update_selection,
)
from .actions import (
    Action,
    action_handler,
    create_action,
    filter_invalid_operation,
)
from .grid_initialization import initialize_working_grids
from .grid_operations import compute_grid_similarity, execute_grid_operation

# ============================================================================
# OBSERVATION FUNCTIONS (moved from observation.py)
# ============================================================================


def _get_observation(state: State, _unused: Any) -> ObservationArray:
    """Extract observation from state.

    Returns the working grid with a channel dimension added (H, W) -> (H, W, 1).
    This ensures compatibility with observation wrappers that concatenate channels.
    """
    return jnp.expand_dims(state.working_grid, axis=-1)


def create_observation(state: State, params: EnvParams) -> ObservationArray:
    """Create agent observation using EnvParams-based API.

    Kept separate to support new functional signatures while maintaining legacy compatibility.
    """
    return _get_observation(state, params)


# ============================================================================
# REWARD FUNCTIONS (moved from reward.py)
# ============================================================================


def _calculate_reward(
    old_state: State,
    new_state: State,
    params: EnvParams,
    *,
    is_submit_step: jnp.ndarray | None = None,
    episode_mode: int | None = None,
) -> RewardValue:
    """Submit-aware reward with optional episode mode selection.

    Reward components:
    - **Similarity reward** (training only): Shaped reward for incremental progress (similarity_weight * delta_similarity)
    - **Success bonus**: Large bonus awarded ONLY when agent submits with 100% similarity
    - **Step penalty**: Small penalty applied every step to encourage efficiency
    - **Unsolved submission penalty**: Penalty for submitting without reaching 100% similarity

    Key behavior:
    - Success bonus requires BOTH conditions: similarity >= 1.0 AND submit action performed
    - Similarity improvements are rewarded independently on every step (during training)
    - Episode mode (train vs eval) determines whether similarity shaping is included

    Args:
        old_state: Previous environment state
        new_state: New environment state after action
        params: Environment parameters containing reward configuration
        is_submit_step: Optional boolean array indicating if this step is a Submit action
        episode_mode: Optional episode mode (0=train, 1=test). When None, treated as train.

    Returns:
        JAX scalar array containing the calculated reward
    """
    reward_cfg = params.reward

    # Resolve optional flags with safe defaults
    submit_flag = is_submit_step if is_submit_step is not None else jnp.asarray(False)
    is_training = (
        jnp.asarray(True) if episode_mode is None else jnp.asarray(episode_mode == 0)
    )

    # 1) Components
    similarity_improvement = new_state.similarity_score - old_state.similarity_score

    # Success condition: Task is solved (similarity >= 1.0) AND submit action is performed
    is_solved = (new_state.similarity_score >= 1.0) & submit_flag

    step_penalty = jnp.asarray(reward_cfg.step_penalty, dtype=jnp.float32)
    similarity_reward = reward_cfg.similarity_weight * similarity_improvement

    # Award success bonus only when task is solved (similarity 100%) AND submit is performed
    success_bonus = jnp.where(is_solved, reward_cfg.success_bonus, 0.0)

    # Penalty for submitting without solving
    submission_penalty = jnp.where(
        submit_flag & ~is_solved, reward_cfg.unsolved_submission_penalty, 0.0
    )

    # 2) Mode-specific totals (training includes similarity shaping)
    training_reward = (
        similarity_reward + step_penalty + success_bonus + submission_penalty
    )
    evaluation_reward = step_penalty + success_bonus + submission_penalty

    # 3) Select by mode
    return jnp.where(is_training, training_reward, evaluation_reward)


# JAX-compatible step info structure - replaces dict for performance.
class StepInfo(eqx.Module):
    """Step info as an Equinox Module for PyTree compatibility.

    Simplified to include only fields available in the minimal state model.
    """

    similarity: jax.Array
    similarity_improvement: jax.Array
    operation_type: jax.Array
    step_count: jax.Array
    success: jax.Array


def _initialize_grids(
    task_data: JaxArcTask,
    selected_pair_idx: jnp.ndarray,
    episode_mode: int,
    params: EnvParams,
    key: PRNGKey,
    initial_pair_idx: int | None = None,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Initialize grids with diverse initialization strategies - enhanced helper function.

    This enhanced helper function sets up the initial, target, and mask grids based on the
    episode mode, selected pair, and diverse initialization configuration. It supports
    multiple initialization modes including demo, permutation, empty, and random grids.

    Args:
        task_data: JaxArcTask containing demonstration and test pair data
        selected_pair_idx: JAX array with the index of the selected pair
        episode_mode: Episode mode (0=train, 1=test) determining grid initialization
        config: Environment configuration containing grid initialization settings
        key: JAX PRNG key for diverse initialization.
        initial_pair_idx: Optional specific pair index for demo-based initialization.
                         If None, uses random selection. If specified, uses that pair.

    Returns:
        Tuple containing:
        - initial_grid: Starting grid for the episode (JAX array)
        - target_grid: Target grid (visible in train mode, masked in test mode)
        - initial_mask: Boolean mask indicating valid grid cells

    Examples:
        ```python
        # Training mode with diverse initialization and specific pair
        init_grid, target, mask = _initialize_grids(
            task, idx, 0, config, key, initial_pair_idx=2
        )

        # Test mode initialization (target masked) with random selection
        init_grid, masked_target, mask = _initialize_grids(task, idx, 1, config, key)
        ```

    Note:
        Uses the new diverse grid initialization engine when grid_initialization
        config is present and key is not provided. Respects initial_pair_idx for demo-based modes.
    """

    # Get target grid and mask based on episode mode
    def get_train_target():
        target_grid = task_data.output_grids_examples[selected_pair_idx]
        target_mask = task_data.output_masks_examples[selected_pair_idx]
        return target_grid, target_mask

    def get_test_target():
        # In test mode, target grid is masked (set to background color) to prevent cheating
        background_color = params.dataset.background_color
        initial_grid = task_data.test_input_grids[selected_pair_idx]
        target_grid = jnp.full_like(initial_grid, background_color)
        target_mask = jnp.zeros_like(task_data.test_input_masks[selected_pair_idx])
        return target_grid, target_mask

    target_grid, target_mask = jax.lax.cond(
        episode_mode == EPISODE_MODE_TRAIN, get_train_target, get_test_target
    )

    # The calling context (`arc_reset`) ensures that `config.grid_initialization` exists
    # and `key` is a valid PRNG key. The Python `if` statement that caused the TracerBoolConversionError
    # is removed. We now directly call the initialization logic.
    initial_grid, initial_mask = initialize_working_grids(
        task_data,
        params.grid_initialization,
        key,
        batch_size=1,
        initial_pair_idx=initial_pair_idx,
    )
    # Remove batch dimension (squeeze first axis)
    initial_grid = jnp.squeeze(initial_grid, axis=0)
    initial_mask = jnp.squeeze(initial_mask, axis=0)

    # Raw (dataset) input grid/mask for this pair based on episode_mode
    def get_train_input():
        return (
            task_data.input_grids_examples[selected_pair_idx],
            task_data.input_masks_examples[selected_pair_idx],
        )

    def get_test_input():
        return (
            task_data.test_input_grids[selected_pair_idx],
            task_data.test_input_masks[selected_pair_idx],
        )

    raw_input_grid, raw_input_mask = jax.lax.cond(
        episode_mode == EPISODE_MODE_TRAIN, get_train_input, get_test_input
    )

    return (
        initial_grid,
        target_grid,
        initial_mask,
        target_mask,
        raw_input_grid,
        raw_input_mask,
    )


def _create_initial_state(
    initial_grid: jnp.ndarray,
    target_grid: jnp.ndarray,
    initial_mask: jnp.ndarray,
    input_grid: jnp.ndarray,
    input_mask: jnp.ndarray,
    target_mask: jnp.ndarray,
    key: PRNGKey,
    task_idx: jnp.ndarray,
    pair_idx: jnp.ndarray,
) -> State:
    """Create initial state - focused helper function.

    This helper function constructs the complete initial State with all
    required fields properly initialized. It handles enhanced functionality
    including action history, operation masks, and completion tracking.

    Args:
        task_data: JaxArcTask containing complete task information
        initial_grid: Starting grid configuration (JAX array)
        target_grid: Target grid (visible or masked based on mode)
        initial_mask: Boolean mask for valid grid cells
        selected_pair_idx: Index of the currently selected pair
        episode_mode: Episode mode (0=train, 1=test)
        key: PRNG key for randomness during state initialization
        task_idx: Scalar task index into EnvParams.buffer identifying active task
        pair_idx: Scalar pair index within the selected task
        params: Environment parameters.

    Returns:
        State: Complete initial environment state with all fields properly
               initialized including action history, completion tracking,
               and operation masks.
    """
    # Calculate initial similarity (will be 0.0 in test mode due to masked target)
    initial_similarity = compute_grid_similarity(
        initial_grid, initial_mask, target_grid, target_mask
    )

    # Initialize allowed operations mask (all operations allowed by default)
    allowed_operations_mask = jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_)

    # Create initial state with simplified fields and task/pair tracking
    return State(
        working_grid=initial_grid,
        working_grid_mask=initial_mask,
        input_grid=input_grid,
        input_grid_mask=input_mask,
        target_grid=target_grid,
        target_grid_mask=target_mask,
        selected=jnp.zeros_like(initial_grid, dtype=jnp.bool_),
        clipboard=jnp.zeros_like(initial_grid, dtype=jnp.int32),
        step_count=jnp.array(0, dtype=jnp.int32),
        task_idx=task_idx,
        pair_idx=pair_idx,
        allowed_operations_mask=allowed_operations_mask,
        similarity_score=jnp.asarray(initial_similarity, dtype=jnp.float32),
        key=key,
    )


@eqx.filter_jit
def reset(params: EnvParams, key: PRNGKey) -> tuple[State, TimeStep]:
    """Pure JAX reset that samples from a JAX-native stacked task buffer.

    Behavior:
    - Samples a task index from params.buffer (optionally through params.subset_indices).
    - Computes a pair index based on episode_mode and task-specific pair counts.
    - Builds the initial TimeStep purely in JAX.
    """
    # Using EnvParams directly; no JaxArcConfig build

    # Split key for independent sampling
    key_id, key_pair, key_init = jax.random.split(key, 3)

    # Resolve candidate task indices
    has_subset = params.subset_indices is not None
    if has_subset:
        indices = params.subset_indices
        num = indices.shape[0]
        subset_index = jax.random.randint(key_id, (), 0, num)
        task_idx = indices[subset_index]
    else:
        # Derive total tasks N from leading dimension of a canonical field
        N = params.buffer.input_grids_examples.shape[0]
        task_idx = jax.random.randint(key_id, (), 0, N)

    # Gather a single task's arrays from the stacked buffer
    single = jax.tree_util.tree_map(lambda x: x[task_idx], params.buffer)

    # Compute pair count based on episode mode (0=train, 1=test)
    episode_mode = jnp.asarray(params.episode_mode, dtype=jnp.int32)
    train_pairs = jnp.asarray(single.num_train_pairs, dtype=jnp.int32)
    test_pairs = jnp.asarray(single.num_test_pairs, dtype=jnp.int32)
    max_pairs = jnp.where(
        episode_mode == jnp.asarray(0, dtype=jnp.int32), train_pairs, test_pairs
    )
    safe_max = jnp.maximum(max_pairs, jnp.asarray(1, dtype=jnp.int32))

    # Sample pair index safely; clamp to 0 when max_pairs == 0
    sampled_pair = jax.random.randint(key_pair, (), 0, safe_max)
    pair_idx = jnp.where(max_pairs > 0, sampled_pair, jnp.asarray(0, dtype=jnp.int32))
    selected_pair_idx = jnp.asarray(pair_idx, dtype=jnp.int32)

    # Build a lightweight JaxArcTask for downstream initialization
    # Note: counts are set to 0 as they are not used by _initialize_grids/_create_initial_state
    task_data = JaxArcTask(
        input_grids_examples=single.input_grids_examples,
        input_masks_examples=single.input_masks_examples,
        output_grids_examples=single.output_grids_examples,
        output_masks_examples=single.output_masks_examples,
        num_train_pairs=single.num_train_pairs,
        test_input_grids=single.test_input_grids,
        test_input_masks=single.test_input_masks,
        true_test_output_grids=single.true_test_output_grids,
        true_test_output_masks=single.true_test_output_masks,
        num_test_pairs=single.num_test_pairs,
        task_index=single.task_index,
    )

    # Initialize grids/masks using dataset and initialization settings
    initial_grid, target_grid, initial_mask, target_mask, input_grid, input_mask = (
        _initialize_grids(
            task_data=task_data,
            selected_pair_idx=selected_pair_idx,
            episode_mode=int(params.episode_mode),
            params=params,
            key=key_init,
            initial_pair_idx=None,
        )
    )

    # Build initial state with dynamic fields initialized (include task/pair tracking)
    # Build initial state with dynamic fields initialized
    state = _create_initial_state(
        initial_grid=initial_grid,
        target_grid=target_grid,
        initial_mask=initial_mask,
        input_grid=input_grid,
        input_mask=input_mask,
        target_mask=target_mask,
        key=key_init,
        task_idx=task_idx,
        pair_idx=pair_idx,
    )

    # Create initial observation
    observation = create_observation(state, params)

    timestep = TimeStep(
        step_type=StepType.FIRST,
        reward=jnp.asarray(0.0, dtype=jnp.float32),
        discount=jnp.asarray(1.0, dtype=jnp.float32),
        observation=observation,
        extras={},  # Initialize extras dict
    )
    return state, timestep


@eqx.filter_jit
def step(params: EnvParams, state: State, action) -> tuple[State, TimeStep]:
    """Functional step(params, state, action) -> (State, TimeStep).

    EnvParams is passed explicitly (not embedded in State) to support Meta-RL.
    """
    # Process action and update state using internal helpers (no legacy arc_step)
    processed_state, validated_action = _process_action(state, action, params)
    final_state = _update_state(state, processed_state, validated_action)

    # Enhanced termination logic with proper StepType semantics
    is_submit_step = validated_action.operation == jnp.asarray(34, dtype=jnp.int32)
    is_solved = final_state.similarity_score >= 1.0
    is_truncated = final_state.step_count >= jnp.asarray(params.max_episode_steps)

    # Compute reward (submit-aware)
    reward = _calculate_reward(
        state,
        final_state,
        params,
        is_submit_step=is_submit_step,
        episode_mode=int(params.episode_mode),
    )

    # TERMINATED: Task completed (solved or failed via submit)
    # TRUNCATED: Hit step/time limit
    # MID: Continue episode
    step_type = jnp.where(
        is_submit_step & is_solved,
        StepType.TERMINATED,  # Successfully completed
        jnp.where(
            is_submit_step & ~is_solved,
            StepType.TERMINATED,  # Failed submission
            jnp.where(
                is_truncated,
                StepType.TRUNCATED,  # Hit step limit
                StepType.MID,  # Continue episode
            ),
        ),
    )

    # Discount: 0.0 for terminal states, 1.0 for continuing
    discount = jnp.where(
        jnp.logical_or(
            step_type == StepType.TERMINATED, step_type == StepType.TRUNCATED
        ),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )

    # Build observation
    obs = create_observation(final_state, params)

    # Enhanced TimeStep with proper step_type and extras
    timestep = TimeStep(
        step_type=step_type,
        reward=reward,
        discount=discount,
        observation=obs,
        extras={},  # Can be populated with additional info
    )
    return final_state, timestep


def _process_action(
    state: State,
    action: Action,
    params: EnvParams,
) -> tuple[State, Action]:
    """Process action and return updated state.

    This function handles actions, which is the canonical action format
    for the core JaxARC environment. All actions are expected to be Action objects.

    Args:
        state: Current environment state before action execution
        action: Action to execute with operation and selection mask
        params: Environment configuration for action processing settings

    Returns:
        Tuple containing:
        - new_state: Updated environment state after action execution
        - validated_action: Validated action

    Examples:
        ```python
        # Process action
        mask = jnp.zeros((10, 10), dtype=jnp.bool_).at[5, 5].set(True)
        action = create_action(operation=15, selection=mask)
        new_state, validated_action = _process_action(state, action, params)
        ```

    Raises:
        ValueError: If action format is invalid or operation is out of range

    Note:
        Core environment only handles mask-based actions. Other action formats
        should be converted to masks by wrapper classes.
    """
    # Get grid shape for validation
    grid_shape = (state.working_grid.shape[0], state.working_grid.shape[1])

    # Validate the mask action
    validated_action = action.validate(grid_shape, max_operations=NUM_OPERATIONS)

    # Extract operation from validated action
    operation = validated_action.operation

    # Apply dynamic action space validation and filtering if enabled
    if (
        hasattr(params.action, "dynamic_action_filtering")
        and params.action.dynamic_action_filtering
    ):
        # Filter invalid operation according to policy
        operation = filter_invalid_operation(operation, state, params.action)

        # Update validated action with filtered operation
        validated_action = create_action(operation, validated_action.selection)

    # Process action using action handler
    selection_mask = action_handler(validated_action, state.working_grid_mask)

    new_state = update_selection(state, selection_mask)
    new_state = execute_grid_operation(new_state, operation)
    return new_state, validated_action


def _update_state(
    _old_state: State,
    new_state: State,
    _action: Action,
) -> State:
    """Update state with action history and step count - focused helper function.

    This helper function handles post-action state updates including action history
    tracking, step count incrementation, and other bookkeeping operations that
    need to occur after action execution.

    Args:
        old_state: Environment state before action execution
        new_state: Environment state after action execution but before updates
        action: Structured action that was executed
        config: Environment configuration for history and update settings

    Returns:
        ArcEnvState: Updated state with action history, incremented step count,
                    and other post-action updates applied.

    Examples:
        ```python
        # Update state after action processing
        updated_state = _update_state(old_state, new_state, action)
        ```

    Note:
        Conditionally applies action history tracking based on configuration.
        Always increments step count and handles other required state updates.
    """

    # Update step count using PyTree utilities (no action history integration)
    return increment_step_count(new_state)


@eqx.filter_jit
def validate_action_jax(action: Action, state: State, _unused: Any) -> jax.Array:
    """JAX-friendly validation returning a boolean predicate.

    This avoids raising inside JIT and can be used with lax.cond.
    """
    # Operation bounds check
    max_ops = jnp.asarray(NUM_OPERATIONS, dtype=jnp.int32)
    op = jnp.asarray(action.operation, dtype=jnp.int32)
    op_valid = (op >= 0) & (op < max_ops)

    # Grid bounds from static shapes
    grid_h: int = state.working_grid.shape[0]
    grid_w: int = state.working_grid.shape[1]

    # Mask validation - check that selection shape matches grid shape
    sel_shape = action.selection.shape
    # Compare static shapes; coerce to boolean scalar array
    shape_ok = (sel_shape[0] == grid_h) and (sel_shape[1] == grid_w)
    selection_valid = jnp.asarray(shape_ok, dtype=jnp.bool_)

    return jnp.asarray(op_valid & selection_valid, dtype=jnp.bool_)
