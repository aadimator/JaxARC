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
    new_state = state.replace(step_count=state.step_count + 1, episode_done=True)
    ```
"""

from __future__ import annotations

from typing import Any, Dict

import equinox as eqx
import jax.numpy as jnp

from .types import JaxArcTask
from .utils.jax_types import (
    ACTION_RECORD_FIELDS,
    DEFAULT_MAX_TEST_PAIRS,
    DEFAULT_MAX_TRAIN_PAIRS,
    MAX_HISTORY_LENGTH,
    NUM_OPERATIONS,
    ActionHistory,
    AvailableTestPairs,
    AvailableTrainPairs,
    EpisodeDone,
    EpisodeIndex,
    EpisodeMode,
    GridArray,
    HistoryLength,
    MaskArray,
    OperationMask,
    SelectionArray,
    SimilarityScore,
    StepCount,
    TestCompletionStatus,
    TrainCompletionStatus,
)
from jaxtyping import Array, Bool, Int


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
        target_grid_mask: Valid cells mask for the target grid
        step_count: Number of steps taken in current episode
        episode_done: Whether the current episode is complete
        current_example_idx: Which training example we're working on
        selected: Selection mask for operations
        clipboard: Grid data for copy/paste operations
        similarity_score: Grid similarity to target (0.0 to 1.0)

        # Enhanced functionality fields
        episode_mode: Episode mode (0=train, 1=test) for JAX compatibility
        available_demo_pairs: Mask of available demonstration pairs
        available_test_pairs: Mask of available test pairs
        demo_completion_status: Per-demo completion tracking
        test_completion_status: Per-test completion tracking
        action_history: Fixed-size action sequence storage
        action_history_length: Current history length
        allowed_operations_mask: Dynamic operation filtering

    Examples:
        ```python
        # Create new state with dataset-specific sizes
        max_train_pairs = 10  # e.g., ARC-AGI 2024
        max_test_pairs = 4    # e.g., ARC-AGI 2024

        state = ArcEnvState(
            task_data=task,
            working_grid=grid,
            working_grid_mask=mask,
            target_grid=target,
            target_grid_mask=target_mask,
            step_count=jnp.array(0),
            episode_done=jnp.array(False),
            current_example_idx=jnp.array(0),
            selected=jnp.zeros_like(grid, dtype=bool),
            clipboard=jnp.zeros_like(grid),
            similarity_score=jnp.array(0.0),
            # Enhanced functionality fields - sizes based on dataset
            episode_mode=jnp.array(0),  # 0=train
            available_demo_pairs=jnp.ones(max_train_pairs, dtype=bool),
            available_test_pairs=jnp.ones(max_test_pairs, dtype=bool),
            demo_completion_status=jnp.zeros(max_train_pairs, dtype=bool),
            test_completion_status=jnp.zeros(max_test_pairs, dtype=bool),
            action_history=jnp.zeros((MAX_HISTORY_LENGTH, ACTION_RECORD_FIELDS)),
            action_history_length=jnp.array(0),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=bool),
        )

        # Update state using Equinox tree_at
        new_state = eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)

        # Update multiple fields
        new_state = eqx.tree_at(
            lambda s: (s.step_count, s.episode_done),
            state,
            (state.step_count + 1, jnp.array(True)),
        )
        ```
    """

    # Core ARC state
    task_data: JaxArcTask
    working_grid: GridArray  # Current grid being modified
    working_grid_mask: MaskArray  # Valid cells mask
    target_grid: GridArray  # Goal grid for current example
    target_grid_mask: MaskArray  # Valid cells mask for target grid

    # Episode management
    step_count: StepCount
    episode_done: EpisodeDone
    current_example_idx: EpisodeIndex  # Which training example we're working on

    # Grid operations fields
    selected: SelectionArray  # Selection mask for operations
    clipboard: GridArray  # For copy/paste operations
    similarity_score: SimilarityScore  # Grid similarity to target

    # Enhanced functionality fields
    episode_mode: EpisodeMode  # 0=train, 1=test for JAX compatibility
    available_demo_pairs: AvailableTrainPairs  # Mask of available training pairs
    available_test_pairs: AvailableTestPairs  # Mask of available test pairs
    demo_completion_status: TrainCompletionStatus  # Per-demo completion tracking
    test_completion_status: TestCompletionStatus  # Per-test completion tracking
    action_history: ActionHistory  # Fixed-size action sequence
    action_history_length: HistoryLength  # Current history length
    allowed_operations_mask: OperationMask  # Dynamic operation filtering

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
            chex.assert_rank(self.target_grid_mask, 2)
            chex.assert_rank(self.selected, 2)
            chex.assert_rank(self.clipboard, 2)

            chex.assert_type(self.working_grid, jnp.integer)
            chex.assert_type(self.working_grid_mask, jnp.bool_)
            chex.assert_type(self.target_grid, jnp.integer)
            chex.assert_type(self.target_grid_mask, jnp.bool_)
            chex.assert_type(self.selected, jnp.bool_)
            chex.assert_type(self.clipboard, jnp.integer)
            chex.assert_type(self.similarity_score, jnp.floating)

            # Check consistent shapes
            chex.assert_shape(self.working_grid_mask, self.working_grid.shape)
            chex.assert_shape(self.target_grid, self.working_grid.shape)
            chex.assert_shape(self.target_grid_mask, self.working_grid.shape)
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

            # Validate enhanced functionality fields
            chex.assert_type(self.episode_mode, jnp.integer)
            chex.assert_type(self.available_demo_pairs, jnp.bool_)
            chex.assert_type(self.available_test_pairs, jnp.bool_)
            chex.assert_type(self.demo_completion_status, jnp.bool_)
            chex.assert_type(self.test_completion_status, jnp.bool_)
            chex.assert_type(self.action_history, jnp.floating)
            chex.assert_type(self.action_history_length, jnp.integer)
            chex.assert_type(self.allowed_operations_mask, jnp.bool_)

            # Validate enhanced field shapes
            chex.assert_shape(self.episode_mode, ())
            chex.assert_shape(self.action_history_length, ())
            chex.assert_rank(self.available_demo_pairs, 1)
            chex.assert_rank(self.available_test_pairs, 1)
            chex.assert_rank(self.demo_completion_status, 1)
            chex.assert_rank(self.test_completion_status, 1)
            chex.assert_rank(self.action_history, 2)
            chex.assert_rank(self.allowed_operations_mask, 1)

            # Validate consistent sizes between demo pairs and completion status
            if self.available_demo_pairs.shape != self.demo_completion_status.shape:
                msg = f"Demo pairs and completion status shape mismatch: {self.available_demo_pairs.shape} vs {self.demo_completion_status.shape}"
                raise ValueError(msg)

            # Validate consistent sizes between test pairs and completion status
            if self.available_test_pairs.shape != self.test_completion_status.shape:
                msg = f"Test pairs and completion status shape mismatch: {self.available_test_pairs.shape} vs {self.test_completion_status.shape}"
                raise ValueError(msg)

            # Validate action history has reasonable number of fields
            # With dynamic sizing, this can vary from 6 (point) to 904 (full mask)
            action_history_fields = self.action_history.shape[1]
            if action_history_fields < 6:  # Minimum: 2 (point) + 4 (metadata)
                msg = f"Action history should have at least 6 fields, got {action_history_fields}"
                raise ValueError(msg)
            if action_history_fields > 1000:  # Reasonable upper bound
                msg = (
                    f"Action history has unusually many fields: {action_history_fields}"
                )
                raise ValueError(msg)

            # Validate operations mask has correct size
            if self.allowed_operations_mask.shape[0] != NUM_OPERATIONS:
                msg = f"Operations mask should have {NUM_OPERATIONS} operations, got {self.allowed_operations_mask.shape[0]}"
                raise ValueError(msg)

            # Validate episode mode is valid (0 or 1)
            if hasattr(self.episode_mode, "item"):
                mode_val = int(self.episode_mode.item())
                if mode_val not in (0, 1):
                    msg = f"Episode mode must be 0 (train) or 1 (test), got {mode_val}"
                    raise ValueError(msg)

        except (AttributeError, TypeError):
            # Skip validation during JAX transformations
            pass

    def replace(self, **kwargs) -> ArcEnvState:
        """Create a new state with updated fields.

        This method provides a convenient way to update multiple fields at once,
        similar to the dataclass replace method but using Equinox patterns.

        Args:
            **kwargs: Fields to update with their new values

        Returns:
            New ArcEnvState with updated fields

        Examples:
            ```python
            new_state = state.replace(step_count=state.step_count + 1, episode_done=True)
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

    def get_actual_grid_shape(self) -> tuple[int, int]:
        """Get the actual shape of the working grid based on the mask.

        JAX-compatible method that delegates to utility function.

        Since JAX requires static shapes, working_grid is always padded to max dimensions,
        but the actual meaningful grid size is determined by working_grid_mask.

        Returns:
            Tuple of (height, width) representing the actual grid dimensions

        Examples:
            ```python
            # For a 5x5 actual grid padded to 30x30
            state = ArcEnvState(...)
            actual_height, actual_width = state.get_actual_grid_shape()
            # Returns (5, 5) instead of (30, 30)

            # Use for extracting the meaningful part of the grid
            actual_grid = state.get_actual_working_grid()
            ```
        """
        from .utils.grid_utils import get_actual_grid_shape_from_mask

        return get_actual_grid_shape_from_mask(self.working_grid_mask)

    def get_actual_working_grid(self) -> GridArray:
        """Get the actual working grid without padding.

        JAX-compatible method that delegates to utility function.

        Returns the working grid cropped to its actual dimensions based on the mask.
        This is useful for visualization and analysis where you only want the
        meaningful part of the grid.

        Returns:
            GridArray containing only the actual grid content (no padding)

        Examples:
            ```python
            state = ArcEnvState(...)
            actual_grid = state.get_actual_working_grid()
            # Returns a 5x5 grid instead of 30x30 padded grid
            ```
        """
        from .utils.grid_utils import crop_grid_to_mask

        return crop_grid_to_mask(self.working_grid, self.working_grid_mask)

    def get_actual_target_grid(self) -> GridArray:
        """Get the actual target grid without padding.

        JAX-compatible method that delegates to utility function.

        Returns the target grid cropped to its actual dimensions based on the mask.
        Useful for comparing against the working grid or for visualization.

        Returns:
            GridArray containing only the actual target grid content (no padding)
        """
        from .utils.grid_utils import crop_grid_to_mask

        return crop_grid_to_mask(self.target_grid, self.working_grid_mask)

    # =========================================================================
    # Agent Utility Methods for Enhanced Functionality
    # =========================================================================

    def is_training_mode(self) -> bool:
        """Check if environment is in training mode.

        Returns:
            True if in training mode (episode_mode == 0), False if in test mode
        """
        return self.episode_mode == 0

    def is_test_mode(self) -> bool:
        """Check if environment is in test/evaluation mode.

        Returns:
            True if in test mode (episode_mode == 1), False if in training mode
        """
        return self.episode_mode == 1

    def get_available_demo_count(self) -> Int[Array, ""]:
        """Get the number of available demonstration pairs.

        Returns:
            JAX scalar array containing the number of available demonstration pairs
        """
        return jnp.sum(self.available_demo_pairs)

    def get_available_test_count(self) -> Int[Array, ""]:
        """Get the number of available test pairs.

        Returns:
            JAX scalar array containing the number of available test pairs
        """
        return jnp.sum(self.available_test_pairs)

    def get_completed_demo_count(self) -> Int[Array, ""]:
        """Get the number of completed demonstration pairs.

        Returns:
            JAX scalar array containing the number of completed demonstration pairs
        """
        return jnp.sum(self.demo_completion_status)

    def get_completed_test_count(self) -> Int[Array, ""]:
        """Get the number of completed test pairs.

        Returns:
            JAX scalar array containing the number of completed test pairs
        """
        return jnp.sum(self.test_completion_status)

    def is_current_pair_completed(self) -> Bool[Array, ""]:
        """Check if the current demonstration/test pair is completed.

        Returns:
            JAX boolean scalar array indicating if current pair is marked as completed
        """
        return jnp.where(
            self.is_training_mode(),
            self.demo_completion_status[self.current_example_idx],
            self.test_completion_status[self.current_example_idx],
        )

    def get_allowed_operations_count(self) -> Int[Array, ""]:
        """Get the number of currently allowed operations.

        Returns:
            JAX scalar array containing the number of operations that are currently allowed
        """
        return jnp.sum(self.allowed_operations_mask)

    def is_operation_allowed(self, operation_id: int) -> Bool[Array, ""]:
        """Check if a specific operation is currently allowed.

        Args:
            operation_id: Operation ID to check (0-41)

        Returns:
            JAX boolean scalar array indicating if the operation is allowed
        """
        return jnp.where(
            (operation_id >= 0) & (operation_id < len(self.allowed_operations_mask)),
            self.allowed_operations_mask[operation_id],
            False,
        )

    def get_action_history_length(self) -> HistoryLength:
        """Get the current length of action history.

        Returns:
            JAX scalar array containing the number of actions stored in history
        """
        return self.action_history_length

    def has_action_history(self) -> Bool[Array, ""]:
        """Check if there is any action history.

        Returns:
            JAX boolean scalar array indicating if action history contains at least one action
        """
        return self.action_history_length > 0

    def get_current_pair_info(self) -> Dict[str, Any]:
        """Get information about the current demonstration/test pair.

        Note: This method converts JAX arrays to Python types for readability.

        Returns:
            Dictionary containing current pair information
        """
        return {
            "pair_index": int(self.current_example_idx),
            "is_training": bool(self.is_training_mode()),
            "is_completed": bool(self.is_current_pair_completed()),
            "total_available": int(
                jnp.where(
                    self.is_training_mode(),
                    self.get_available_demo_count(),
                    self.get_available_test_count(),
                )
            ),
            "total_completed": int(
                jnp.where(
                    self.is_training_mode(),
                    self.get_completed_demo_count(),
                    self.get_completed_test_count(),
                )
            ),
        }

    def get_episode_progress(self) -> Dict[str, Any]:
        """Get overall episode progress information.

        Note: This method converts JAX arrays to Python types for readability.

        Returns:
            Dictionary containing episode progress metrics
        """
        return {
            "step_count": int(self.step_count),
            "episode_done": bool(self.episode_done),
            "similarity_score": float(self.similarity_score),
            "current_pair": self.get_current_pair_info(),
            "action_history_length": int(self.get_action_history_length()),
            "allowed_operations": int(self.get_allowed_operations_count()),
        }

    def get_observation_summary(self) -> Dict[str, Any]:
        """Get a summary of the observation for agents.

        This method provides a structured summary of the most important
        information from the state that agents typically need.

        Note: This method converts JAX arrays to Python types for readability.
        For JAX-compatible operations, use the individual utility methods directly.

        Returns:
            Dictionary containing key observation information
        """
        grid_shape = self.get_actual_grid_shape()

        return {
            # Core grid information
            "grid_shape": grid_shape,
            "similarity_score": float(self.similarity_score),
            # Episode context
            "episode_mode": "train" if bool(self.is_training_mode()) else "test",
            "step_count": int(self.step_count),
            "episode_done": bool(self.episode_done),
            # Pair information
            "current_pair_index": int(self.current_example_idx),
            "available_pairs": int(
                jnp.where(
                    self.is_training_mode(),
                    self.get_available_demo_count(),
                    self.get_available_test_count(),
                )
            ),
            "completed_pairs": int(
                jnp.where(
                    self.is_training_mode(),
                    self.get_completed_demo_count(),
                    self.get_completed_test_count(),
                )
            ),
            # Action space information
            "allowed_operations": int(self.get_allowed_operations_count()),
            "action_history_length": int(self.get_action_history_length()),
            # Grid statistics
            "has_selection": bool(jnp.any(self.selected)),
            "has_clipboard_data": bool(jnp.any(self.clipboard != 0)),
        }

    # =========================================================================
    # Enhanced Utility Methods for Demonstration and Test Pair Access
    # =========================================================================

    def get_current_demo_pair_data(self) -> tuple[GridArray, GridArray, MaskArray]:
        """Get current demonstration pair data.

        Returns:
            Tuple of (input_grid, output_grid, input_mask) for current demo pair

        Raises:
            ValueError: If not in training mode or current pair is invalid
        """
        if not self.is_training_mode():
            raise ValueError("Cannot get demo pair data in test mode")

        if not self.task_data.is_demo_pair_available(int(self.current_example_idx)):
            raise ValueError(f"Demo pair {self.current_example_idx} is not available")

        return self.task_data.get_demo_pair_data(int(self.current_example_idx))

    def get_current_test_pair_data(self) -> tuple[GridArray, MaskArray]:
        """Get current test pair data.

        Returns:
            Tuple of (input_grid, input_mask) for current test pair

        Raises:
            ValueError: If not in test mode or current pair is invalid
        """
        if not self.is_test_mode():
            raise ValueError("Cannot get test pair data in training mode")

        if not self.task_data.is_test_pair_available(int(self.current_example_idx)):
            raise ValueError(f"Test pair {self.current_example_idx} is not available")

        return self.task_data.get_test_pair_data(int(self.current_example_idx))

    def get_available_demo_indices(self) -> Int[Array, "max_train_pairs"]:
        """Get indices of available demonstration pairs.

        Returns:
            JAX array of indices for available demonstration pairs (padded with -1)
        """
        available_mask = self.available_demo_pairs
        indices = jnp.arange(len(available_mask))
        return jnp.where(available_mask, indices, -1)

    def get_available_test_indices(self) -> Int[Array, "max_test_pairs"]:
        """Get indices of available test pairs.

        Returns:
            JAX array of indices for available test pairs (padded with -1)
        """
        available_mask = self.available_test_pairs
        indices = jnp.arange(len(available_mask))
        return jnp.where(available_mask, indices, -1)

    def get_completed_demo_indices(self) -> Int[Array, "max_train_pairs"]:
        """Get indices of completed demonstration pairs.

        Returns:
            JAX array of indices for completed demonstration pairs (padded with -1)
        """
        completed_mask = self.demo_completion_status
        indices = jnp.arange(len(completed_mask))
        return jnp.where(completed_mask, indices, -1)

    def get_completed_test_indices(self) -> Int[Array, "max_test_pairs"]:
        """Get indices of completed test pairs.

        Returns:
            JAX array of indices for completed test pairs (padded with -1)
        """
        completed_mask = self.test_completion_status
        indices = jnp.arange(len(completed_mask))
        return jnp.where(completed_mask, indices, -1)

    def get_uncompleted_demo_indices(self) -> Int[Array, "max_train_pairs"]:
        """Get indices of uncompleted demonstration pairs.

        Returns:
            JAX array of indices for uncompleted demonstration pairs (padded with -1)
        """
        uncompleted_mask = self.available_demo_pairs & ~self.demo_completion_status
        indices = jnp.arange(len(uncompleted_mask))
        return jnp.where(uncompleted_mask, indices, -1)

    def get_uncompleted_test_indices(self) -> Int[Array, "max_test_pairs"]:
        """Get indices of uncompleted test pairs.

        Returns:
            JAX array of indices for uncompleted test pairs (padded with -1)
        """
        uncompleted_mask = self.available_test_pairs & ~self.test_completion_status
        indices = jnp.arange(len(uncompleted_mask))
        return jnp.where(uncompleted_mask, indices, -1)

    def is_demo_pair_available(self, pair_idx: int) -> Bool[Array, ""]:
        """Check if a specific demonstration pair is available.

        Args:
            pair_idx: Index of the demonstration pair to check

        Returns:
            JAX boolean scalar array indicating if the pair is available
        """
        return jnp.where(
            (pair_idx >= 0) & (pair_idx < len(self.available_demo_pairs)),
            self.available_demo_pairs[pair_idx],
            False,
        )

    def is_test_pair_available(self, pair_idx: int) -> Bool[Array, ""]:
        """Check if a specific test pair is available.

        Args:
            pair_idx: Index of the test pair to check

        Returns:
            JAX boolean scalar array indicating if the pair is available
        """
        return jnp.where(
            (pair_idx >= 0) & (pair_idx < len(self.available_test_pairs)),
            self.available_test_pairs[pair_idx],
            False,
        )

    def is_demo_pair_completed(self, pair_idx: int) -> Bool[Array, ""]:
        """Check if a specific demonstration pair is completed.

        Args:
            pair_idx: Index of the demonstration pair to check

        Returns:
            JAX boolean scalar array indicating if the pair is completed
        """
        return jnp.where(
            (pair_idx >= 0) & (pair_idx < len(self.demo_completion_status)),
            self.demo_completion_status[pair_idx],
            False,
        )

    def is_test_pair_completed(self, pair_idx: int) -> Bool[Array, ""]:
        """Check if a specific test pair is completed.

        Args:
            pair_idx: Index of the test pair to check

        Returns:
            JAX boolean scalar array indicating if the pair is completed
        """
        return jnp.where(
            (pair_idx >= 0) & (pair_idx < len(self.test_completion_status)),
            self.test_completion_status[pair_idx],
            False,
        )

    def get_next_available_demo_index(self) -> Int[Array, ""]:
        """Get the next available demonstration pair index.

        Returns:
            JAX scalar array with index of next available demo pair, or -1 if none available
        """
        current_idx = self.current_example_idx

        # Simple linear search for next available pair
        def find_next(i):
            return jnp.where(
                (i < len(self.available_demo_pairs)) & self.available_demo_pairs[i],
                i,
                -1,
            )

        # Check indices after current
        next_idx = -1
        for i in range(len(self.available_demo_pairs)):
            candidate = current_idx + 1 + i
            wrapped_candidate = candidate % len(self.available_demo_pairs)
            is_valid = (
                candidate < len(self.available_demo_pairs)
            ) & self.available_demo_pairs[candidate]
            is_wrapped_valid = self.available_demo_pairs[wrapped_candidate]

            next_idx = jnp.where(
                (next_idx == -1) & is_valid,
                candidate,
                jnp.where(
                    (next_idx == -1)
                    & (candidate >= len(self.available_demo_pairs))
                    & is_wrapped_valid,
                    wrapped_candidate,
                    next_idx,
                ),
            )

        return next_idx

    def get_prev_available_demo_index(self) -> Int[Array, ""]:
        """Get the previous available demonstration pair index.

        Returns:
            JAX scalar array with index of previous available demo pair, or -1 if none available
        """
        current_idx = self.current_example_idx

        # Simple linear search for previous available pair
        prev_idx = -1
        for i in range(len(self.available_demo_pairs)):
            candidate = current_idx - 1 - i
            wrapped_candidate = candidate % len(self.available_demo_pairs)
            is_valid = (candidate >= 0) & self.available_demo_pairs[candidate]
            is_wrapped_valid = self.available_demo_pairs[wrapped_candidate]

            prev_idx = jnp.where(
                (prev_idx == -1) & is_valid,
                candidate,
                jnp.where(
                    (prev_idx == -1) & (candidate < 0) & is_wrapped_valid,
                    wrapped_candidate,
                    prev_idx,
                ),
            )

        return prev_idx

    def get_next_available_test_index(self) -> Int[Array, ""]:
        """Get the next available test pair index.

        Returns:
            JAX scalar array with index of next available test pair, or -1 if none available
        """
        current_idx = self.current_example_idx

        # Simple linear search for next available pair
        next_idx = -1
        for i in range(len(self.available_test_pairs)):
            candidate = current_idx + 1 + i
            wrapped_candidate = candidate % len(self.available_test_pairs)
            is_valid = (
                candidate < len(self.available_test_pairs)
            ) & self.available_test_pairs[candidate]
            is_wrapped_valid = self.available_test_pairs[wrapped_candidate]

            next_idx = jnp.where(
                (next_idx == -1) & is_valid,
                candidate,
                jnp.where(
                    (next_idx == -1)
                    & (candidate >= len(self.available_test_pairs))
                    & is_wrapped_valid,
                    wrapped_candidate,
                    next_idx,
                ),
            )

        return next_idx

    def get_prev_available_test_index(self) -> Int[Array, ""]:
        """Get the previous available test pair index.

        Returns:
            JAX scalar array with index of previous available test pair, or -1 if none available
        """
        current_idx = self.current_example_idx

        # Simple linear search for previous available pair
        prev_idx = -1
        for i in range(len(self.available_test_pairs)):
            candidate = current_idx - 1 - i
            wrapped_candidate = candidate % len(self.available_test_pairs)
            is_valid = (candidate >= 0) & self.available_test_pairs[candidate]
            is_wrapped_valid = self.available_test_pairs[wrapped_candidate]

            prev_idx = jnp.where(
                (prev_idx == -1) & is_valid,
                candidate,
                jnp.where(
                    (prev_idx == -1) & (candidate < 0) & is_wrapped_valid,
                    wrapped_candidate,
                    prev_idx,
                ),
            )

        return prev_idx

    def get_first_unsolved_demo_index(self) -> Int[Array, ""]:
        """Get the first unsolved demonstration pair index.

        Returns:
            JAX scalar array with index of first unsolved demo pair, or -1 if all solved or none available
        """
        uncompleted_mask = self.available_demo_pairs & ~self.demo_completion_status

        # Find first True value in mask
        first_idx = -1
        for i in range(len(uncompleted_mask)):
            first_idx = jnp.where((first_idx == -1) & uncompleted_mask[i], i, first_idx)

        return first_idx

    def get_first_unsolved_test_index(self) -> Int[Array, ""]:
        """Get the first unsolved test pair index.

        Returns:
            JAX scalar array with index of first unsolved test pair, or -1 if all solved or none available
        """
        uncompleted_mask = self.available_test_pairs & ~self.test_completion_status

        # Find first True value in mask
        first_idx = -1
        for i in range(len(uncompleted_mask)):
            first_idx = jnp.where((first_idx == -1) & uncompleted_mask[i], i, first_idx)

        return first_idx

    def get_pair_availability_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of pair availability and completion status.

        Returns:
            Dictionary containing detailed pair status information
        """
        return {
            # Demo pair information
            "demo_pairs": {
                "available_count": self.get_available_demo_count(),
                "completed_count": self.get_completed_demo_count(),
                "available_indices": self.get_available_demo_indices().tolist(),
                "completed_indices": self.get_completed_demo_indices().tolist(),
                "uncompleted_indices": self.get_uncompleted_demo_indices().tolist(),
            },
            # Test pair information
            "test_pairs": {
                "available_count": self.get_available_test_count(),
                "completed_count": self.get_completed_test_count(),
                "available_indices": self.get_available_test_indices().tolist(),
                "completed_indices": self.get_completed_test_indices().tolist(),
                "uncompleted_indices": self.get_uncompleted_test_indices().tolist(),
            },
            # Current state
            "current_pair_index": int(self.current_example_idx),
            "current_mode": "train" if self.is_training_mode() else "test",
            "current_pair_completed": self.is_current_pair_completed(),
        }

    @property
    def __dataclass_fields__(self) -> dict:
        """Property to mimic dataclass fields for replace method."""
        return {
            "task_data": None,
            "working_grid": None,
            "working_grid_mask": None,
            "target_grid": None,
            "step_count": None,
            "episode_done": None,
            "current_example_idx": None,
            "selected": None,
            "clipboard": None,
            "similarity_score": None,
            # Enhanced functionality fields
            "episode_mode": None,
            "available_demo_pairs": None,
            "available_test_pairs": None,
            "demo_completion_status": None,
            "test_completion_status": None,
            "action_history": None,
            "action_history_length": None,
            "allowed_operations_mask": None,
        }


def create_arc_env_state(
    task_data: JaxArcTask,
    working_grid: GridArray,
    working_grid_mask: MaskArray,
    target_grid: GridArray,
    target_grid_mask: MaskArray,
    max_train_pairs: int = DEFAULT_MAX_TRAIN_PAIRS,
    max_test_pairs: int = DEFAULT_MAX_TEST_PAIRS,
    step_count: int = 0,
    episode_done: bool = False,
    current_example_idx: int = 0,
    episode_mode: int = 0,  # 0=train, 1=test
    action_history_length: int = 0,
) -> ArcEnvState:
    """Factory function to create ArcEnvState with dataset-appropriate sizes.

    This function simplifies creating ArcEnvState instances by automatically
    creating the enhanced functionality arrays with the correct sizes based
    on the dataset configuration.

    Args:
        task_data: The ARC task data
        working_grid: Current grid being modified
        working_grid_mask: Valid cells mask for the working grid
        target_grid: Goal grid for current example
        target_grid_mask: Valid cells mask for the target grid
        max_train_pairs: Maximum number of training pairs (dataset-dependent)
        max_test_pairs: Maximum number of test pairs (dataset-dependent)
        step_count: Initial step count
        episode_done: Initial episode done status
        current_example_idx: Initial example index
        episode_mode: Episode mode (0=train, 1=test)
        action_history_length: Initial action history length

    Returns:
        ArcEnvState configured for the specified dataset sizes

    Examples:
        ```python
        # For ARC-AGI 2024 (10 train, 4 test pairs)
        state = create_arc_env_state(
            task_data=task,
            working_grid=grid,
            working_grid_mask=mask,
            target_grid=target,
            target_grid_mask=target_mask,
            max_train_pairs=10,
            max_test_pairs=4
        )

        # For MiniARC (5 train, 1 test pair)
        state = create_arc_env_state(
            task_data=task,
            working_grid=grid,
            working_grid_mask=mask,
            target_grid=target,
            target_grid_mask=target_mask,
            max_train_pairs=5,
            max_test_pairs=1
        )

        # For augmented training (20 train pairs, 4 test pairs)
        state = create_arc_env_state(
            task_data=task,
            working_grid=grid,
            working_grid_mask=mask,
            target_grid=target,
            target_grid_mask=target_mask,
            max_train_pairs=20,  # Increased for augmentation
            max_test_pairs=4
        )
        ```
    """
    return ArcEnvState(
        task_data=task_data,
        working_grid=working_grid,
        working_grid_mask=working_grid_mask,
        target_grid=target_grid,
        target_grid_mask=target_grid_mask,
        step_count=jnp.array(step_count, dtype=jnp.int32),
        episode_done=jnp.array(episode_done),
        current_example_idx=jnp.array(current_example_idx, dtype=jnp.int32),
        selected=jnp.zeros_like(working_grid, dtype=bool),
        clipboard=jnp.zeros_like(working_grid, dtype=jnp.int32),
        similarity_score=jnp.array(0.0),
        # Enhanced functionality fields with dataset-appropriate sizes
        episode_mode=jnp.array(episode_mode, dtype=jnp.int32),
        available_demo_pairs=jnp.ones(max_train_pairs, dtype=bool),
        available_test_pairs=jnp.ones(max_test_pairs, dtype=bool),
        demo_completion_status=jnp.zeros(max_train_pairs, dtype=bool),
        test_completion_status=jnp.zeros(max_test_pairs, dtype=bool),
        action_history=jnp.zeros((MAX_HISTORY_LENGTH, ACTION_RECORD_FIELDS)),
        action_history_length=jnp.array(action_history_length, dtype=jnp.int32),
        allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=bool),
    )
