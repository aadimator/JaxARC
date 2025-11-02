"""
Tests for action system in jaxarc.envs.actions.

This module tests mask-based action creation, validation,
action processing pipeline, and transformations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxarc.configs.action_config import ActionConfig
from jaxarc.envs.actions import (
    Action,
    action_handler,
    create_action,
    filter_invalid_operation,
    get_allowed_operations,
    validate_operation,
)
from jaxarc.state import State
from jaxarc.types import NUM_OPERATIONS


class TestActionCreation:
    """Test Action class and creation utilities."""

    def test_action_creation(self):
        """Test basic Action creation."""
        operation = jnp.array(5, dtype=jnp.int32)
        selection = jnp.ones((3, 3), dtype=jnp.bool_)

        action = Action(operation=operation, selection=selection)

        assert isinstance(action, Action)
        assert action.operation == 5
        assert action.selection.shape == (3, 3)
        assert action.selection.dtype == jnp.bool_

    def test_create_action_utility(self):
        """Test create_action utility function."""
        operation = jnp.array(10, dtype=jnp.int32)
        selection = jnp.zeros((5, 5), dtype=jnp.bool_)
        selection = selection.at[2, 2].set(True)

        action = create_action(operation=operation, selection=selection)

        assert isinstance(action, Action)
        assert action.operation == 10
        assert action.selection.shape == (5, 5)
        assert jnp.sum(action.selection) == 1  # Only one cell selected

    def test_action_validation(self):
        """Test Action validation method."""
        operation = jnp.array(15, dtype=jnp.int32)
        selection = jnp.ones((4, 4), dtype=jnp.bool_)

        action = Action(operation=operation, selection=selection)

        # Test validation with correct grid shape
        validated_action = action.validate(grid_shape=(4, 4), max_operations=35)

        assert isinstance(validated_action, Action)
        assert validated_action.operation == 15
        assert validated_action.selection.shape == (4, 4)

    def test_action_validation_clipping(self):
        """Test Action validation with operation clipping."""
        # Test operation out of range
        operation = jnp.array(100, dtype=jnp.int32)  # Too large
        selection = jnp.ones((3, 3), dtype=jnp.bool_)

        action = Action(operation=operation, selection=selection)
        validated_action = action.validate(grid_shape=(3, 3), max_operations=35)

        # Operation should be clipped to valid range
        assert 0 <= validated_action.operation < 35


class TestActionValidation:
    """Test action validation functions."""

    def test_validate_operation(self):
        """Test validate_operation function."""
        # Create a simple state for testing
        working_grid = jnp.ones((3, 3), dtype=jnp.int32)
        working_grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        target_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        target_grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        selected = jnp.zeros((3, 3), dtype=jnp.bool_)
        clipboard = jnp.zeros((3, 3), dtype=jnp.int32)

        state = State(
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            input_grid=working_grid,
            input_grid_mask=working_grid_mask,
            target_grid=target_grid,
            target_grid_mask=target_grid_mask,
            selected=selected,
            clipboard=clipboard,
            step_count=jnp.array(0, dtype=jnp.int32),
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42),
        )

        action_config = ActionConfig()

        # Test valid operation
        valid_op = jnp.array(10, dtype=jnp.int32)
        result = validate_operation(valid_op, state, action_config)
        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.bool_

        # Test invalid operation (too large)
        invalid_op = jnp.array(100, dtype=jnp.int32)
        result = validate_operation(invalid_op, state, action_config)
        assert isinstance(result, jax.Array)
        assert result.dtype == jnp.bool_

    def test_get_allowed_operations(self):
        """Test get_allowed_operations function."""
        # Create a simple state for testing
        working_grid = jnp.ones((3, 3), dtype=jnp.int32)
        working_grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        target_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        target_grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        selected = jnp.zeros((3, 3), dtype=jnp.bool_)
        clipboard = jnp.zeros((3, 3), dtype=jnp.int32)

        state = State(
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            input_grid=working_grid,
            input_grid_mask=working_grid_mask,
            target_grid=target_grid,
            target_grid_mask=target_grid_mask,
            selected=selected,
            clipboard=clipboard,
            step_count=jnp.array(0, dtype=jnp.int32),
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42),
        )

        action_config = ActionConfig()
        allowed_ops = get_allowed_operations(state, action_config)

        assert isinstance(allowed_ops, jax.Array)
        assert allowed_ops.dtype == jnp.bool_
        assert allowed_ops.shape == (NUM_OPERATIONS,)

    def test_filter_invalid_operation(self):
        """Test filter_invalid_operation function."""
        # Create a simple state for testing
        working_grid = jnp.ones((3, 3), dtype=jnp.int32)
        working_grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        target_grid = jnp.zeros((3, 3), dtype=jnp.int32)
        target_grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        selected = jnp.zeros((3, 3), dtype=jnp.bool_)
        clipboard = jnp.zeros((3, 3), dtype=jnp.int32)

        state = State(
            working_grid=working_grid,
            working_grid_mask=working_grid_mask,
            input_grid=working_grid,
            input_grid_mask=working_grid_mask,
            target_grid=target_grid,
            target_grid_mask=target_grid_mask,
            selected=selected,
            clipboard=clipboard,
            step_count=jnp.array(0, dtype=jnp.int32),
            task_idx=jnp.array(0, dtype=jnp.int32),
            pair_idx=jnp.array(0, dtype=jnp.int32),
            allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
            similarity_score=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42),
        )

        action_config = ActionConfig()

        # Test valid operation
        valid_op = jnp.array(5, dtype=jnp.int32)
        filtered_op = filter_invalid_operation(valid_op, state, action_config)
        assert isinstance(filtered_op, jax.Array)
        assert filtered_op.dtype == jnp.int32

    def test_action_handler(self):
        """Test action_handler function."""
        # Create test action
        operation = jnp.array(0, dtype=jnp.int32)  # Fill operation
        selection = jnp.zeros((3, 3), dtype=jnp.bool_)
        selection = selection.at[1, 1].set(True)  # Select center cell

        action = Action(operation=operation, selection=selection)

        # Create mask for handler
        grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)

        # Test action handler
        result_mask = action_handler(action, grid_mask)

        assert isinstance(result_mask, jax.Array)
        assert result_mask.dtype == jnp.bool_
        assert result_mask.shape == (3, 3)


class TestActionProcessing:
    """Test action processing pipeline."""

    def test_action_pipeline_basic(self):
        """Test basic action processing pipeline."""
        # Create action
        operation = jnp.array(1, dtype=jnp.int32)  # Fill with color 1
        selection = jnp.ones((3, 3), dtype=jnp.bool_)

        action = create_action(operation=operation, selection=selection)

        # Validate action
        validated_action = action.validate(grid_shape=(3, 3), max_operations=35)

        # Process with action handler
        grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        result_mask = action_handler(validated_action, grid_mask)

        # Verify pipeline results
        assert isinstance(validated_action, Action)
        assert isinstance(result_mask, jax.Array)
        assert result_mask.shape == (3, 3)

    def test_action_jax_compatibility(self):
        """Test that action functions are JAX-compatible."""
        operation = jnp.array(2, dtype=jnp.int32)
        selection = jnp.ones((4, 4), dtype=jnp.bool_)

        action = create_action(operation=operation, selection=selection)
        grid_mask = jnp.ones((4, 4), dtype=jnp.bool_)

        # Test JIT compilation
        jitted_handler = jax.jit(action_handler)
        result = jitted_handler(action, grid_mask)

        assert isinstance(result, jax.Array)
        assert result.shape == (4, 4)

    def test_action_transformations(self):
        """Test action transformations and operations."""
        # Test different operation types
        operations = [0, 1, 2, 10, 11, 20, 21]  # Various operation IDs

        for op_id in operations:
            operation = jnp.array(op_id, dtype=jnp.int32)
            selection = jnp.zeros((3, 3), dtype=jnp.bool_)
            selection = selection.at[1, 1].set(True)

            action = create_action(operation=operation, selection=selection)
            validated_action = action.validate(grid_shape=(3, 3), max_operations=35)

            # Verify action is properly created and validated
            assert isinstance(validated_action, Action)
            assert 0 <= validated_action.operation < 35
            assert validated_action.selection.shape == (3, 3)

    def test_action_edge_cases(self):
        """Test action edge cases and boundary conditions."""
        # Test empty selection
        operation = jnp.array(5, dtype=jnp.int32)
        empty_selection = jnp.zeros((3, 3), dtype=jnp.bool_)

        action = create_action(operation=operation, selection=empty_selection)
        validated_action = action.validate(grid_shape=(3, 3), max_operations=35)

        assert isinstance(validated_action, Action)
        assert jnp.sum(validated_action.selection) == 0

        # Test full selection
        full_selection = jnp.ones((3, 3), dtype=jnp.bool_)
        action = create_action(operation=operation, selection=full_selection)
        validated_action = action.validate(grid_shape=(3, 3), max_operations=35)

        assert isinstance(validated_action, Action)
        assert jnp.sum(validated_action.selection) == 9

    def test_action_space_compatibility(self):
        """Test action compatibility with action spaces."""
        # Test various grid sizes
        grid_sizes = [(3, 3), (5, 5), (10, 10)]

        for height, width in grid_sizes:
            operation = jnp.array(7, dtype=jnp.int32)
            selection = jnp.ones((height, width), dtype=jnp.bool_)

            action = create_action(operation=operation, selection=selection)
            validated_action = action.validate(
                grid_shape=(height, width), max_operations=35
            )

            assert validated_action.selection.shape == (height, width)
            assert isinstance(validated_action.operation, jax.Array)


class TestActionIntegration:
    """Test action system integration."""

    def test_action_operation_execution(self):
        """Test that actions can be executed with operations."""
        # This is a basic integration test to ensure actions work with the operation system
        operation = jnp.array(0, dtype=jnp.int32)  # Fill operation
        selection = jnp.zeros((3, 3), dtype=jnp.bool_)
        selection = selection.at[0, 0].set(True)  # Select top-left

        action = create_action(operation=operation, selection=selection)

        # Validate action
        validated_action = action.validate(grid_shape=(3, 3), max_operations=35)

        # Process action
        grid_mask = jnp.ones((3, 3), dtype=jnp.bool_)
        processed_selection = action_handler(validated_action, grid_mask)

        # Verify integration
        assert isinstance(processed_selection, jax.Array)
        assert processed_selection.dtype == jnp.bool_
        assert processed_selection.shape == (3, 3)

        # Verify that the selection was processed correctly
        # The exact behavior depends on the action_handler implementation
        assert jnp.any(processed_selection)  # Some cells should be selected
