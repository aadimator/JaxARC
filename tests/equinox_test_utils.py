"""
Testing utilities specifically for Equinox modules in JaxARC.

This module provides utilities for testing Equinox modules, including mock objects,
validation helpers, and JAX transformation testing for PyTree structures.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxarc.types import ARCLEAction, Grid, JaxArcTask, TaskPair

T = TypeVar("T", bound=eqx.Module)


class EquinoxModuleTester:
    """Comprehensive tester for Equinox modules."""
    
    def __init__(self, module_class: type[T], valid_init_args: dict[str, Any]):
        """
        Initialize the Equinox module tester.
        
        Args:
            module_class: Equinox module class to test
            valid_init_args: Valid arguments for module initialization
        """
        self.module_class = module_class
        self.valid_init_args = valid_init_args
        self._test_instance = None
    
    def get_test_instance(self) -> T:
        """Get a test instance of the module."""
        if self._test_instance is None:
            self._test_instance = self.module_class(**self.valid_init_args)
        return self._test_instance
    
    def test_module_creation(self) -> None:
        """Test that module can be created successfully."""
        instance = self.get_test_instance()
        assert isinstance(instance, self.module_class)
        assert eqx.is_array_like(instance)
    
    def test_pytree_structure(self) -> None:
        """Test that module is a valid PyTree."""
        instance = self.get_test_instance()
        
        # Test that it can be flattened and unflattened
        leaves, treedef = jax.tree.flatten(instance)
        reconstructed = jax.tree.unflatten(treedef, leaves)
        
        # Should be equal after reconstruction
        assert eqx.tree_equal(instance, reconstructed)
    
    def test_jax_transformations(self) -> None:
        """Test that module works with JAX transformations."""
        instance = self.get_test_instance()
        
        # Test jit compilation
        def identity(x):
            return x
        
        jitted_identity = jax.jit(identity)
        jitted_instance = jitted_identity(instance)
        
        assert eqx.tree_equal(instance, jitted_instance)
        
        # Test vmap (if applicable)
        try:
            batched_instances = jax.tree.map(
                lambda x: jnp.stack([x] * 3, axis=0) if isinstance(x, jnp.ndarray) else x,
                instance
            )
            vmapped_identity = jax.vmap(identity)
            vmapped_result = vmapped_identity(batched_instances)
            
            # First element should match original
            first_result = jax.tree.map(lambda x: x[0], vmapped_result)
            assert eqx.tree_equal(instance, first_result)
        except (ValueError, TypeError):
            # Some modules might not be vmappable
            pass
    
    def test_validation_method(self) -> None:
        """Test that __check_init__ validation works correctly."""
        instance = self.get_test_instance()
        
        # If module has __check_init__, it should not raise during normal creation
        if hasattr(instance, '__check_init__'):
            # This should not raise
            instance.__check_init__()
    
    def test_immutability(self) -> None:
        """Test that module is immutable (fields cannot be directly modified)."""
        instance = self.get_test_instance()
        
        # Try to modify fields (should fail or create new instance)
        if hasattr(instance, '__dataclass_fields__'):
            for field_name in instance.__dataclass_fields__:
                original_value = getattr(instance, field_name)
                
                # Direct assignment should fail (frozen dataclass)
                with pytest.raises(AttributeError):
                    setattr(instance, field_name, original_value)
    
    def test_replace_method(self) -> None:
        """Test that eqx.tree_at works for updating fields."""
        instance = self.get_test_instance()
        
        # Test that we can create modified versions using eqx.tree_at
        if hasattr(instance, '__dataclass_fields__'):
            field_names = list(instance.__dataclass_fields__.keys())
            if field_names:
                field_name = field_names[0]
                original_value = getattr(instance, field_name)
                
                # Create a modified version
                modified_instance = eqx.tree_at(
                    lambda x: getattr(x, field_name),
                    instance,
                    original_value
                )
                
                # Should be able to create modified instance
                assert isinstance(modified_instance, self.module_class)
    
    def test_all_properties(self) -> None:
        """Run all tests for the Equinox module."""
        self.test_module_creation()
        self.test_pytree_structure()
        self.test_jax_transformations()
        self.test_validation_method()
        self.test_immutability()
        self.test_replace_method()


class EquinoxMockFactory:
    """Factory for creating mock Equinox modules for testing."""
    
    @staticmethod
    def create_mock_grid(height: int = 5, width: int = 5, key: jax.Array | None = None) -> Grid:
        """
        Create a mock Grid instance.
        
        Args:
            height: Grid height
            width: Grid width
            key: JAX PRNG key for randomization
        
        Returns:
            Mock Grid instance
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        keys = jax.random.split(key, 2)
        
        # Create grid data with valid ARC colors (0-9)
        data = jax.random.randint(keys[0], (height, width), 0, 10, dtype=jnp.int32)
        
        # Create mask (all True for simplicity)
        mask = jnp.ones((height, width), dtype=jnp.bool_)
        
        return Grid(data=data, mask=mask)
    
    @staticmethod
    def create_mock_task_pair(
        input_height: int = 5,
        input_width: int = 5,
        output_height: int = 5,
        output_width: int = 5,
        key: jax.Array | None = None
    ) -> TaskPair:
        """
        Create a mock TaskPair instance.
        
        Args:
            input_height: Input grid height
            input_width: Input grid width
            output_height: Output grid height
            output_width: Output grid width
            key: JAX PRNG key for randomization
        
        Returns:
            Mock TaskPair instance
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        keys = jax.random.split(key, 2)
        
        input_grid = EquinoxMockFactory.create_mock_grid(
            input_height, input_width, keys[0]
        )
        output_grid = EquinoxMockFactory.create_mock_grid(
            output_height, output_width, keys[1]
        )
        
        return TaskPair(input_grid=input_grid, output_grid=output_grid)
    
    @staticmethod
    def create_mock_jax_arc_task(
        num_train_pairs: int = 3,
        num_test_pairs: int = 1,
        max_height: int = 10,
        max_width: int = 10,
        key: jax.Array | None = None
    ) -> JaxArcTask:
        """
        Create a mock JaxArcTask instance.
        
        Args:
            num_train_pairs: Number of training pairs
            num_test_pairs: Number of test pairs
            max_height: Maximum grid height
            max_width: Maximum grid width
            key: JAX PRNG key for randomization
        
        Returns:
            Mock JaxArcTask instance
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        keys = jax.random.split(key, 8)
        
        # Create training data
        input_grids_examples = jax.random.randint(
            keys[0], (num_train_pairs, max_height, max_width), 0, 10, dtype=jnp.int32
        )
        input_masks_examples = jnp.ones((num_train_pairs, max_height, max_width), dtype=jnp.bool_)
        output_grids_examples = jax.random.randint(
            keys[1], (num_train_pairs, max_height, max_width), 0, 10, dtype=jnp.int32
        )
        output_masks_examples = jnp.ones((num_train_pairs, max_height, max_width), dtype=jnp.bool_)
        
        # Create test data
        test_input_grids = jax.random.randint(
            keys[2], (num_test_pairs, max_height, max_width), 0, 10, dtype=jnp.int32
        )
        test_input_masks = jnp.ones((num_test_pairs, max_height, max_width), dtype=jnp.bool_)
        true_test_output_grids = jax.random.randint(
            keys[3], (num_test_pairs, max_height, max_width), 0, 10, dtype=jnp.int32
        )
        true_test_output_masks = jnp.ones((num_test_pairs, max_height, max_width), dtype=jnp.bool_)
        
        # Create task index
        task_index = jnp.array(0, dtype=jnp.int32)
        
        return JaxArcTask(
            input_grids_examples=input_grids_examples,
            input_masks_examples=input_masks_examples,
            output_grids_examples=output_grids_examples,
            output_masks_examples=output_masks_examples,
            num_train_pairs=num_train_pairs,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=true_test_output_grids,
            true_test_output_masks=true_test_output_masks,
            num_test_pairs=num_test_pairs,
            task_index=task_index,
        )
    
    @staticmethod
    def create_mock_arcle_action(
        height: int = 10,
        width: int = 10,
        operation_id: int | None = None,
        key: jax.Array | None = None
    ) -> ARCLEAction:
        """
        Create a mock ARCLEAction instance.
        
        Args:
            height: Selection array height
            width: Selection array width
            operation_id: Specific operation ID (random if None)
            key: JAX PRNG key for randomization
        
        Returns:
            Mock ARCLEAction instance
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        keys = jax.random.split(key, 2)
        
        # Create continuous selection (0.0 to 1.0)
        selection = jax.random.uniform(keys[0], (height, width), dtype=jnp.float32)
        
        # Create operation ID
        if operation_id is None:
            operation = jax.random.randint(keys[1], (), 0, 35, dtype=jnp.int32)
        else:
            operation = jnp.array(operation_id, dtype=jnp.int32)
        
        return ARCLEAction(
            selection=selection,
            operation=operation,
            agent_id=0,
            timestamp=0,
        )


class EquinoxValidationTester:
    """Test validation logic for Equinox modules."""
    
    @staticmethod
    def test_grid_validation():
        """Test Grid validation logic."""
        # Valid grid
        valid_data = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        valid_mask = jnp.array([[True, True], [False, True]], dtype=jnp.bool_)
        valid_grid = Grid(data=valid_data, mask=valid_mask)
        
        # Should not raise
        valid_grid.__check_init__()
        
        # Invalid grid - wrong dtype
        with pytest.raises((ValueError, TypeError)):
            invalid_data = jnp.array([[1.5, 2.5]], dtype=jnp.float32)
            invalid_mask = jnp.array([[True, True]], dtype=jnp.bool_)
            invalid_grid = Grid(data=invalid_data, mask=invalid_mask)
            invalid_grid.__check_init__()
        
        # Invalid grid - shape mismatch
        with pytest.raises((ValueError, TypeError)):
            mismatched_data = jnp.array([[1, 2]], dtype=jnp.int32)
            mismatched_mask = jnp.array([[True], [False]], dtype=jnp.bool_)
            mismatched_grid = Grid(data=mismatched_data, mask=mismatched_mask)
            mismatched_grid.__check_init__()
        
        # Invalid grid - out of range values
        with pytest.raises(ValueError):
            out_of_range_data = jnp.array([[15, 20]], dtype=jnp.int32)
            out_of_range_mask = jnp.array([[True, True]], dtype=jnp.bool_)
            out_of_range_grid = Grid(data=out_of_range_data, mask=out_of_range_mask)
            out_of_range_grid.__check_init__()
    
    @staticmethod
    def test_arcle_action_validation():
        """Test ARCLEAction validation logic."""
        # Valid action
        valid_selection = jnp.array([[0.5, 0.8], [0.2, 1.0]], dtype=jnp.float32)
        valid_operation = jnp.array(15, dtype=jnp.int32)
        valid_action = ARCLEAction(
            selection=valid_selection,
            operation=valid_operation,
            agent_id=0,
            timestamp=0
        )
        
        # Should not raise
        valid_action.__check_init__()
        
        # Invalid action - selection out of range
        with pytest.raises(ValueError):
            invalid_selection = jnp.array([[1.5, 0.8]], dtype=jnp.float32)
            invalid_operation = jnp.array(15, dtype=jnp.int32)
            invalid_action = ARCLEAction(
                selection=invalid_selection,
                operation=invalid_operation,
                agent_id=0,
                timestamp=0
            )
            invalid_action.__check_init__()
        
        # Invalid action - operation out of range
        with pytest.raises(ValueError):
            valid_selection = jnp.array([[0.5, 0.8]], dtype=jnp.float32)
            invalid_operation = jnp.array(50, dtype=jnp.int32)
            invalid_action = ARCLEAction(
                selection=valid_selection,
                operation=invalid_operation,
                agent_id=0,
                timestamp=0
            )
            invalid_action.__check_init__()


# Convenience functions
def run_equinox_module_tests(module_class: type[T], valid_init_args: dict[str, Any]) -> None:
    """Convenience function to test an Equinox module comprehensively."""
    tester = EquinoxModuleTester(module_class, valid_init_args)
    tester.test_all_properties()


def create_test_grid(height: int = 5, width: int = 5) -> Grid:
    """Convenience function to create a test Grid."""
    return EquinoxMockFactory.create_mock_grid(height, width)


def create_test_task() -> JaxArcTask:
    """Convenience function to create a test JaxArcTask."""
    return EquinoxMockFactory.create_mock_jax_arc_task()


def create_test_action() -> ARCLEAction:
    """Convenience function to create a test ARCLEAction."""
    return EquinoxMockFactory.create_mock_arcle_action()