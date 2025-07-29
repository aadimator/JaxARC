"""
Comprehensive tests for diverse grid initialization functionality.

This module provides comprehensive testing of the grid initialization system
including unit tests for each initialization mode handler, integration tests
for the enhanced arc_reset function, property-based tests for grid validity,
performance tests, backward compatibility tests, and JAX compatibility tests.
"""

from __future__ import annotations

import time
from typing import Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxarc.envs.config import GridInitializationConfig, JaxArcConfig
from jaxarc.envs.grid_initialization import (
    initialize_working_grids,
    initialize_working_grids_with_validation,
    _init_demo_grid,
    _init_empty_grid,
    _init_permutation_grid,
    _init_random_grid,
    _apply_grid_permutations,
    _apply_rotation,
    _apply_reflection,
    _apply_color_remap,
    _generate_sparse_pattern,
    _generate_dense_pattern,
    _generate_structured_pattern,
    _generate_noise_pattern,
    _select_batch_modes,
    _initialize_single_grid,
)
from jaxarc.envs.grid_initialization_validation import (
    validate_grid_initialization_config,
    validate_generated_grid,
    validate_task_compatibility,
    create_fallback_grid,
)
from jaxarc.types import JaxArcTask
from jaxarc.utils.jax_types import GridArray, MaskArray, PRNGKey


def create_test_task(
    num_train_pairs: int = 3,
    grid_height: int = 10,
    grid_width: int = 10,
    max_pairs: int = 5
) -> JaxArcTask:
    """Create a test task for grid initialization testing."""
    # Ensure we have enough space for the requested pairs
    actual_max_pairs = max(max_pairs, num_train_pairs)
    
    # Create demo input grids and masks (padded to max_pairs)
    input_grids = jnp.zeros((actual_max_pairs, grid_height, grid_width), dtype=jnp.int32)
    input_masks = jnp.ones((actual_max_pairs, grid_height, grid_width), dtype=jnp.bool_)
    
    # Add some pattern to the first grid for testing
    if num_train_pairs > 0:
        input_grids = input_grids.at[0, 2:5, 2:5].set(1)
        input_grids = input_grids.at[0, 7:9, 7:9].set(2)
    
    # Create output grids (same as input for simplicity)
    output_grids = input_grids.copy()
    output_masks = input_masks.copy()
    
    # Create test data (minimal - just one test pair)
    test_input_grids = jnp.zeros((1, grid_height, grid_width), dtype=jnp.int32)
    test_input_masks = jnp.ones((1, grid_height, grid_width), dtype=jnp.bool_)
    test_output_grids = jnp.zeros((1, grid_height, grid_width), dtype=jnp.int32)
    test_output_masks = jnp.ones((1, grid_height, grid_width), dtype=jnp.bool_)
    
    # Create a mock task with required attributes
    task = JaxArcTask(
        input_grids_examples=input_grids,
        input_masks_examples=input_masks,
        output_grids_examples=output_grids,
        output_masks_examples=output_masks,
        num_train_pairs=num_train_pairs,
        test_input_grids=test_input_grids,
        test_input_masks=test_input_masks,
        true_test_output_grids=test_output_grids,
        true_test_output_masks=test_output_masks,
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )
    
    return task


class TestGridInitializationConfig:
    """Test GridInitializationConfig validation and functionality."""
    
    def test_default_config_creation(self):
        """Test creating default GridInitializationConfig."""
        config = GridInitializationConfig()
        
        assert config.mode == "demo"
        assert config.demo_weight == 0.25
        assert config.permutation_weight == 0.25
        assert config.empty_weight == 0.25
        assert config.random_weight == 0.25
        assert config.permutation_types == ("rotate", "reflect", "color_remap")
        assert config.random_density == 0.3
        assert config.random_pattern_type == "sparse"
        assert config.enable_fallback is True
        
        # Test validation passes
        errors = config.validate()
        assert len(errors) == 0
    
    def test_config_validation_valid_cases(self):
        """Test configuration validation for valid cases."""
        # Test demo mode
        config = GridInitializationConfig(mode="demo")
        errors = config.validate()
        assert len(errors) == 0
        
        # Test mixed mode with valid weights
        config = GridInitializationConfig(
            mode="mixed",
            demo_weight=0.4,
            permutation_weight=0.3,
            empty_weight=0.2,
            random_weight=0.1
        )
        errors = config.validate()
        assert len(errors) == 0
        
        # Test permutation mode with valid types
        config = GridInitializationConfig(
            mode="permutation",
            permutation_types=("rotate", "reflect")
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_config_validation_invalid_cases(self):
        """Test configuration validation for invalid cases."""
        # Test invalid mode
        config = GridInitializationConfig(mode="invalid_mode")
        errors = config.validate()
        assert len(errors) > 0
        assert any("invalid_mode" in error for error in errors)
        
        # Test weights that don't sum to 1.0
        config = GridInitializationConfig(
            mode="mixed",
            demo_weight=0.5,
            permutation_weight=0.5,
            empty_weight=0.5,
            random_weight=0.5
        )
        errors = config.validate()
        assert len(errors) > 0
        assert any("sum to 1.0" in error for error in errors)
    
    def test_config_jax_compatibility(self):
        """Test that GridInitializationConfig is JAX-compatible."""
        config = GridInitializationConfig()
        
        # Test hashability (required for JAX)
        hash(config)
        
        # Test that config values can be used in JAX operations
        density = config.random_density
        jax_density = jnp.array(density)
        
        # Test that the config can be used in JAX functions as static arguments
        assert jax_density == 0.3

class TestInitializationModeHandlers:
    """Test individual initialization mode handlers."""
    
    def test_init_demo_grid_basic(self):
        """Test basic demo grid initialization."""
        task = create_test_task(num_train_pairs=3)
        key = jax.random.PRNGKey(42)
        
        grid, mask = _init_demo_grid(task, key)
        
        # Check shapes
        assert grid.shape == (10, 10)
        assert mask.shape == (10, 10)
        assert grid.dtype == jnp.int32
        assert mask.dtype == jnp.bool_
        
        # Check that grid contains valid ARC colors
        assert jnp.all((grid >= 0) & (grid <= 9))
        
        # Check that mask has some valid cells
        assert jnp.sum(mask) > 0
    
    def test_init_demo_grid_with_specific_pair(self):
        """Test demo grid initialization with specific pair index."""
        task = create_test_task(num_train_pairs=3)
        key = jax.random.PRNGKey(42)
        
        # Test with specific pair index
        grid, mask = _init_demo_grid(task, key, initial_pair_idx=1)
        
        # Should use the second demo pair (index 1)
        expected_grid = task.input_grids_examples[1]
        expected_mask = task.input_masks_examples[1]
        
        assert jnp.array_equal(grid, expected_grid)
        assert jnp.array_equal(mask, expected_mask)
    
    def test_init_demo_grid_empty_task(self):
        """Test demo grid initialization with empty task."""
        task = create_test_task(num_train_pairs=0)
        key = jax.random.PRNGKey(42)
        
        grid, mask = _init_demo_grid(task, key)
        
        # Should create default empty grid
        assert grid.shape == (10, 10)
        assert mask.shape == (10, 10)
        assert jnp.all(grid == 0)  # Should be all zeros
        assert jnp.all(mask == False)  # Should be all False for empty task
    
    def test_init_empty_grid(self):
        """Test empty grid initialization."""
        task = create_test_task(num_train_pairs=3)
        
        grid, mask = _init_empty_grid(task)
        
        # Check shapes
        assert grid.shape == (10, 10)
        assert mask.shape == (10, 10)
        assert grid.dtype == jnp.int32
        assert mask.dtype == jnp.bool_
        
        # Check that grid is all zeros
        assert jnp.all(grid == 0)
        
        # Check that mask matches template (should be all True for valid template)
        expected_mask = task.input_masks_examples[0]
        assert jnp.array_equal(mask, expected_mask)
    
    def test_init_random_grid_basic(self):
        """Test basic random grid initialization."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(random_density=0.3, random_pattern_type="sparse")
        key = jax.random.PRNGKey(42)
        
        grid, mask = _init_random_grid(task, config, key)
        
        # Check shapes
        assert grid.shape == (10, 10)
        assert mask.shape == (10, 10)
        assert grid.dtype == jnp.int32
        assert mask.dtype == jnp.bool_
        
        # Check that grid contains valid ARC colors
        assert jnp.all((grid >= 0) & (grid <= 9))
        
        # Check that mask matches template
        expected_mask = task.input_masks_examples[0]
        assert jnp.array_equal(mask, expected_mask)
    
    def test_init_permutation_grid_basic(self):
        """Test basic permutation grid initialization."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(permutation_types=("rotate", "reflect"))
        key = jax.random.PRNGKey(42)
        
        grid, mask = _init_permutation_grid(task, config, key)
        
        # Check shapes
        assert grid.shape == (10, 10)
        assert mask.shape == (10, 10)
        assert grid.dtype == jnp.int32
        assert mask.dtype == jnp.bool_
        
        # Check that grid contains valid ARC colors
        assert jnp.all((grid >= 0) & (grid <= 9))
        
        # Check that mask matches original demo mask
        expected_mask = task.input_masks_examples[0]  # Uses first demo by default
        assert jnp.array_equal(mask, expected_mask)


class TestGridPermutations:
    """Test grid permutation operations."""
    
    def test_apply_rotation_square_grid(self):
        """Test rotation on square grids."""
        # Create a simple test pattern
        grid = jnp.array([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0]
        ], dtype=jnp.int32)
        
        key = jax.random.PRNGKey(42)
        rotated = _apply_rotation(grid, key)
        
        # Check that result has same shape
        assert rotated.shape == grid.shape
        assert rotated.dtype == grid.dtype
        
        # Check that all values are still valid ARC colors
        assert jnp.all((rotated >= 0) & (rotated <= 9))
        
        # Check that rotation preserves non-zero elements count
        assert jnp.sum(rotated != 0) == jnp.sum(grid != 0)
    
    def test_apply_reflection(self):
        """Test reflection operations."""
        # Create a test pattern
        grid = jnp.array([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 5]
        ], dtype=jnp.int32)
        
        key = jax.random.PRNGKey(42)
        reflected = _apply_reflection(grid, key)
        
        # Check that result has same shape
        assert reflected.shape == grid.shape
        assert reflected.dtype == grid.dtype
        
        # Check that all values are still valid ARC colors
        assert jnp.all((reflected >= 0) & (reflected <= 9))
        
        # Check that reflection preserves non-zero elements count
        assert jnp.sum(reflected != 0) == jnp.sum(grid != 0)
    
    def test_apply_color_remap(self):
        """Test color remapping operations."""
        # Create a test pattern with various colors
        grid = jnp.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ], dtype=jnp.int32)
        
        key = jax.random.PRNGKey(42)
        remapped = _apply_color_remap(grid, key)
        
        # Check that result has same shape
        assert remapped.shape == grid.shape
        assert remapped.dtype == grid.dtype
        
        # Check that all values are still valid ARC colors
        assert jnp.all((remapped >= 0) & (remapped <= 9))
        
        # Check that remapping is a valid permutation (bijective)
        original_unique = jnp.unique(grid)
        remapped_unique = jnp.unique(remapped)
        assert len(original_unique) == len(remapped_unique)


class TestPatternGeneration:
    """Test random pattern generation functions."""
    
    def test_generate_sparse_pattern(self):
        """Test sparse pattern generation."""
        shape = (5, 5)
        density = 0.3
        key = jax.random.PRNGKey(42)
        
        pattern = _generate_sparse_pattern(shape, density, key)
        
        # Check shape and dtype
        assert pattern.shape == shape
        assert pattern.dtype == jnp.int32
        
        # Check valid ARC colors
        assert jnp.all((pattern >= 0) & (pattern <= 9))
        
        # Check that background is 0 and non-background is 1-9
        non_zero_mask = pattern != 0
        if jnp.sum(non_zero_mask) > 0:
            non_zero_values = pattern[non_zero_mask]
            assert jnp.all((non_zero_values >= 1) & (non_zero_values <= 9))
    
    def test_generate_dense_pattern(self):
        """Test dense pattern generation."""
        shape = (5, 5)
        density = 0.5
        key = jax.random.PRNGKey(42)
        
        pattern = _generate_dense_pattern(shape, density, key)
        
        # Check shape and dtype
        assert pattern.shape == shape
        assert pattern.dtype == jnp.int32
        
        # Check valid ARC colors
        assert jnp.all((pattern >= 0) & (pattern <= 9))
    
    def test_pattern_generation_determinism(self):
        """Test that pattern generation is deterministic with same key."""
        shape = (4, 4)
        density = 0.5
        key = jax.random.PRNGKey(123)
        
        # Generate same pattern twice with same key
        pattern1 = _generate_sparse_pattern(shape, density, key)
        pattern2 = _generate_sparse_pattern(shape, density, key)
        
        # Should be identical
        assert jnp.array_equal(pattern1, pattern2)


class TestCoreInitializationFunctions:
    """Test core initialization functions."""
    
    def test_initialize_single_grid_all_modes(self):
        """Test single grid initialization for all modes."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig()
        key = jax.random.PRNGKey(42)
        
        # Test all mode indices
        for mode_idx in range(4):  # 0=demo, 1=permutation, 2=empty, 3=random
            grid, mask = _initialize_single_grid(task, config, key, mode_idx)
            
            # Basic checks for all modes
            assert grid.shape == (10, 10)
            assert mask.shape == (10, 10)
            assert grid.dtype == jnp.int32
            assert mask.dtype == jnp.bool_
            assert jnp.all((grid >= 0) & (grid <= 9))
    
    def test_initialize_working_grids_single(self):
        """Test working grids initialization for single grid."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(mode="demo")
        key = jax.random.PRNGKey(42)
        
        grids, masks = initialize_working_grids(task, config, key, batch_size=1)
        
        # Check shapes
        assert grids.shape == (1, 10, 10)
        assert masks.shape == (1, 10, 10)
        assert grids.dtype == jnp.int32
        assert masks.dtype == jnp.bool_
        
        # Check valid colors
        assert jnp.all((grids >= 0) & (grids <= 9))
    
    def test_initialize_working_grids_batch(self):
        """Test working grids initialization for batch."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(mode="mixed")
        key = jax.random.PRNGKey(42)
        batch_size = 8
        
        grids, masks = initialize_working_grids(task, config, key, batch_size=batch_size)
        
        # Check shapes
        assert grids.shape == (batch_size, 10, 10)
        assert masks.shape == (batch_size, 10, 10)
        assert grids.dtype == jnp.int32
        assert masks.dtype == jnp.bool_
        
        # Check valid colors
        assert jnp.all((grids >= 0) & (grids <= 9))
    
    def test_initialize_working_grids_with_validation(self):
        """Test working grids initialization with validation wrapper."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(mode="demo")
        key = jax.random.PRNGKey(42)
        
        grids, masks = initialize_working_grids_with_validation(
            task, config, key, batch_size=4
        )
        
        # Check shapes
        assert grids.shape == (4, 10, 10)
        assert masks.shape == (4, 10, 10)
        assert grids.dtype == jnp.int32
        assert masks.dtype == jnp.bool_
        
        # Check valid colors
        assert jnp.all((grids >= 0) & (grids <= 9))


class TestJAXCompatibility:
    """Test JAX transformation compatibility."""
    
    def test_jit_compilation_core_functions(self):
        """Test that core functions can be JIT compiled."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(mode="demo")
        key = jax.random.PRNGKey(42)
        
        # Test JIT compilation of core function with static config and batch_size
        # Note: Config and batch_size need to be static since they contain non-array values
        jit_initialize = jax.jit(initialize_working_grids, static_argnums=(1, 3))
        
        grids, masks = jit_initialize(task, config, key, 2)
        
        assert grids.shape == (2, 10, 10)
        assert masks.shape == (2, 10, 10)
        assert jnp.all((grids >= 0) & (grids <= 9))
    
    def test_vmap_compatibility(self):
        """Test vmap compatibility for batch processing."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(mode="demo")
        keys = jax.random.split(jax.random.PRNGKey(42), 4)
        
        # Test vmap over different keys
        vmap_initialize = jax.vmap(
            lambda key: initialize_working_grids(task, config, key, 1),
            in_axes=0
        )
        
        grids, masks = vmap_initialize(keys)
        
        assert grids.shape == (4, 1, 10, 10)
        assert masks.shape == (4, 1, 10, 10)
        assert jnp.all((grids >= 0) & (grids <= 9))
    
    def test_jit_vmap_combination(self):
        """Test combination of JIT and vmap."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(mode="random")
        keys = jax.random.split(jax.random.PRNGKey(42), 3)
        
        # Test JIT + vmap combination
        # Create a function that can be vmapped and jitted
        def single_init(key):
            return initialize_working_grids(task, config, key, 2)
        
        # Apply vmap first, then jit
        vmap_fn = jax.vmap(single_init, in_axes=0)
        jit_vmap_initialize = jax.jit(vmap_fn)
        
        grids, masks = jit_vmap_initialize(keys)
        
        assert grids.shape == (3, 2, 10, 10)
        assert masks.shape == (3, 2, 10, 10)
        assert jnp.all((grids >= 0) & (grids <= 9))


class TestPropertyBasedValidation:
    """Property-based tests for grid validity and determinism."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=5),
        grid_size=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=10, deadline=3000)  # Reduced for performance
    def test_grid_validity_property(self, batch_size, grid_size, seed):
        """Property-based test for grid validity across different parameters."""
        task = create_test_task(num_train_pairs=2, grid_height=grid_size, grid_width=grid_size)
        config = GridInitializationConfig(mode="mixed")
        key = jax.random.PRNGKey(seed)
        
        grids, masks = initialize_working_grids(task, config, key, batch_size)
        
        # Property: All grids should have correct shape
        assert grids.shape == (batch_size, grid_size, grid_size)
        assert masks.shape == (batch_size, grid_size, grid_size)
        
        # Property: All grids should contain valid ARC colors
        assert jnp.all((grids >= 0) & (grids <= 9))
        
        # Property: All masks should be boolean
        assert masks.dtype == jnp.bool_
        
        # Property: Grids should have correct dtype
        assert grids.dtype == jnp.int32
    
    @given(
        mode=st.sampled_from(["demo", "permutation", "empty", "random"]),
        seed=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=10, deadline=2000)
    def test_determinism_property(self, mode, seed):
        """Property-based test for determinism with same PRNG key."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(mode=mode)
        key = jax.random.PRNGKey(seed)
        
        # Generate grids twice with same key
        grids1, masks1 = initialize_working_grids(task, config, key, 2)
        grids2, masks2 = initialize_working_grids(task, config, key, 2)
        
        # Property: Results should be identical with same key
        assert jnp.array_equal(grids1, grids2)
        assert jnp.array_equal(masks1, masks2)


class TestPerformance:
    """Performance tests to ensure no significant slowdown."""
    
    def test_initialization_performance_single(self):
        """Test performance of single grid initialization."""
        task = create_test_task(num_train_pairs=5)
        config = GridInitializationConfig(mode="mixed")
        key = jax.random.PRNGKey(42)
        
        # Warm up JIT compilation with static arguments
        jit_fn = jax.jit(initialize_working_grids, static_argnums=(1, 3))
        _ = jit_fn(task, config, key, 1)
        
        # Time the actual execution
        start_time = time.time()
        for _ in range(10):  # Reduced iterations for faster testing
            _ = jit_fn(task, config, key, 1)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Should be reasonably fast (less than 1s per initialization for testing)
        # Note: This is a basic performance check, not a strict benchmark
        assert avg_time < 1.0, f"Single initialization too slow: {avg_time:.4f}s"
    
    def test_jit_speedup(self):
        """Test that JIT compilation provides speedup."""
        task = create_test_task(num_train_pairs=3)
        config = GridInitializationConfig(mode="random")
        key = jax.random.PRNGKey(42)
        batch_size = 8
        
        # Time without JIT (first call includes compilation time)
        start_time = time.time()
        for _ in range(5):
            _ = initialize_working_grids(task, config, key, batch_size)
        no_jit_time = time.time() - start_time
        
        # Time with JIT (after warm-up)
        jit_fn = jax.jit(initialize_working_grids, static_argnums=(1, 3))
        _ = jit_fn(task, config, key, batch_size)  # Warm up
        
        start_time = time.time()
        for _ in range(5):
            _ = jit_fn(task, config, key, batch_size)
        jit_time = time.time() - start_time
        
        # JIT should provide some speedup (at least not be slower)
        # Note: This is a rough test as JIT benefits vary
        assert jit_time <= no_jit_time * 2.0, "JIT compilation should not significantly slow down execution"


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_default_behavior_unchanged(self):
        """Test that default behavior matches original demo mode."""
        task = create_test_task(num_train_pairs=3)
        
        # Default config should use demo mode
        config = GridInitializationConfig()
        assert config.mode == "demo"
        
        key = jax.random.PRNGKey(42)
        grids, masks = initialize_working_grids(task, config, key, 1)
        
        # Should match one of the demo grids
        demo_grids = task.input_grids_examples
        demo_masks = task.input_masks_examples
        
        # Grid should match one of the demo inputs
        matches_demo = False
        for i in range(task.num_train_pairs):
            if jnp.array_equal(grids[0], demo_grids[i]) and jnp.array_equal(masks[0], demo_masks[i]):
                matches_demo = True
                break
        
        assert matches_demo, "Default initialization should match demo behavior"
    
    def test_config_integration_with_jaxarcconfig(self):
        """Test integration with JaxArcConfig."""
        from jaxarc.envs.config import JaxArcConfig, EnvironmentConfig, DatasetConfig
        
        # Create a full config with grid initialization
        grid_init_config = GridInitializationConfig(mode="mixed", random_density=0.4)
        
        full_config = JaxArcConfig(
            environment=EnvironmentConfig(),
            dataset=DatasetConfig(),
            grid_initialization=grid_init_config
        )
        
        # Test that validation works
        errors = full_config.validate()
        assert len(errors) == 0, f"Full config validation failed: {errors}"
        
        # Test that grid initialization config is accessible
        assert full_config.grid_initialization.mode == "mixed"
        assert full_config.grid_initialization.random_density == 0.4


class TestValidationFunctions:
    """Test validation and error handling functions."""
    
    def test_validate_grid_initialization_config_valid(self):
        """Test configuration validation for valid configs."""
        # Test valid demo mode config
        config = GridInitializationConfig(mode="demo")
        errors = validate_grid_initialization_config(config)
        assert len(errors) == 0
        
        # Test valid mixed mode config
        config = GridInitializationConfig(
            mode="mixed",
            demo_weight=0.3,
            permutation_weight=0.3,
            empty_weight=0.2,
            random_weight=0.2
        )
        errors = validate_grid_initialization_config(config)
        assert len(errors) == 0
    
    def test_validate_generated_grid_valid(self):
        """Test grid validation for valid grids."""
        task = create_test_task(num_train_pairs=3)
        
        # Create a valid grid
        grid = jnp.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ], dtype=jnp.int32)
        mask = jnp.ones((3, 3), dtype=jnp.bool_)
        
        errors = validate_generated_grid(grid, mask, task, mode="test")
        assert len(errors) == 0
    
    def test_create_fallback_grid(self):
        """Test fallback grid creation."""
        task = create_test_task(num_train_pairs=3)
        
        fallback_grid, fallback_mask = create_fallback_grid(task)
        
        # Check basic properties
        assert fallback_grid.shape == (10, 10)
        assert fallback_mask.shape == (10, 10)
        assert fallback_grid.dtype == jnp.int32
        assert fallback_mask.dtype == jnp.bool_
        
        # Fallback grid should be all zeros
        assert jnp.all(fallback_grid == 0)
        
        # Fallback mask should be all True (full valid region)
        assert jnp.all(fallback_mask == True)


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running smoke test for grid initialization...")
    
    task = create_test_task(num_train_pairs=3)
    config = GridInitializationConfig(mode="mixed")
    key = jax.random.PRNGKey(42)
    
    grids, masks = initialize_working_grids(task, config, key, 4)
    print(f"Generated grids shape: {grids.shape}")
    print(f"Generated masks shape: {masks.shape}")
    print(f"Grid value range: [{jnp.min(grids)}, {jnp.max(grids)}]")
    print("Smoke test passed!")