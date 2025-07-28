"""
Test filtered transformations with batch processing (Task 4.3).

This test verifies that equinox.filter_jit works correctly with jax.vmap
for batch processing, ensuring deterministic behavior and correct results.
"""

import time
from typing import Dict, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from jaxarc.envs.config import (
    EnvironmentConfig,
    DatasetConfig, 
    ActionConfig,
    RewardConfig,
    LoggingConfig,
    StorageConfig,
    VisualizationConfig,
    WandbConfig,
    JaxArcConfig
)
from jaxarc.envs.functional import (
    arc_reset, 
    arc_step,
    batch_reset,
    batch_step, 
    create_batch_episode_runner,
    analyze_batch_performance
)
from jaxarc.envs.structured_actions import PointAction, BboxAction, MaskAction
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask


def create_test_config() -> JaxArcConfig:
    """Create a test configuration for batch processing tests."""
    return JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=50),
        dataset=DatasetConfig(max_grid_height=10, max_grid_width=10),
        action=ActionConfig(),
        reward=RewardConfig(),
        visualization=VisualizationConfig(enabled=False),  # Disable for performance
        storage=StorageConfig(),
        logging=LoggingConfig(),
        wandb=WandbConfig(),
    )


def create_test_task(config: JaxArcConfig) -> JaxArcTask:
    """Create a simple test task for batch processing."""
    grid_height = min(6, config.dataset.max_grid_height)
    grid_width = min(6, config.dataset.max_grid_width)
    grid_shape = (grid_height, grid_width)

    # Create input grid with a simple pattern
    input_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
    input_grid = input_grid.at[1:3, 1:3].set(1)  # Small square pattern

    # Create target grid (move pattern to different location)
    target_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
    target_grid = target_grid.at[3:5, 3:5].set(1)  # Same pattern, different location

    # Create masks
    mask = jnp.ones(grid_shape, dtype=jnp.bool_)

    # Pad to max size
    max_shape = (config.dataset.max_grid_height, config.dataset.max_grid_width)
    padded_input = jnp.full(max_shape, -1, dtype=jnp.int32)
    padded_target = jnp.full(max_shape, -1, dtype=jnp.int32)
    padded_mask = jnp.zeros(max_shape, dtype=jnp.bool_)

    padded_input = padded_input.at[:grid_shape[0], :grid_shape[1]].set(input_grid)
    padded_target = padded_target.at[:grid_shape[0], :grid_shape[1]].set(target_grid)
    padded_mask = padded_mask.at[:grid_shape[0], :grid_shape[1]].set(mask)

    return JaxArcTask(
        input_grids_examples=jnp.expand_dims(padded_input, 0),
        output_grids_examples=jnp.expand_dims(padded_target, 0),
        input_masks_examples=jnp.expand_dims(padded_mask, 0),
        output_masks_examples=jnp.expand_dims(padded_mask, 0),
        num_train_pairs=1,
        test_input_grids=jnp.expand_dims(padded_input, 0),
        test_input_masks=jnp.expand_dims(padded_mask, 0),
        true_test_output_grids=jnp.expand_dims(padded_target, 0),
        true_test_output_masks=jnp.expand_dims(padded_mask, 0),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )


class TestFilteredTransformationsBatch:
    """Test suite for filtered transformations with batch processing."""

    def test_filtered_jit_with_vmap(self):
        """Test that @eqx.filter_jit works with jax.vmap."""
        config = create_test_config()
        task = create_test_task(config)
        
        batch_size = 4
        keys = jrandom.split(jrandom.PRNGKey(42), batch_size)
        
        # Test vmap over filtered JIT arc_reset
        vmapped_reset = jax.vmap(arc_reset, in_axes=(0, None, None))
        states, observations = vmapped_reset(keys, config, task)
        
        assert states.working_grid.shape[0] == batch_size
        assert observations.shape[0] == batch_size
        
        # Test vmap over filtered JIT arc_step
        actions = PointAction(
            operation=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
            row=jnp.array([2, 3, 2, 3], dtype=jnp.int32),
            col=jnp.array([2, 3, 3, 2], dtype=jnp.int32)
        )
        
        vmapped_step = jax.vmap(arc_step, in_axes=(0, 0, None))
        new_states, new_obs, rewards, dones, infos = vmapped_step(states, actions, config)
        
        assert new_states.working_grid.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert dones.shape[0] == batch_size

    def test_batch_processing_functions(self):
        """Test dedicated batch processing functions."""
        config = create_test_config()
        task = create_test_task(config)
        
        batch_size = 6
        keys = jrandom.split(jrandom.PRNGKey(123), batch_size)
        
        # Test batch_reset
        states, observations = batch_reset(keys, config, task)
        
        assert states.working_grid.shape[0] == batch_size
        assert observations.shape[0] == batch_size
        assert jnp.all(states.step_count == 0)
        
        # Test batch_step
        actions = PointAction(
            operation=jnp.zeros(batch_size, dtype=jnp.int32),
            row=jnp.arange(batch_size, dtype=jnp.int32) + 2,
            col=jnp.arange(batch_size, dtype=jnp.int32) + 2
        )
        
        new_states, new_obs, rewards, dones, infos = batch_step(states, actions, config)
        
        assert new_states.working_grid.shape[0] == batch_size
        assert rewards.shape[0] == batch_size
        assert dones.shape[0] == batch_size
        assert jnp.all(new_states.step_count == 1)

    def test_deterministic_behavior(self):
        """Test deterministic behavior with PRNG key splitting."""
        config = create_test_config()
        task = create_test_task(config)
        
        base_key = jrandom.PRNGKey(42)
        batch_size = 3
        
        # Same keys should produce identical results
        keys1 = jrandom.split(base_key, batch_size)
        keys2 = jrandom.split(base_key, batch_size)
        
        states1, obs1 = batch_reset(keys1, config, task)
        states2, obs2 = batch_reset(keys2, config, task)
        
        assert jnp.allclose(states1.working_grid, states2.working_grid)
        assert jnp.allclose(states1.similarity_score, states2.similarity_score)
        assert jnp.allclose(obs1, obs2)
        
        # Different keys should produce different split keys
        different_key = jrandom.PRNGKey(12345)
        keys3 = jrandom.split(different_key, batch_size)
        
        assert not jnp.allclose(keys1, keys3)

    def test_batch_correctness(self):
        """Test that batch operations produce correct results."""
        config = create_test_config()
        task = create_test_task(config)
        
        batch_size = 3
        keys = jrandom.split(jrandom.PRNGKey(42), batch_size)
        
        # Compare batch vs individual reset
        batch_states, batch_obs = batch_reset(keys, config, task)
        
        individual_states = []
        individual_obs = []
        for i in range(batch_size):
            state, obs = arc_reset(keys[i], config, task)
            individual_states.append(state)
            individual_obs.append(obs)
        
        for i in range(batch_size):
            assert jnp.allclose(batch_states.working_grid[i], individual_states[i].working_grid)
            assert jnp.allclose(batch_states.target_grid[i], individual_states[i].target_grid)
            assert batch_states.step_count[i] == individual_states[i].step_count
            assert jnp.allclose(batch_obs[i], individual_obs[i])

    def test_performance_characteristics(self):
        """Test performance characteristics of batch processing."""
        config = create_test_config()
        task = create_test_task(config)
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            keys = jrandom.split(jrandom.PRNGKey(42), batch_size)
            
            # Time batch reset (with warmup)
            _ = batch_reset(keys, config, task)
            
            start_time = time.perf_counter()
            states, _ = batch_reset(keys, config, task)
            reset_time = time.perf_counter() - start_time
            
            # Time batch step (with warmup)
            actions = PointAction(
                operation=jnp.zeros(batch_size, dtype=jnp.int32),
                row=jnp.full(batch_size, 3, dtype=jnp.int32),
                col=jnp.full(batch_size, 3, dtype=jnp.int32)
            )
            
            _ = batch_step(states, actions, config)
            
            start_time = time.perf_counter()
            _ = batch_step(states, actions, config)
            step_time = time.perf_counter() - start_time
            
            per_env_reset = (reset_time / batch_size) * 1000  # ms
            per_env_step = (step_time / batch_size) * 1000    # ms
            
            # Performance should be reasonable (< 50ms per environment)
            assert per_env_reset < 50.0, f"Reset too slow: {per_env_reset:.3f}ms"
            assert per_env_step < 50.0, f"Step too slow: {per_env_step:.3f}ms"

    def test_action_type_compatibility(self):
        """Test that all structured action types work with batch processing."""
        config = create_test_config()
        task = create_test_task(config)
        
        batch_size = 3
        keys = jrandom.split(jrandom.PRNGKey(42), batch_size)
        states, _ = batch_reset(keys, config, task)
        
        # Test PointAction
        point_actions = PointAction(
            operation=jnp.zeros(batch_size, dtype=jnp.int32),
            row=jnp.array([2, 3, 4], dtype=jnp.int32),
            col=jnp.array([2, 3, 4], dtype=jnp.int32)
        )
        
        new_states, _, _, _, _ = batch_step(states, point_actions, config)
        assert new_states.working_grid.shape[0] == batch_size
        
        # Test BboxAction
        bbox_actions = BboxAction(
            operation=jnp.zeros(batch_size, dtype=jnp.int32),
            r1=jnp.array([1, 2, 3], dtype=jnp.int32),
            c1=jnp.array([1, 2, 3], dtype=jnp.int32),
            r2=jnp.array([3, 4, 5], dtype=jnp.int32),
            c2=jnp.array([3, 4, 5], dtype=jnp.int32)
        )
        
        new_states, _, _, _, _ = batch_step(states, bbox_actions, config)
        assert new_states.working_grid.shape[0] == batch_size
        
        # Test MaskAction
        grid_shape = (config.dataset.max_grid_height, config.dataset.max_grid_width)
        
        masks = []
        for i in range(batch_size):
            mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
            mask = mask.at[i+1:i+3, i+1:i+3].set(True)
            masks.append(mask)
        
        batched_masks = jnp.stack(masks, axis=0)
        mask_actions = MaskAction(
            operation=jnp.zeros(batch_size, dtype=jnp.int32),
            selection=batched_masks
        )
        
        new_states, _, _, _, _ = batch_step(states, mask_actions, config)
        assert new_states.working_grid.shape[0] == batch_size

    def test_batch_episode_runner(self):
        """Test the batch episode runner functionality."""
        config = create_test_config()
        task = create_test_task(config)
        
        batch_size = 4
        keys = jrandom.split(jrandom.PRNGKey(42), batch_size)
        
        # Create and test batch episode runner
        runner = create_batch_episode_runner(config, task, max_steps=5)
        
        final_states, episode_rewards, episode_lengths = runner(keys, 5)
        
        assert final_states.working_grid.shape[0] == batch_size
        assert episode_rewards.shape[0] == batch_size
        assert episode_lengths.shape[0] == batch_size
        assert jnp.all(episode_lengths <= 5)

    def test_performance_analysis(self):
        """Test the performance analysis functionality."""
        config = create_test_config()
        task = create_test_task(config)
        
        # Test performance analysis with small batch sizes for speed
        analysis = analyze_batch_performance(
            config, task, batch_sizes=[2, 4], num_steps=3
        )
        
        assert 'batch_metrics' in analysis
        assert 'optimal_batch_size' in analysis
        assert 'peak_throughput' in analysis
        assert analysis['optimal_batch_size'] in [2, 4]
        assert analysis['peak_throughput'] > 0
        
        # Check that metrics are reasonable
        for batch_size, metrics in analysis['batch_metrics'].items():
            assert metrics['batch_size'] == batch_size
            assert metrics['total_time'] > 0
            assert metrics['time_per_env'] > 0
            assert metrics['steps_per_second'] > 0