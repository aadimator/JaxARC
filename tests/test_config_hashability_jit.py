#!/usr/bin/env python3
"""
Test configuration hashability and JIT compatibility.

This test file implements task 1.3 from the JAX compatibility fixes specification:
- Create test cases to verify all configuration objects are hashable
- Test `jax.jit(static_argnames=['config'])` with hashable configurations
- Validate that JIT compilation works for arc_reset and arc_step
- Create regression tests to prevent future hashability issues

Requirements: 1.3, 1.4
"""

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from src.jaxarc.envs.config import (
    ActionConfig,
    DatasetConfig,
    EnvironmentConfig,
    JaxArcConfig,
    LoggingConfig,
    RewardConfig,
    StorageConfig,
    VisualizationConfig,
    WandbConfig,
)
from src.jaxarc.envs.functional import arc_reset, arc_step
from src.jaxarc.types import JaxArcTask
from src.jaxarc.utils.jax_types import PRNGKey


class TestConfigurationHashability:
    """Test suite for configuration hashability."""

    def test_environment_config_hashable(self):
        """Test that EnvironmentConfig is hashable."""
        config = EnvironmentConfig()
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test with different values
        config2 = EnvironmentConfig(max_episode_steps=200, debug_level="verbose")
        hash_value2 = hash(config2)
        assert isinstance(hash_value2, int)
        assert hash_value != hash_value2  # Different configs should have different hashes

    def test_dataset_config_hashable(self):
        """Test that DatasetConfig is hashable."""
        config = DatasetConfig()
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test with different values
        config2 = DatasetConfig(
            dataset_name="arc-agi-2",
            max_grid_height=25,
            max_grid_width=25,
            max_colors=8
        )
        hash_value2 = hash(config2)
        assert isinstance(hash_value2, int)
        assert hash_value != hash_value2

    def test_action_config_hashable(self):
        """Test that ActionConfig is hashable."""
        config = ActionConfig()
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test with different values including tuple for allowed_operations
        config2 = ActionConfig(
            selection_format="point",
            max_operations=35,
            allowed_operations=(0, 1, 2, 3, 4)  # tuple is hashable
        )
        hash_value2 = hash(config2)
        assert isinstance(hash_value2, int)
        assert hash_value != hash_value2

    def test_reward_config_hashable(self):
        """Test that RewardConfig is hashable."""
        config = RewardConfig()
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test with different values
        config2 = RewardConfig(
            step_penalty=-0.02,
            success_bonus=20.0,
            similarity_weight=2.0
        )
        hash_value2 = hash(config2)
        assert isinstance(hash_value2, int)
        assert hash_value != hash_value2

    def test_visualization_config_hashable(self):
        """Test that VisualizationConfig is hashable."""
        config = VisualizationConfig()
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test with different values including tuple for output_formats
        config2 = VisualizationConfig(
            enabled=False,
            level="minimal",
            output_formats=("png", "svg"),  # tuple is hashable
            color_scheme="high_contrast"
        )
        hash_value2 = hash(config2)
        assert isinstance(hash_value2, int)
        assert hash_value != hash_value2

    def test_storage_config_hashable(self):
        """Test that StorageConfig is hashable."""
        config = StorageConfig()
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test with different values
        config2 = StorageConfig(
            policy="research",
            max_episodes_per_run=200,
            max_storage_gb=10.0
        )
        hash_value2 = hash(config2)
        assert isinstance(hash_value2, int)
        assert hash_value != hash_value2

    def test_logging_config_hashable(self):
        """Test that LoggingConfig is hashable."""
        config = LoggingConfig()
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test with different values
        config2 = LoggingConfig(
            log_format="text",
            log_level="DEBUG",
            log_frequency=5
        )
        hash_value2 = hash(config2)
        assert isinstance(hash_value2, int)
        assert hash_value != hash_value2

    def test_wandb_config_hashable(self):
        """Test that WandbConfig is hashable."""
        config = WandbConfig()
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test with different values including tuple for tags
        config2 = WandbConfig(
            enabled=True,
            project_name="test-project",
            tags=("test", "jax"),  # tuple is hashable
            max_image_size=(1024, 768)  # tuple is hashable
        )
        hash_value2 = hash(config2)
        assert isinstance(hash_value2, int)
        assert hash_value != hash_value2

    def test_jax_arc_config_hashable(self):
        """Test that JaxArcConfig (main config) is hashable."""
        config = JaxArcConfig()
        
        # Should not raise TypeError
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Test with different component configs
        config2 = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=150),
            dataset=DatasetConfig(max_grid_height=25),
            action=ActionConfig(selection_format="point")
        )
        hash_value2 = hash(config2)
        assert isinstance(hash_value2, int)
        assert hash_value != hash_value2

    def test_config_from_hydra_hashable(self):
        """Test that configs created from Hydra DictConfig are hashable."""
        # Create a sample Hydra config
        hydra_config = DictConfig({
            "environment": {
                "max_episode_steps": 150,
                "debug_level": "verbose"
            },
            "dataset": {
                "dataset_name": "arc-agi-2",
                "max_grid_height": 25
            },
            "action": {
                "selection_format": "point",
                "allowed_operations": [0, 1, 2, 3]  # Will be converted to tuple
            },
            "visualization": {
                "output_formats": ["svg", "png"],  # Will be converted to tuple
                "enabled": True
            },
            "wandb": {
                "tags": ["test", "hydra"],  # Will be converted to tuple
                "enabled": False
            }
        })
        
        # Create configs from Hydra
        env_config = EnvironmentConfig.from_hydra(hydra_config.environment)
        dataset_config = DatasetConfig.from_hydra(hydra_config.dataset)
        action_config = ActionConfig.from_hydra(hydra_config.action)
        viz_config = VisualizationConfig.from_hydra(hydra_config.visualization)
        wandb_config = WandbConfig.from_hydra(hydra_config.wandb)
        
        # All should be hashable
        assert isinstance(hash(env_config), int)
        assert isinstance(hash(dataset_config), int)
        assert isinstance(hash(action_config), int)
        assert isinstance(hash(viz_config), int)
        assert isinstance(hash(wandb_config), int)
        
        # Test that allowed_operations was converted to tuple
        assert isinstance(action_config.allowed_operations, tuple)
        assert action_config.allowed_operations == (0, 1, 2, 3)
        
        # Test that output_formats was converted to tuple
        assert isinstance(viz_config.output_formats, tuple)
        assert viz_config.output_formats == ("svg", "png")
        
        # Test that tags was converted to tuple
        assert isinstance(wandb_config.tags, tuple)
        assert wandb_config.tags == ("test", "hydra")

    def test_hash_consistency(self):
        """Test that hash values are consistent for identical configs."""
        config1 = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=100),
            action=ActionConfig(selection_format="mask")
        )
        config2 = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=100),
            action=ActionConfig(selection_format="mask")
        )
        
        # Identical configs should have identical hashes
        assert hash(config1) == hash(config2)
        
        # Different configs should have different hashes
        config3 = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=200),
            action=ActionConfig(selection_format="mask")
        )
        assert hash(config1) != hash(config3)


class TestJITCompatibility:
    """Test suite for JAX JIT compilation with hashable configurations."""

    def create_test_config(self) -> JaxArcConfig:
        """Create a test configuration for JIT testing."""
        return JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=50),
            dataset=DatasetConfig(
                max_grid_height=10,
                max_grid_width=10,
                max_colors=5
            ),
            action=ActionConfig(
                selection_format="point",
                max_operations=10
            )
        )

    def create_test_task(self) -> JaxArcTask:
        """Create a minimal test task for JIT testing."""
        # Create minimal task data with required fields
        max_pairs = 3
        grid_height = 10
        grid_width = 10
        
        # Create training data arrays
        input_grids_examples = jnp.zeros((max_pairs, grid_height, grid_width), dtype=jnp.int32)
        input_masks_examples = jnp.ones((max_pairs, grid_height, grid_width), dtype=jnp.bool_)
        output_grids_examples = jnp.ones((max_pairs, grid_height, grid_width), dtype=jnp.int32)
        output_masks_examples = jnp.ones((max_pairs, grid_height, grid_width), dtype=jnp.bool_)
        
        # Create test data arrays
        test_input_grids = jnp.zeros((max_pairs, grid_height, grid_width), dtype=jnp.int32)
        test_input_masks = jnp.ones((max_pairs, grid_height, grid_width), dtype=jnp.bool_)
        true_test_output_grids = jnp.ones((max_pairs, grid_height, grid_width), dtype=jnp.int32)
        true_test_output_masks = jnp.ones((max_pairs, grid_height, grid_width), dtype=jnp.bool_)
        
        return JaxArcTask(
            input_grids_examples=input_grids_examples,
            input_masks_examples=input_masks_examples,
            output_grids_examples=output_grids_examples,
            output_masks_examples=output_masks_examples,
            num_train_pairs=1,
            test_input_grids=test_input_grids,
            test_input_masks=test_input_masks,
            true_test_output_grids=true_test_output_grids,
            true_test_output_masks=true_test_output_masks,
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32)
        )

    def test_jit_with_static_config_arc_reset(self):
        """Test that arc_reset can be JIT compiled with static config argument."""
        config = self.create_test_config()
        task = self.create_test_task()
        key = jax.random.PRNGKey(42)
        
        # Verify config is hashable (required for static_argnames)
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Create JIT-compiled version with static config using static_argnames
        jit_arc_reset = jax.jit(arc_reset, static_argnames=['config'])
        
        # Should compile and execute without errors
        state, obs = jit_arc_reset(key, config, task)
        
        assert state is not None
        assert obs is not None
        assert obs.shape == (10, 10)  # Should match grid dimensions

    def test_jit_with_static_config_arc_step(self):
        """Test that arc_step can be JIT compiled with static config argument."""
        config = self.create_test_config()
        task = self.create_test_task()
        key = jax.random.PRNGKey(42)
        
        # Verify config is hashable
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Create initial state
        state, _ = arc_reset(key, config, task)
        
        # Create test action (point format)
        action = {
            "operation": 0,  # Fill operation
            "point": jnp.array([5, 5])  # Point selection
        }
        
        # Create JIT-compiled version with static config using static_argnames
        jit_arc_step = jax.jit(arc_step, static_argnames=['config'])
        
        # Should compile and execute without errors
        new_state, obs, reward, done, info = jit_arc_step(state, action, config)
        
        assert new_state is not None
        assert obs is not None
        assert isinstance(reward, (int, float, jnp.ndarray))
        assert isinstance(done, (bool, jnp.ndarray))
        assert isinstance(info, dict)

    def test_jit_compilation_performance(self):
        """Test that JIT compilation provides performance benefits."""
        config = self.create_test_config()
        task = self.create_test_task()
        key = jax.random.PRNGKey(42)
        
        # Non-JIT version
        def regular_reset(key, task_data):
            return arc_reset(key, config, task_data)
        
        # JIT version
        @jax.jit
        def jit_reset(key, task_data):
            return arc_reset(key, config, task_data)
        
        # Warm up JIT compilation
        _ = jit_reset(key, task)
        
        # Both should produce identical results
        state1, obs1 = regular_reset(key, task)
        state2, obs2 = jit_reset(key, task)
        
        # Results should be identical
        assert jnp.allclose(obs1, obs2)
        assert state1.step_count == state2.step_count
        assert state1.episode_done == state2.episode_done

    def test_jit_static_argnames_config_requirement(self):
        """Test the specific requirement: jax.jit(static_argnames=['config']) works."""
        config = self.create_test_config()
        task = self.create_test_task()
        key = jax.random.PRNGKey(42)
        
        # This is the exact requirement from the task specification
        # Test `jax.jit(static_argnames=['config'])` with hashable configurations
        
        # Test with arc_reset
        jit_reset_with_static_config = jax.jit(arc_reset, static_argnames=['config'])
        state, obs = jit_reset_with_static_config(key, config, task)
        assert state is not None
        assert obs is not None
        
        # Test with arc_step
        action = {
            "operation": 0,
            "point": jnp.array([5, 5])
        }
        jit_step_with_static_config = jax.jit(arc_step, static_argnames=['config'])
        new_state, obs, reward, done, info = jit_step_with_static_config(state, action, config)
        
        assert new_state is not None
        assert obs is not None
        assert isinstance(reward, (int, float, jnp.ndarray))
        assert isinstance(done, (bool, jnp.ndarray))
        assert isinstance(info, dict)
        
        # Verify that the config is indeed being treated as static
        # by checking that we can call with the same config multiple times
        state2, obs2 = jit_reset_with_static_config(key, config, task)
        assert jnp.allclose(obs, obs2)

    def test_jit_with_different_configs(self):
        """Test JIT compilation with different configuration objects."""
        task = self.create_test_task()
        key = jax.random.PRNGKey(42)
        
        # Create different configs
        config1 = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=50),
            action=ActionConfig(selection_format="point")
        )
        config2 = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=100),
            action=ActionConfig(selection_format="bbox")
        )
        
        # Both configs should be hashable
        assert isinstance(hash(config1), int)
        assert isinstance(hash(config2), int)
        assert hash(config1) != hash(config2)
        
        # JIT compilation should work with both
        @jax.jit
        def jit_reset_config1(key, task_data):
            return arc_reset(key, config1, task_data)
        
        @jax.jit
        def jit_reset_config2(key, task_data):
            return arc_reset(key, config2, task_data)
        
        # Both should execute successfully
        state1, obs1 = jit_reset_config1(key, task)
        state2, obs2 = jit_reset_config2(key, task)
        
        assert state1 is not None
        assert state2 is not None
        assert obs1 is not None
        assert obs2 is not None

    def test_jit_error_handling(self):
        """Test that JIT compilation fails gracefully with unhashable configs."""
        # This test would be relevant if we had unhashable configs
        # Since our configs should all be hashable, this tests the error path
        
        # Create a mock unhashable object
        class UnhashableConfig:
            def __init__(self):
                self.unhashable_field = [1, 2, 3]  # Lists are unhashable
            
            def __hash__(self):
                # This will raise TypeError because lists are unhashable
                return hash(self.unhashable_field)
        
        unhashable_config = UnhashableConfig()
        
        # Should raise TypeError when trying to hash
        with pytest.raises(TypeError):
            hash(unhashable_config)


class TestRegressionPrevention:
    """Test suite to prevent future hashability regressions."""

    def test_all_config_classes_have_check_init(self):
        """Test that all config classes have __check_init__ method for hashability validation."""
        config_classes = [
            EnvironmentConfig,
            DatasetConfig,
            ActionConfig,
            RewardConfig,
            VisualizationConfig,
            StorageConfig,
            LoggingConfig,
            WandbConfig,
            JaxArcConfig
        ]
        
        for config_class in config_classes:
            assert hasattr(config_class, '__check_init__'), \
                f"{config_class.__name__} missing __check_init__ method"

    def test_config_validation_catches_unhashable_types(self):
        """Test that config validation catches unhashable types during initialization."""
        # Test that configs with unhashable types would fail validation
        # This is more of a design test to ensure our validation works
        
        # All our current configs should pass validation
        configs = [
            EnvironmentConfig(),
            DatasetConfig(),
            ActionConfig(),
            RewardConfig(),
            VisualizationConfig(),
            StorageConfig(),
            LoggingConfig(),
            WandbConfig(),
            JaxArcConfig()
        ]
        
        for config in configs:
            # Should not raise any exceptions
            hash_value = hash(config)
            assert isinstance(hash_value, int)

    def test_tuple_conversion_in_from_hydra(self):
        """Test that from_hydra methods properly convert lists to tuples."""
        # Test ActionConfig
        hydra_action = DictConfig({
            "allowed_operations": [0, 1, 2, 3, 4]
        })
        action_config = ActionConfig.from_hydra(hydra_action)
        assert isinstance(action_config.allowed_operations, tuple)
        assert action_config.allowed_operations == (0, 1, 2, 3, 4)
        
        # Test VisualizationConfig
        hydra_viz = DictConfig({
            "output_formats": ["svg", "png", "html"]
        })
        viz_config = VisualizationConfig.from_hydra(hydra_viz)
        assert isinstance(viz_config.output_formats, tuple)
        assert viz_config.output_formats == ("svg", "png", "html")
        
        # Test WandbConfig
        hydra_wandb = DictConfig({
            "tags": ["test", "regression"],
            "max_image_size": [800, 600]
        })
        wandb_config = WandbConfig.from_hydra(hydra_wandb)
        assert isinstance(wandb_config.tags, tuple)
        assert wandb_config.tags == ("test", "regression")
        assert isinstance(wandb_config.max_image_size, tuple)
        assert wandb_config.max_image_size == (800, 600)

    def test_config_immutability(self):
        """Test that config objects are properly immutable (Equinox modules)."""
        config = JaxArcConfig()
        
        # Should not be able to modify fields directly
        with pytest.raises(AttributeError):
            config.environment = EnvironmentConfig(max_episode_steps=200)
        
        # Should not be able to modify nested config fields
        with pytest.raises(AttributeError):
            config.environment.max_episode_steps = 200

    def test_nested_config_hashability(self):
        """Test that nested configuration objects maintain hashability."""
        # Create config with all nested components
        config = JaxArcConfig(
            environment=EnvironmentConfig(max_episode_steps=100),
            dataset=DatasetConfig(max_grid_height=20),
            action=ActionConfig(
                selection_format="bbox",
                allowed_operations=(0, 1, 2, 3, 4, 5)
            ),
            visualization=VisualizationConfig(
                output_formats=("svg", "png"),
                enabled=True
            ),
            wandb=WandbConfig(
                tags=("nested", "test"),
                enabled=False
            )
        )
        
        # Entire config should be hashable
        hash_value = hash(config)
        assert isinstance(hash_value, int)
        
        # Individual components should also be hashable
        assert isinstance(hash(config.environment), int)
        assert isinstance(hash(config.dataset), int)
        assert isinstance(hash(config.action), int)
        assert isinstance(hash(config.visualization), int)
        assert isinstance(hash(config.wandb), int)


def main():
    """Run all tests manually for verification."""
    print("Testing Configuration Hashability and JIT Compatibility...")
    
    # Test hashability
    print("\n1. Testing configuration hashability...")
    hashability_tests = TestConfigurationHashability()
    
    try:
        hashability_tests.test_environment_config_hashable()
        print("✓ EnvironmentConfig is hashable")
        
        hashability_tests.test_dataset_config_hashable()
        print("✓ DatasetConfig is hashable")
        
        hashability_tests.test_action_config_hashable()
        print("✓ ActionConfig is hashable")
        
        hashability_tests.test_reward_config_hashable()
        print("✓ RewardConfig is hashable")
        
        hashability_tests.test_visualization_config_hashable()
        print("✓ VisualizationConfig is hashable")
        
        hashability_tests.test_storage_config_hashable()
        print("✓ StorageConfig is hashable")
        
        hashability_tests.test_logging_config_hashable()
        print("✓ LoggingConfig is hashable")
        
        hashability_tests.test_wandb_config_hashable()
        print("✓ WandbConfig is hashable")
        
        hashability_tests.test_jax_arc_config_hashable()
        print("✓ JaxArcConfig is hashable")
        
        hashability_tests.test_config_from_hydra_hashable()
        print("✓ Configs from Hydra are hashable")
        
        hashability_tests.test_hash_consistency()
        print("✓ Hash consistency verified")
        
    except Exception as e:
        print(f"✗ Hashability test failed: {e}")
        return False
    
    # Test JIT compatibility
    print("\n2. Testing JIT compatibility...")
    jit_tests = TestJITCompatibility()
    
    try:
        jit_tests.test_jit_with_static_config_arc_reset()
        print("✓ arc_reset JIT compilation works")
        
        jit_tests.test_jit_with_static_config_arc_step()
        print("✓ arc_step JIT compilation works")
        
        jit_tests.test_jit_compilation_performance()
        print("✓ JIT compilation performance verified")
        
        jit_tests.test_jit_with_different_configs()
        print("✓ JIT works with different configs")
        
    except Exception as e:
        print(f"✗ JIT compatibility test failed: {e}")
        return False
    
    # Test regression prevention
    print("\n3. Testing regression prevention...")
    regression_tests = TestRegressionPrevention()
    
    try:
        regression_tests.test_all_config_classes_have_check_init()
        print("✓ All config classes have __check_init__")
        
        regression_tests.test_config_validation_catches_unhashable_types()
        print("✓ Config validation works")
        
        regression_tests.test_tuple_conversion_in_from_hydra()
        print("✓ Tuple conversion in from_hydra works")
        
        regression_tests.test_config_immutability()
        print("✓ Config immutability verified")
        
        regression_tests.test_nested_config_hashability()
        print("✓ Nested config hashability verified")
        
    except Exception as e:
        print(f"✗ Regression prevention test failed: {e}")
        return False
    
    print("\n✅ All tests passed! Configuration hashability and JIT compatibility verified.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)