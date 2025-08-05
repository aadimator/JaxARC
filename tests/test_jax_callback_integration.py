"""
Integration tests for JAX callback system with ExperimentLogger.

This test module verifies that the updated JAX callback integration works
correctly with JAX transformations (jit, vmap, pmap) and the new ExperimentLogger.
"""

import tempfile
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
# import pytest  # Not available in this environment

from jaxarc.envs.config import JaxArcConfig, StorageConfig, EnvironmentConfig, DatasetConfig
from jaxarc.envs.environment import ArcEnvironment
from jaxarc.envs.structured_actions import create_point_action
from jaxarc.utils.logging.experiment_logger import ExperimentLogger
from jaxarc.utils.visualization.jax_callbacks import (
    jax_save_step_visualization,
    jax_save_episode_summary,
    create_step_logging_callback,
    create_episode_logging_callback,
    ensure_jax_callback_compatibility,
    validate_jax_callback_data
)


class TestJaxCallbackDataValidation:
    """Test JAX callback data validation functionality."""
    
    def test_validate_jax_callback_data(self):
        """Test that JAX callback data validation works correctly."""
        # Create test data with JAX arrays
        test_data = {
            'step_num': 1,
            'reward': 0.5,
            'jax_array': jnp.array([[1, 2], [3, 4]]),
            'nested': {
                'another_array': jnp.array([1.0, 2.0, 3.0]),
                'scalar': 42
            },
            'list_with_arrays': [jnp.array([1, 2]), 'string', 123]
        }
        
        # Validate the data
        validated_data = validate_jax_callback_data(test_data)
        
        # Check that JAX arrays were serialized
        assert isinstance(validated_data['jax_array'], np.ndarray), "JAX array should be serialized to numpy array"
        assert isinstance(validated_data['nested']['another_array'], np.ndarray), "Nested JAX array should be serialized"
        assert validated_data['nested']['scalar'] == 42, "Non-JAX data should be preserved"
    
    def test_callback_compatibility_wrapper(self):
        """Test that callback compatibility wrapper works correctly."""
        # Create a test callback that processes JAX arrays
        def test_callback(data: Dict[str, Any]) -> None:
            # This callback expects serialized data
            assert isinstance(data['array'], np.ndarray), "Array should be serialized"
            assert data['scalar'] == 42, "Scalar should be preserved"
        
        # Wrap the callback for JAX compatibility
        jax_compatible_callback = ensure_jax_callback_compatibility(test_callback)
        
        # Test with JAX array data
        test_data = {
            'array': jnp.array([1, 2, 3]),
            'scalar': 42
        }
        
        # This should not raise an exception
        jax_compatible_callback(test_data)


class TestJaxTransformationCompatibility:
    """Test JAX transformation compatibility (jit, vmap, pmap)."""
    
    def test_jax_jit_compatibility(self):
        """Test that callbacks work correctly with JAX JIT compilation."""
        def jax_function_with_callback(x: jnp.ndarray, callback_func) -> jnp.ndarray:
            """Test JAX function that uses debug callbacks."""
            result = x * 2
            
            # Use JAX debug callback
            test_data = {'value': result, 'input': x}
            jax.debug.callback(callback_func, test_data)
            
            return result
        
        # JIT compile with static callback
        jit_function_with_callback = jax.jit(jax_function_with_callback, static_argnames=['callback_func'])
        
        # Create a simple callback that doesn't require logger
        def test_callback(data: Dict[str, Any]) -> None:
            # Just verify we can access the data
            assert 'value' in data, "Callback should receive data"
            assert 'input' in data, "Callback should receive input data"
        
        # Make callback JAX-compatible
        safe_callback = ensure_jax_callback_compatibility(test_callback)
        
        # Test with JIT compilation
        test_input = jnp.array([1.0, 2.0, 3.0])
        result = jit_function_with_callback(test_input, safe_callback)
        
        # Verify result is correct
        expected = test_input * 2
        assert jnp.allclose(result, expected), "JIT function should produce correct result"
    
    def test_jax_vmap_compatibility(self):
        """Test that callbacks work correctly with JAX vmap."""
        def batch_function_with_callback(x: jnp.ndarray, callback_func) -> jnp.ndarray:
            """Test function for vmap compatibility."""
            result = x + 1
            
            # Use callback with batch data
            test_data = {'batch_value': result}
            jax.debug.callback(callback_func, test_data)
            
            return result
        
        # JIT compile with static callback
        jit_batch_function = jax.jit(batch_function_with_callback, static_argnames=['callback_func'])
        
        # Create callback for batch processing
        def batch_callback(data: Dict[str, Any]) -> None:
            # Just verify we can access the batch data
            assert 'batch_value' in data, "Callback should receive batch data"
        
        # Make callback JAX-compatible
        safe_callback = ensure_jax_callback_compatibility(batch_callback)
        
        # Create vectorized function
        vmapped_func = jax.vmap(jit_batch_function, in_axes=(0, None))
        
        # Test with batch input
        batch_input = jnp.array([[1.0], [2.0], [3.0]])
        result = vmapped_func(batch_input, safe_callback)
        
        # Verify result shape and values
        expected = batch_input + 1
        assert result.shape == expected.shape, "vmap should preserve batch dimensions"
        assert jnp.allclose(result, expected), "vmap should produce correct results"


class TestCallbackFactoryFunctions:
    """Test callback factory functions."""
    
    def test_callback_factory_functions(self):
        """Test that callback factory functions work correctly."""
        # Create a mock logger that just stores calls
        class MockLogger:
            def __init__(self):
                self.step_calls = []
                self.episode_calls = []
            
            def log_step(self, step_data):
                self.step_calls.append(step_data)
            
            def log_episode_summary(self, summary_data):
                self.episode_calls.append(summary_data)
        
        # Create mock logger
        mock_logger = MockLogger()
        
        # Create bound callbacks
        step_callback = create_step_logging_callback(mock_logger)
        episode_callback = create_episode_logging_callback(mock_logger)
        
        # Test that callbacks are callable
        assert callable(step_callback), "Step callback should be callable"
        assert callable(episode_callback), "Episode callback should be callable"
        
        # Test calling the callbacks
        step_data = {'step_num': 1, 'reward': 0.5}
        episode_data = {'episode_num': 1, 'total_steps': 10}
        
        # These should not raise exceptions
        step_callback(step_data)
        episode_callback(episode_data)
        
        # Verify the mock logger received the calls
        assert len(mock_logger.step_calls) == 1, "Step callback should have been called"
        assert len(mock_logger.episode_calls) == 1, "Episode callback should have been called"
    
    def test_direct_callback_functions(self):
        """Test the direct callback functions without logger."""
        # Test step visualization callback without logger (should not crash)
        step_data = {
            'step_num': 1,
            'reward': 0.5,
            'info': {'metrics': {'similarity': 0.8}}
        }
        
        # This should not raise an exception (just logs a debug message)
        jax_save_step_visualization(step_data, None)
        
        # Test episode summary callback without logger (should not crash)
        summary_data = {
            'episode_num': 1,
            'total_steps': 10,
            'total_reward': 5.0,
            'success': True
        }
        
        # This should not raise an exception (just logs a debug message)
        jax_save_episode_summary(summary_data, None)


class TestEnvironmentIntegration:
    """Test environment integration with JAX callbacks."""
    
    def test_environment_with_logging_enabled(self):
        """Test that environment works with logging enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal config for testing
            config = JaxArcConfig(
                storage=StorageConfig(base_output_dir=temp_dir),
                environment=EnvironmentConfig(debug_level="standard"),
                dataset=DatasetConfig(dataset_name="mini_arc")
            )
            
            # Create environment
            env = ArcEnvironment(config)
            
            # Verify logger was created
            assert env._logger is not None, "Environment should have logger when debug_level != 'off'"
            
            # Test reset
            key = jax.random.PRNGKey(42)
            state, obs = env.reset(key)
            assert state is not None, "Environment should reset successfully"
            assert obs is not None, "Environment should return observation"
            
            # Test step with logging (this will trigger JAX callbacks)
            action = create_point_action(operation=0, row=0, col=0)
            next_state, next_obs, reward, info = env.step(action)
            
            assert next_state is not None, "Environment should step successfully"
            assert isinstance(reward, (int, float, jnp.ndarray)), "Reward should be numeric"
            
            # Clean up
            env.close()
    
    def test_environment_with_logging_disabled(self):
        """Test that environment works with logging disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config with logging disabled
            config = JaxArcConfig(
                storage=StorageConfig(base_output_dir=temp_dir),
                environment=EnvironmentConfig(debug_level="off"),
                dataset=DatasetConfig(dataset_name="mini_arc")
            )
            
            # Create environment
            env = ArcEnvironment(config)
            
            # Verify logger was not created
            assert env._logger is None, "Environment should not have logger when debug_level == 'off'"
            
            # Test reset
            key = jax.random.PRNGKey(42)
            state, obs = env.reset(key)
            assert state is not None, "Environment should reset successfully without logging"
            
            # Test step without logging
            action = create_point_action(operation=0, row=0, col=0)
            next_state, next_obs, reward, info = env.step(action)
            
            assert next_state is not None, "Environment should step successfully without logging"
            
            # Clean up
            env.close()


class TestExperimentLoggerIntegration:
    """Test ExperimentLogger integration with JAX callbacks."""
    
    def test_experiment_logger_callback_integration(self):
        """Test that ExperimentLogger integrates correctly with JAX callbacks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test configuration
            config = JaxArcConfig(
                storage=StorageConfig(base_output_dir=temp_dir),
                environment=EnvironmentConfig(debug_level="standard"),
                dataset=DatasetConfig(dataset_name="mini_arc")
            )
            
            # Create ExperimentLogger
            logger = ExperimentLogger(config)
            
            # Test step logging callback
            step_data = {
                'step_num': 1,
                'reward': 0.5,
                'info': {'metrics': {'similarity': 0.8}}
            }
            
            # This should not raise an exception
            jax_save_step_visualization(step_data, logger)
            
            # Test episode summary callback
            summary_data = {
                'episode_num': 1,
                'total_steps': 10,
                'total_reward': 5.0,
                'success': True
            }
            
            # This should not raise an exception
            jax_save_episode_summary(summary_data, logger)
            
            # Clean up
            logger.close()


def run_all_tests():
    """Run all integration tests."""
    print("Running JAX callback integration tests...\n")
    
    test_classes = [
        TestJaxCallbackDataValidation(),
        TestJaxTransformationCompatibility(),
        TestCallbackFactoryFunctions(),
        TestEnvironmentIntegration(),
        TestExperimentLoggerIntegration()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"Running {class_name}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\nTest Results: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)