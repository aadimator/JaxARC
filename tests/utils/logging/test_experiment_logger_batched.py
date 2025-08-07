"""Unit tests for ExperimentLogger batched logging extensions.

This module tests the log_batch_step method, _aggregate_batch_metrics,
_sample_episodes_from_batch, and error isolation functionality.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc.utils.logging.experiment_logger import ExperimentLogger


class MockLoggingConfig:
    """Mock logging configuration for testing."""
    
    def __init__(self, **kwargs):
        self.batched_logging_enabled = kwargs.get('batched_logging_enabled', True)
        self.sampling_enabled = kwargs.get('sampling_enabled', True)
        self.log_frequency = kwargs.get('log_frequency', 10)
        self.sample_frequency = kwargs.get('sample_frequency', 50)
        self.num_samples = kwargs.get('num_samples', 3)
        
        # Aggregated metrics selection
        self.log_aggregated_rewards = kwargs.get('log_aggregated_rewards', True)
        self.log_aggregated_similarity = kwargs.get('log_aggregated_similarity', True)
        self.log_loss_metrics = kwargs.get('log_loss_metrics', True)
        self.log_gradient_norms = kwargs.get('log_gradient_norms', True)
        self.log_episode_lengths = kwargs.get('log_episode_lengths', True)
        self.log_success_rates = kwargs.get('log_success_rates', True)


class MockEnvironmentConfig:
    """Mock environment configuration."""
    
    def __init__(self, debug_level="standard"):
        self.debug_level = debug_level


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self, logging_config=None, environment_config=None):
        self.logging = logging_config if logging_config is not None else MockLoggingConfig()
        self.environment = environment_config if environment_config is not None else MockEnvironmentConfig()


class MockHandler:
    """Mock handler for testing."""
    
    def __init__(self, name="mock_handler", should_fail=False):
        self.name = name
        self.should_fail = should_fail
        self.log_aggregated_metrics_calls = []
        self.log_episode_summary_calls = []
    
    def log_aggregated_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Mock log_aggregated_metrics method."""
        if self.should_fail:
            raise Exception(f"Handler {self.name} failed")
        self.log_aggregated_metrics_calls.append((metrics, step))
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Mock log_episode_summary method."""
        if self.should_fail:
            raise Exception(f"Handler {self.name} failed")
        self.log_episode_summary_calls.append(summary_data)


def create_test_batch_data(batch_size: int = 5, update_step: int = 10) -> Dict[str, Any]:
    """Create test batch data for testing."""
    return {
        'update_step': update_step,
        'episode_returns': jnp.array([1.0, 2.0, 3.0, 4.0, 5.0][:batch_size]),
        'episode_lengths': jnp.array([10, 20, 30, 40, 50][:batch_size]),
        'similarity_scores': jnp.array([0.1, 0.2, 0.3, 0.4, 0.5][:batch_size]),
        'policy_loss': 0.5,
        'value_loss': 0.3,
        'gradient_norm': 1.2,
        'success_mask': jnp.array([True, False, True, False, True][:batch_size]),
        'task_ids': [f'task_{i}' for i in range(batch_size)],
        'initial_states': [f'initial_state_{i}' for i in range(batch_size)],
        'final_states': [f'final_state_{i}' for i in range(batch_size)]
    }


class TestExperimentLoggerBatchedLogging:
    """Test cases for ExperimentLogger batched logging functionality."""
    
    def test_log_batch_step_disabled(self):
        """Test log_batch_step when batched logging is disabled."""
        config = MockConfig(logging_config=MockLoggingConfig(batched_logging_enabled=False))
        logger = ExperimentLogger(config)
        
        # Add mock handler
        mock_handler = MockHandler()
        logger.handlers['mock'] = mock_handler
        
        batch_data = create_test_batch_data()
        
        # Should not call handler methods when disabled
        logger.log_batch_step(batch_data)
        
        assert len(mock_handler.log_aggregated_metrics_calls) == 0
        assert len(mock_handler.log_episode_summary_calls) == 0
    
    def test_log_batch_step_frequency_control(self):
        """Test that log_batch_step respects frequency settings."""
        config = MockConfig(logging_config=MockLoggingConfig(
            batched_logging_enabled=True,
            sampling_enabled=True,
            log_frequency=10,
            sample_frequency=50
        ))
        logger = ExperimentLogger(config)
        
        # Add mock handler
        mock_handler = MockHandler()
        logger.handlers['mock'] = mock_handler
        
        # Test aggregation frequency (should log at step 10, 20, 30...)
        batch_data = create_test_batch_data(update_step=5)
        logger.log_batch_step(batch_data)
        assert len(mock_handler.log_aggregated_metrics_calls) == 0  # Not at frequency
        
        batch_data = create_test_batch_data(update_step=10)
        logger.log_batch_step(batch_data)
        assert len(mock_handler.log_aggregated_metrics_calls) == 1  # At frequency
        
        # Test sampling frequency (should log at step 50, 100, 150...)
        batch_data = create_test_batch_data(update_step=25)
        logger.log_batch_step(batch_data)
        assert len(mock_handler.log_episode_summary_calls) == 0  # Not at frequency
        
        batch_data = create_test_batch_data(update_step=50)
        logger.log_batch_step(batch_data)
        assert len(mock_handler.log_episode_summary_calls) > 0  # At frequency
    
    def test_log_batch_step_no_logging_config(self):
        """Test log_batch_step when logging config is missing."""
        # Create config without logging attribute
        class ConfigWithoutLogging:
            def __init__(self):
                self.environment = MockEnvironmentConfig()
        
        config = ConfigWithoutLogging()
        logger = ExperimentLogger(config)
        
        batch_data = create_test_batch_data()
        
        # Should not crash, just return early
        logger.log_batch_step(batch_data)
    
    def test_aggregate_batch_metrics_basic(self):
        """Test basic batch metrics aggregation."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        batch_data = create_test_batch_data(batch_size=3)
        metrics = logger._aggregate_batch_metrics(batch_data)
        
        # Check reward aggregation
        assert 'reward_mean' in metrics
        assert 'reward_std' in metrics
        assert 'reward_max' in metrics
        assert 'reward_min' in metrics
        assert metrics['reward_mean'] == 2.0  # Mean of [1, 2, 3]
        assert metrics['reward_max'] == 3.0
        assert metrics['reward_min'] == 1.0
        
        # Check similarity aggregation
        assert 'similarity_mean' in metrics
        assert abs(metrics['similarity_mean'] - 0.2) < 1e-6  # Mean of [0.1, 0.2, 0.3]
        
        # Check episode length aggregation
        assert 'episode_length_mean' in metrics
        assert metrics['episode_length_mean'] == 20.0  # Mean of [10, 20, 30]
        
        # Check success rate
        assert 'success_rate' in metrics
        assert abs(metrics['success_rate'] - 2.0/3.0) < 1e-6  # 2 True out of 3
        
        # Check scalar metrics
        assert 'policy_loss' in metrics
        assert metrics['policy_loss'] == 0.5
        assert 'value_loss' in metrics
        assert metrics['value_loss'] == 0.3
        assert 'gradient_norm' in metrics
        assert metrics['gradient_norm'] == 1.2
    
    def test_aggregate_batch_metrics_selective_logging(self):
        """Test aggregation with selective metric logging."""
        config = MockConfig(logging_config=MockLoggingConfig(
            log_aggregated_rewards=False,
            log_aggregated_similarity=True,
            log_loss_metrics=False,
            log_gradient_norms=True,
            log_episode_lengths=False,
            log_success_rates=True
        ))
        logger = ExperimentLogger(config)
        
        batch_data = create_test_batch_data()
        metrics = logger._aggregate_batch_metrics(batch_data)
        
        # Should not include disabled metrics
        assert 'reward_mean' not in metrics
        assert 'episode_length_mean' not in metrics
        assert 'policy_loss' not in metrics
        assert 'value_loss' not in metrics
        
        # Should include enabled metrics
        assert 'similarity_mean' in metrics
        assert 'gradient_norm' in metrics
        assert 'success_rate' in metrics
    
    def test_aggregate_batch_metrics_missing_data(self):
        """Test aggregation with missing data fields."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        # Create batch data with missing fields
        batch_data = {
            'update_step': 10,
            'episode_returns': jnp.array([1.0, 2.0, 3.0]),
            # Missing similarity_scores, episode_lengths, success_mask
            'policy_loss': 0.5
        }
        
        metrics = logger._aggregate_batch_metrics(batch_data)
        
        # Should include available metrics
        assert 'reward_mean' in metrics
        assert 'policy_loss' in metrics
        
        # Should not include missing metrics
        assert 'similarity_mean' not in metrics
        assert 'episode_length_mean' not in metrics
        assert 'success_rate' not in metrics
    
    def test_aggregate_batch_metrics_unknown_fields(self):
        """Test aggregation with unknown/custom fields."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        batch_data = create_test_batch_data()
        # Add unknown fields
        batch_data['custom_scalar'] = 42.0
        batch_data['custom_array'] = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        batch_data['empty_array'] = jnp.array([])
        batch_data['single_element'] = jnp.array([7.0])
        batch_data['metadata'] = {'info': 'test'}  # Non-numeric
        
        metrics = logger._aggregate_batch_metrics(batch_data)
        
        # Should include custom scalar
        assert 'custom_scalar' in metrics
        assert metrics['custom_scalar'] == 42.0
        
        # Should aggregate custom array
        assert 'custom_array_mean' in metrics
        assert 'custom_array_std' in metrics
        assert 'custom_array_max' in metrics
        assert 'custom_array_min' in metrics
        
        # Should handle single element as scalar
        assert 'single_element' in metrics
        assert metrics['single_element'] == 7.0
        
        # Should skip empty arrays and non-numeric data
        assert 'empty_array' not in metrics
        assert 'metadata' not in metrics
    
    def test_sample_episodes_from_batch_basic(self):
        """Test basic episode sampling from batch."""
        config = MockConfig(logging_config=MockLoggingConfig(num_samples=2))
        logger = ExperimentLogger(config)
        
        batch_data = create_test_batch_data(batch_size=5, update_step=100)
        sampled_episodes = logger._sample_episodes_from_batch(batch_data)
        
        assert len(sampled_episodes) == 2
        
        # Check episode structure
        episode = sampled_episodes[0]
        assert 'episode_num' in episode
        # Episode num is now unique: update_step * 10000 + environment_id
        assert episode['episode_num'] >= 1000000  # Should be 100 * 10000 + env_id
        assert 'total_reward' in episode
        assert 'total_steps' in episode
        assert 'final_similarity' in episode
        assert 'success' in episode
        assert 'environment_id' in episode
        assert 'task_id' in episode
        assert 'initial_state' in episode
        assert 'final_state' in episode
    
    def test_sample_episodes_from_batch_more_samples_than_batch(self):
        """Test sampling when num_samples >= batch_size."""
        config = MockConfig(logging_config=MockLoggingConfig(num_samples=10))
        logger = ExperimentLogger(config)
        
        batch_data = create_test_batch_data(batch_size=3)
        sampled_episodes = logger._sample_episodes_from_batch(batch_data)
        
        # Should return all episodes when num_samples >= batch_size
        assert len(sampled_episodes) == 3
    
    def test_sample_episodes_from_batch_deterministic(self):
        """Test that sampling is deterministic based on update_step."""
        config = MockConfig(logging_config=MockLoggingConfig(num_samples=2))
        logger = ExperimentLogger(config)
        
        batch_data = create_test_batch_data(batch_size=5, update_step=42)
        
        # Sample twice with same update_step
        sampled1 = logger._sample_episodes_from_batch(batch_data)
        sampled2 = logger._sample_episodes_from_batch(batch_data)
        
        # Should be identical
        assert len(sampled1) == len(sampled2)
        for ep1, ep2 in zip(sampled1, sampled2):
            assert ep1['environment_id'] == ep2['environment_id']
    
    def test_sample_episodes_from_batch_missing_optional_data(self):
        """Test sampling with missing optional data fields."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        # Create batch data without optional fields
        batch_data = {
            'update_step': 10,
            'episode_returns': jnp.array([1.0, 2.0, 3.0]),
            # Missing episode_lengths, similarity_scores, success_mask, etc.
        }
        
        sampled_episodes = logger._sample_episodes_from_batch(batch_data)
        
        assert len(sampled_episodes) == 3  # All episodes since num_samples >= batch_size
        
        # Check default values for missing fields
        episode = sampled_episodes[0]
        assert episode['total_steps'] == 0  # Default
        assert episode['final_similarity'] == 0.0  # Default
        assert episode['success'] is False  # Default
        assert episode['initial_state'] is None  # Default
        assert episode['final_state'] is None  # Default
    
    def test_error_isolation_aggregation(self):
        """Test error isolation when aggregation fails."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        # Add mock handler
        mock_handler = MockHandler()
        logger.handlers['mock'] = mock_handler
        
        # Create invalid batch data that will cause aggregation to fail
        batch_data = {
            'update_step': 10,
            'episode_returns': "invalid_data"  # Should cause error
        }
        
        # Should not crash, should handle error gracefully
        logger.log_batch_step(batch_data)
        
        # Handler should not be called due to aggregation error
        assert len(mock_handler.log_aggregated_metrics_calls) == 0
    
    def test_error_isolation_sampling(self):
        """Test error isolation when sampling fails."""
        config = MockConfig(logging_config=MockLoggingConfig(
            batched_logging_enabled=False,  # Disable aggregation
            sampling_enabled=True
        ))
        logger = ExperimentLogger(config)
        
        # Add mock handler
        mock_handler = MockHandler()
        logger.handlers['mock'] = mock_handler
        
        # Create invalid batch data that will cause sampling to fail
        batch_data = {
            'update_step': 50,  # At sample frequency
            'episode_returns': "invalid_data"  # Should cause error
        }
        
        # Should not crash, should handle error gracefully
        logger.log_batch_step(batch_data)
        
        # Handler should not be called due to sampling error
        assert len(mock_handler.log_episode_summary_calls) == 0
    
    def test_error_isolation_handler_failures(self):
        """Test error isolation when handlers fail."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        # Add handlers - one that works, one that fails
        good_handler = MockHandler("good")
        bad_handler = MockHandler("bad", should_fail=True)
        logger.handlers['good'] = good_handler
        logger.handlers['bad'] = bad_handler
        
        batch_data = create_test_batch_data(update_step=10)  # At aggregation frequency
        
        # Should not crash despite handler failure
        logger.log_batch_step(batch_data)
        
        # Good handler should have been called
        assert len(good_handler.log_aggregated_metrics_calls) == 1
        
        # Bad handler should have failed but not crashed the system
        assert len(bad_handler.log_aggregated_metrics_calls) == 0
    
    def test_handler_without_batched_methods(self):
        """Test handling of handlers that don't support batched logging."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        # Add handler without log_aggregated_metrics method
        class LegacyHandler:
            def __init__(self):
                self.calls = []
            
            def log_episode_summary(self, data):
                self.calls.append(data)
        
        legacy_handler = LegacyHandler()
        logger.handlers['legacy'] = legacy_handler
        
        batch_data = create_test_batch_data(update_step=10)
        
        # Should not crash when handler doesn't have batched methods
        logger.log_batch_step(batch_data)
        
        # Legacy handler should not be called for aggregated metrics
        # but should be called for episode summaries if sampling is enabled
        batch_data_sampling = create_test_batch_data(update_step=50)  # At sample frequency
        logger.log_batch_step(batch_data_sampling)
        
        # Should have called log_episode_summary for sampled episodes
        assert len(legacy_handler.calls) > 0
    
    def test_integration_with_existing_methods(self):
        """Test that batched logging integrates properly with existing methods."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        # Add mock handler that supports both old and new methods
        class FullHandler:
            def __init__(self):
                self.aggregated_calls = []
                self.episode_calls = []
                self.step_calls = []
            
            def log_aggregated_metrics(self, metrics, step):
                self.aggregated_calls.append((metrics, step))
            
            def log_episode_summary(self, data):
                self.episode_calls.append(data)
            
            def log_step(self, data):
                self.step_calls.append(data)
        
        full_handler = FullHandler()
        logger.handlers['full'] = full_handler
        
        # Test batched logging
        batch_data = create_test_batch_data(update_step=50)  # At both frequencies
        logger.log_batch_step(batch_data)
        
        # Should call both aggregated metrics and episode summary
        assert len(full_handler.aggregated_calls) == 1
        assert len(full_handler.episode_calls) > 0  # Sampled episodes
        
        # Test regular logging still works
        step_data = {'step_num': 1, 'reward': 0.5}
        logger.log_step(step_data)
        
        assert len(full_handler.step_calls) == 1


class TestBatchMetricsAggregationEdgeCases:
    """Test edge cases for batch metrics aggregation."""
    
    def test_aggregation_with_nan_values(self):
        """Test aggregation handles NaN values gracefully."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        batch_data = {
            'update_step': 10,
            'episode_returns': jnp.array([1.0, float('nan'), 3.0]),
            'policy_loss': float('nan')
        }
        
        # Should not crash with NaN values
        metrics = logger._aggregate_batch_metrics(batch_data)
        
        # Results may contain NaN, but should not crash
        assert 'reward_mean' in metrics
        assert 'policy_loss' in metrics
    
    def test_aggregation_with_inf_values(self):
        """Test aggregation handles infinite values gracefully."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        batch_data = {
            'update_step': 10,
            'episode_returns': jnp.array([1.0, float('inf'), 3.0]),
            'policy_loss': float('-inf')
        }
        
        # Should not crash with infinite values
        metrics = logger._aggregate_batch_metrics(batch_data)
        
        assert 'reward_mean' in metrics
        assert 'policy_loss' in metrics
    
    def test_aggregation_with_zero_batch_size(self):
        """Test aggregation with empty arrays."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        batch_data = {
            'update_step': 10,
            'episode_returns': jnp.array([]),
            'policy_loss': 0.5
        }
        
        # Should handle empty arrays gracefully
        metrics = logger._aggregate_batch_metrics(batch_data)
        
        # Should skip empty arrays but include scalars
        assert 'reward_mean' not in metrics
        assert 'policy_loss' in metrics
    
    def test_sampling_with_zero_batch_size(self):
        """Test sampling with empty batch."""
        config = MockConfig()
        logger = ExperimentLogger(config)
        
        batch_data = {
            'update_step': 10,
            'episode_returns': jnp.array([])
        }
        
        # Should handle empty batch gracefully
        sampled_episodes = logger._sample_episodes_from_batch(batch_data)
        
        assert len(sampled_episodes) == 0