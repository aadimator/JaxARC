"""Integration tests for end-to-end batched logging.

This module tests the complete batched logging pipeline with mock training data,
verifies file outputs, wandb integration, console display, configuration-driven
behavior changes, and performance with various batch sizes.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from jaxarc.utils.logging.experiment_logger import ExperimentLogger


class MockLoggingConfig:
    """Mock logging configuration for integration testing."""
    
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


class MockStorageConfig:
    """Mock storage configuration."""
    
    def __init__(self, base_output_dir="test_outputs", logs_dir="logs"):
        self.base_output_dir = base_output_dir
        self.logs_dir = logs_dir


class MockWandbConfig:
    """Mock wandb configuration."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.project_name = 'test-project'
        self.entity = None
        self.tags = ['test']
        self.notes = 'Integration test'
        self.group = None
        self.job_type = 'test'
        self.offline_mode = False
        self.save_code = True


class MockConfig:
    """Mock configuration for integration testing."""
    
    def __init__(self, logging_config=None, environment_config=None, storage_config=None, wandb_config=None):
        self.logging = logging_config if logging_config is not None else MockLoggingConfig()
        self.environment = environment_config if environment_config is not None else MockEnvironmentConfig()
        self.storage = storage_config if storage_config is not None else MockStorageConfig()
        if wandb_config is not None:
            self.wandb = wandb_config


def create_realistic_batch_data(batch_size: int = 10, update_step: int = 100) -> Dict[str, Any]:
    """Create realistic batch data for integration testing."""
    np.random.seed(42)  # For reproducible tests
    
    return {
        'update_step': update_step,
        'episode_returns': jnp.array(np.random.normal(2.0, 1.0, batch_size)),
        'episode_lengths': jnp.array(np.random.randint(10, 100, batch_size)),
        'similarity_scores': jnp.array(np.random.uniform(0.0, 1.0, batch_size)),
        'policy_loss': np.random.uniform(0.1, 1.0),
        'value_loss': np.random.uniform(0.1, 1.0),
        'gradient_norm': np.random.uniform(0.5, 2.0),
        'success_mask': jnp.array(np.random.choice([True, False], batch_size)),
        'task_ids': [f'task_{i:03d}' for i in range(batch_size)],
        'initial_states': [f'initial_state_{i}' for i in range(batch_size)],
        'final_states': [f'final_state_{i}' for i in range(batch_size)],
        # Additional custom metrics
        'entropy': np.random.uniform(0.1, 2.0),
        'explained_variance': np.random.uniform(0.0, 1.0),
        'learning_rate': 0.001,
        'custom_array_metric': jnp.array(np.random.normal(0.0, 1.0, batch_size))
    }


class TestEndToEndBatchedLogging:
    """Integration tests for complete batched logging pipeline."""
    
    def test_complete_pipeline_with_file_output(self):
        """Test complete batched logging pipeline with file output verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    sampling_enabled=True,
                    log_frequency=10,
                    sample_frequency=50,
                    num_samples=2
                ),
                storage_config=storage_config
            )
            
            logger = ExperimentLogger(config)
            
            # Simulate training loop with batched logging
            batch_data_1 = create_realistic_batch_data(batch_size=5, update_step=10)  # At aggregation frequency
            batch_data_2 = create_realistic_batch_data(batch_size=5, update_step=20)  # At aggregation frequency
            batch_data_3 = create_realistic_batch_data(batch_size=5, update_step=50)  # At both frequencies
            
            logger.log_batch_step(batch_data_1)
            logger.log_batch_step(batch_data_2)
            logger.log_batch_step(batch_data_3)
            
            # Verify file outputs
            logs_dir = Path(temp_dir) / "logs"
            
            # Check batch metrics file
            batch_metrics_file = logs_dir / "batch_metrics.jsonl"
            assert batch_metrics_file.exists()
            
            with open(batch_metrics_file, 'r') as f:
                lines = f.readlines()
            
            # Should have 3 entries (steps 10, 20, 50)
            assert len(lines) == 3
            
            # Verify batch metrics content
            for i, line in enumerate(lines):
                entry = json.loads(line)
                assert 'timestamp' in entry
                assert 'step' in entry
                assert 'metrics' in entry
                
                metrics = entry['metrics']
                assert 'reward_mean' in metrics
                assert 'reward_std' in metrics
                assert 'policy_loss' in metrics
                assert 'success_rate' in metrics
            
            # Check episode summary files (from sampling at step 50)
            episode_files = list(logs_dir.glob("episode_*.json"))
            assert len(episode_files) == 2  # num_samples = 2
            
            # Verify episode summary content
            for episode_file in episode_files:
                with open(episode_file, 'r') as f:
                    episode_data = json.load(f)
                
                assert 'episode_num' in episode_data
                assert 'total_reward' in episode_data
                assert 'environment_id' in episode_data
                assert 'task_id' in episode_data
    
    def test_complete_pipeline_with_wandb_integration(self):
        """Test complete batched logging pipeline with wandb integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            wandb_config = MockWandbConfig(enabled=True)
            config = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    log_frequency=10
                ),
                storage_config=storage_config,
                wandb_config=wandb_config
            )
            
            # Mock wandb at the handler level instead of module level
            with patch('jaxarc.utils.logging.wandb_handler.logger') as mock_logger:
                logger = ExperimentLogger(config)
                
                # Mock the wandb handler's run object
                if 'wandb' in logger.handlers:
                    mock_run = MagicMock()
                    logger.handlers['wandb'].run = mock_run
                    
                    # Simulate training steps
                    batch_data = create_realistic_batch_data(batch_size=8, update_step=10)
                    logger.log_batch_step(batch_data)
                    
                    # Check that aggregated metrics were logged to wandb
                    assert mock_run.log.call_count >= 1
                    
                    # Verify batch prefix in wandb metrics
                    logged_calls = mock_run.log.call_args_list
                    batch_calls = [call for call in logged_calls 
                                  if call[0] and any(key.startswith('batch/') for key in call[0][0].keys())]
                    assert len(batch_calls) >= 1
                    
                    batch_call = batch_calls[0]
                    logged_metrics = batch_call[0][0]
                    assert 'batch/reward_mean' in logged_metrics
                    assert 'batch/policy_loss' in logged_metrics
                    assert batch_call[1]['step'] == 10
                else:
                    # If wandb handler wasn't created, just verify no crash
                    batch_data = create_realistic_batch_data(batch_size=8, update_step=10)
                    logger.log_batch_step(batch_data)
                    # Test passes if no exception is raised
    
    def test_configuration_driven_behavior_changes(self):
        """Test that configuration changes affect logging behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            
            # Test 1: Disabled batched logging
            config_disabled = MockConfig(
                logging_config=MockLoggingConfig(batched_logging_enabled=False),
                storage_config=storage_config
            )
            
            logger_disabled = ExperimentLogger(config_disabled)
            batch_data = create_realistic_batch_data(update_step=10)
            logger_disabled.log_batch_step(batch_data)
            
            # Should not create batch metrics file
            batch_metrics_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            assert not batch_metrics_file.exists()
            
            # Test 2: Selective metric logging
            config_selective = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    log_frequency=10,
                    log_aggregated_rewards=True,
                    log_aggregated_similarity=False,
                    log_loss_metrics=False,
                    log_success_rates=True
                ),
                storage_config=storage_config
            )
            
            logger_selective = ExperimentLogger(config_selective)
            logger_selective.log_batch_step(batch_data)
            
            # Check selective logging
            with open(batch_metrics_file, 'r') as f:
                entry = json.loads(f.readline())
            
            metrics = entry['metrics']
            assert 'reward_mean' in metrics  # Enabled
            assert 'success_rate' in metrics  # Enabled
            assert 'similarity_mean' not in metrics  # Disabled
            assert 'policy_loss' not in metrics  # Disabled
            
            # Test 3: Different frequencies
            config_freq = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    sampling_enabled=True,
                    log_frequency=25,  # Different frequency
                    sample_frequency=100,  # Different frequency
                    num_samples=1
                ),
                storage_config=storage_config
            )
            
            logger_freq = ExperimentLogger(config_freq)
            
            # Clear previous file
            if batch_metrics_file.exists():
                batch_metrics_file.unlink()
            
            # Test frequency behavior
            logger_freq.log_batch_step(create_realistic_batch_data(update_step=10))  # Not at frequency
            logger_freq.log_batch_step(create_realistic_batch_data(update_step=25))  # At aggregation frequency
            logger_freq.log_batch_step(create_realistic_batch_data(update_step=100))  # At both frequencies
            
            # Should have 2 aggregation entries (steps 25, 100)
            with open(batch_metrics_file, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 2
            
            # Should have 1 episode summary (step 100)
            episode_files = list((Path(temp_dir) / "logs").glob("episode_*.json"))
            assert len(episode_files) == 1
    
    def test_performance_with_various_batch_sizes(self):
        """Test performance and correctness with various batch sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    sampling_enabled=True,
                    log_frequency=10,
                    sample_frequency=10,
                    num_samples=5
                ),
                storage_config=storage_config
            )
            
            logger = ExperimentLogger(config)
            
            # Test different batch sizes
            batch_sizes = [1, 10, 100, 1000]
            
            for i, batch_size in enumerate(batch_sizes):
                update_step = (i + 1) * 10  # Steps 10, 20, 30, 40
                
                start_time = time.time()
                batch_data = create_realistic_batch_data(batch_size=batch_size, update_step=update_step)
                logger.log_batch_step(batch_data)
                end_time = time.time()
                
                # Performance should be reasonable (< 1 second for any batch size)
                processing_time = end_time - start_time
                assert processing_time < 1.0, f"Batch size {batch_size} took {processing_time:.3f}s"
            
            # Verify all batches were logged
            batch_metrics_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            with open(batch_metrics_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == len(batch_sizes)
            
            # Verify sampling worked correctly for different batch sizes
            episode_files = list((Path(temp_dir) / "logs").glob("episode_*.json"))
            
            # Should have sampled episodes from each batch
            # For batch_size < num_samples, should sample all episodes
            # For batch_size >= num_samples, should sample exactly num_samples
            expected_total_episodes = sum(min(batch_size, 5) for batch_size in batch_sizes)
            assert len(episode_files) == expected_total_episodes
    
    def test_error_resilience_and_recovery(self):
        """Test that the system is resilient to errors and recovers gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    sampling_enabled=True,
                    log_frequency=10,
                    sample_frequency=10
                ),
                storage_config=storage_config
            )
            
            logger = ExperimentLogger(config)
            
            # Add a handler that will fail
            class FailingHandler:
                def __init__(self, should_fail=True):
                    self.should_fail = should_fail
                    self.calls = []
                
                def log_aggregated_metrics(self, metrics, step):
                    if self.should_fail:
                        raise Exception("Handler failed")
                    self.calls.append(('aggregated', metrics, step))
                
                def log_episode_summary(self, data):
                    if self.should_fail:
                        raise Exception("Handler failed")
                    self.calls.append(('summary', data))
            
            failing_handler = FailingHandler(should_fail=True)
            working_handler = FailingHandler(should_fail=False)
            
            logger.handlers['failing'] = failing_handler
            logger.handlers['working'] = working_handler
            
            # Test with handler failures
            batch_data = create_realistic_batch_data(update_step=10)
            
            # Should not crash despite handler failures
            logger.log_batch_step(batch_data)
            
            # Working handler should have been called
            assert len(working_handler.calls) > 0
            
            # Failing handler should not have successful calls
            assert len(failing_handler.calls) == 0
            
            # File handler should still work (built-in error handling)
            batch_metrics_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            assert batch_metrics_file.exists()
    
    def test_memory_usage_with_large_batches(self):
        """Test memory usage remains reasonable with large batches."""
        import psutil
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    sampling_enabled=True,
                    log_frequency=10,
                    sample_frequency=10,
                    num_samples=10
                ),
                storage_config=storage_config
            )
            
            logger = ExperimentLogger(config)
            
            # Measure initial memory
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process large batch
            large_batch_data = create_realistic_batch_data(batch_size=10000, update_step=10)
            logger.log_batch_step(large_batch_data)
            
            # Measure memory after processing
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 100MB for this test)
            assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
    
    def test_concurrent_logging_safety(self):
        """Test that logging is safe with concurrent access patterns."""
        import threading
        import queue
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    log_frequency=1,  # Log every step
                    sample_frequency=1
                ),
                storage_config=storage_config
            )
            
            logger = ExperimentLogger(config)
            
            # Queue to collect any exceptions from threads
            exception_queue = queue.Queue()
            
            def logging_worker(worker_id, num_steps):
                try:
                    for step in range(num_steps):
                        batch_data = create_realistic_batch_data(
                            batch_size=5, 
                            update_step=worker_id * 100 + step + 1
                        )
                        logger.log_batch_step(batch_data)
                except Exception as e:
                    exception_queue.put(e)
            
            # Start multiple threads
            threads = []
            num_workers = 3
            steps_per_worker = 5
            
            for worker_id in range(num_workers):
                thread = threading.Thread(
                    target=logging_worker, 
                    args=(worker_id, steps_per_worker)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check for exceptions
            exceptions = []
            while not exception_queue.empty():
                exceptions.append(exception_queue.get())
            
            assert len(exceptions) == 0, f"Concurrent logging raised exceptions: {exceptions}"
            
            # Verify that all steps were logged
            batch_metrics_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            with open(batch_metrics_file, 'r') as f:
                lines = f.readlines()
            
            # Should have entries from all workers
            assert len(lines) == num_workers * steps_per_worker
    
    def test_integration_with_existing_logging_methods(self):
        """Test that batched logging integrates seamlessly with existing methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    sampling_enabled=True,
                    log_frequency=10,
                    sample_frequency=10
                ),
                storage_config=storage_config
            )
            
            logger = ExperimentLogger(config)
            
            # Use traditional logging methods
            task_data = {
                'task_id': 'integration_test_task',
                'episode_num': 1,
                'num_train_pairs': 3,
                'num_test_pairs': 1
            }
            logger.log_task_start(task_data)
            
            step_data = {
                'step_num': 1,
                'before_state': None,
                'after_state': None,
                'action': {'operation': 'fill', 'color': 1},
                'reward': 1.0,
                'info': {'metrics': {'similarity': 0.8}}
            }
            logger.log_step(step_data)
            
            # Use batched logging
            batch_data = create_realistic_batch_data(update_step=10)
            logger.log_batch_step(batch_data)
            
            # Use traditional episode summary
            summary_data = {
                'episode_num': 1,
                'total_steps': 1,
                'total_reward': 1.0,
                'final_similarity': 0.8,
                'success': True,
                'task_id': 'integration_test_task'
            }
            logger.log_episode_summary(summary_data)
            
            # Verify all outputs exist
            logs_dir = Path(temp_dir) / "logs"
            
            # Traditional episode files
            episode_files = list(logs_dir.glob("episode_*.json"))
            assert len(episode_files) >= 1  # At least one from traditional logging
            
            # Batch metrics file
            batch_metrics_file = logs_dir / "batch_metrics.jsonl"
            assert batch_metrics_file.exists()
            
            # Verify content doesn't interfere
            with open(batch_metrics_file, 'r') as f:
                batch_entry = json.loads(f.readline())
            
            assert 'reward_mean' in batch_entry['metrics']
            assert batch_entry['step'] == 10
            
            # Check traditional episode file
            with open(episode_files[0], 'r') as f:
                episode_data = json.load(f)
            
            assert episode_data['episode_num'] == 1
            assert episode_data['task_id'] == 'integration_test_task'


class TestBatchedLoggingConfigurationScenarios:
    """Test various configuration scenarios for batched logging."""
    
    def test_minimal_configuration(self):
        """Test batched logging with minimal configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    # Use all defaults for other settings
                ),
                storage_config=storage_config
            )
            
            logger = ExperimentLogger(config)
            batch_data = create_realistic_batch_data(update_step=10)
            logger.log_batch_step(batch_data)
            
            # Should work with defaults
            batch_metrics_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            assert batch_metrics_file.exists()
    
    def test_maximum_configuration(self):
        """Test batched logging with all features enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(
                logging_config=MockLoggingConfig(
                    batched_logging_enabled=True,
                    sampling_enabled=True,
                    log_frequency=1,  # Log every step
                    sample_frequency=1,  # Sample every step
                    num_samples=10,
                    # Enable all metrics
                    log_aggregated_rewards=True,
                    log_aggregated_similarity=True,
                    log_loss_metrics=True,
                    log_gradient_norms=True,
                    log_episode_lengths=True,
                    log_success_rates=True
                ),
                storage_config=storage_config
            )
            
            logger = ExperimentLogger(config)
            
            # Process multiple steps
            for step in range(1, 6):
                batch_data = create_realistic_batch_data(batch_size=5, update_step=step)
                logger.log_batch_step(batch_data)
            
            # Should have comprehensive logging
            logs_dir = Path(temp_dir) / "logs"
            
            # Batch metrics for every step
            batch_metrics_file = logs_dir / "batch_metrics.jsonl"
            with open(batch_metrics_file, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 5
            
            # Episode samples for every step
            episode_files = list(logs_dir.glob("episode_*.json"))
            assert len(episode_files) == 5 * 5  # 5 steps * 5 episodes per step
    
    def test_debug_level_integration(self):
        """Test that batched logging respects debug level settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            
            # Test with debug level "off"
            config_off = MockConfig(
                logging_config=MockLoggingConfig(batched_logging_enabled=True),
                environment_config=MockEnvironmentConfig(debug_level="off"),
                storage_config=storage_config
            )
            
            logger_off = ExperimentLogger(config_off)
            
            # Should have minimal handlers due to debug level
            assert len(logger_off.handlers) == 0  # All handlers disabled
            
            # Batched logging should still work (no crash)
            batch_data = create_realistic_batch_data(update_step=10)
            logger_off.log_batch_step(batch_data)
            
            # Test with debug level "research"
            config_research = MockConfig(
                logging_config=MockLoggingConfig(batched_logging_enabled=True),
                environment_config=MockEnvironmentConfig(debug_level="research"),
                storage_config=storage_config
            )
            
            logger_research = ExperimentLogger(config_research)
            
            # Should have more handlers
            assert len(logger_research.handlers) > 0
            
            logger_research.log_batch_step(batch_data)
            
            # Should create output files
            batch_metrics_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            assert batch_metrics_file.exists()