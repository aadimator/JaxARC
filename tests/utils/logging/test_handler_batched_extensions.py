"""Unit tests for handler batched logging extensions.

This module tests the log_aggregated_metrics method for each handler type,
integration with existing handler methods, error handling, and handler-specific
functionality.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from jaxarc.utils.logging.file_handler import FileHandler
from jaxarc.utils.logging.wandb_handler import WandbHandler
from jaxarc.utils.logging.rich_handler import RichHandler


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else MockStorageConfig()


class MockStorageConfig:
    """Mock storage configuration."""
    def __init__(self, base_output_dir="test_outputs", logs_dir="logs"):
        self.base_output_dir = base_output_dir
        self.logs_dir = logs_dir


class MockWandbConfig:
    """Mock wandb configuration for testing."""
    def __init__(self, **kwargs):
        self.enabled = kwargs.get('enabled', True)
        self.project_name = kwargs.get('project_name', 'test-project')
        self.entity = kwargs.get('entity', None)
        self.tags = kwargs.get('tags', ['test'])
        self.notes = kwargs.get('notes', 'Test run')
        self.group = kwargs.get('group', None)
        self.job_type = kwargs.get('job_type', 'test')
        self.offline_mode = kwargs.get('offline_mode', False)
        self.save_code = kwargs.get('save_code', True)


def create_test_metrics() -> Dict[str, float]:
    """Create test aggregated metrics."""
    return {
        'reward_mean': 2.5,
        'reward_std': 1.2,
        'reward_max': 5.0,
        'reward_min': 0.5,
        'similarity_mean': 0.75,
        'similarity_std': 0.15,
        'similarity_max': 0.95,
        'similarity_min': 0.45,
        'episode_length_mean': 25.0,
        'episode_length_std': 8.5,
        'episode_length_max': 50,
        'episode_length_min': 10,
        'policy_loss': 0.35,
        'value_loss': 0.28,
        'gradient_norm': 1.8,
        'success_rate': 0.6
    }


class TestFileHandlerBatchedExtensions:
    """Test cases for FileHandler batched logging extensions."""
    
    def test_log_aggregated_metrics_basic(self):
        """Test basic aggregated metrics logging to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            metrics = create_test_metrics()
            step = 100
            
            # Log aggregated metrics
            handler.log_aggregated_metrics(metrics, step)
            
            # Check that batch metrics file was created
            batch_metrics_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            assert batch_metrics_file.exists()
            
            # Check file content
            with open(batch_metrics_file, 'r') as f:
                line = f.readline().strip()
                log_entry = json.loads(line)
            
            assert 'timestamp' in log_entry
            assert log_entry['step'] == 100
            assert 'metrics' in log_entry
            assert log_entry['metrics']['reward_mean'] == 2.5
            assert log_entry['metrics']['policy_loss'] == 0.35
    
    def test_log_aggregated_metrics_multiple_calls(self):
        """Test multiple calls to log_aggregated_metrics append to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            # Log multiple batches
            metrics1 = {'reward_mean': 1.0, 'step': 10}
            metrics2 = {'reward_mean': 2.0, 'step': 20}
            
            handler.log_aggregated_metrics(metrics1, 10)
            handler.log_aggregated_metrics(metrics2, 20)
            
            # Check that both entries are in the file
            batch_metrics_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            with open(batch_metrics_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            
            entry1 = json.loads(lines[0])
            entry2 = json.loads(lines[1])
            
            assert entry1['step'] == 10
            assert entry1['metrics']['reward_mean'] == 1.0
            assert entry2['step'] == 20
            assert entry2['metrics']['reward_mean'] == 2.0
    
    def test_log_aggregated_metrics_creates_directory(self):
        """Test that log_aggregated_metrics creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a non-existent subdirectory
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="nonexistent/logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            metrics = {'reward_mean': 1.0}
            
            # Should create the directory
            handler.log_aggregated_metrics(metrics, 10)
            
            # Check that directory and file were created
            expected_dir = Path(temp_dir) / "nonexistent" / "logs"
            batch_metrics_file = expected_dir / "batch_metrics.jsonl"
            
            assert expected_dir.exists()
            assert batch_metrics_file.exists()
    
    def test_log_aggregated_metrics_error_handling(self):
        """Test error handling in log_aggregated_metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid handler first
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            handler = FileHandler(config)
            
            # Now make the output directory read-only to trigger write error
            output_dir = Path(temp_dir) / "logs"
            output_dir.chmod(0o444)  # Read-only
            
            metrics = {'reward_mean': 1.0}
            
            # Should not crash, should handle error gracefully
            handler.log_aggregated_metrics(metrics, 10)
            # No assertion needed - just checking it doesn't crash
            
            # Restore permissions for cleanup
            output_dir.chmod(0o755)
    
    def test_log_aggregated_metrics_with_complex_values(self):
        """Test logging with complex metric values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            # Include various types of values
            metrics = {
                'float_metric': 3.14159,
                'int_metric': 42,
                'zero_metric': 0.0,
                'negative_metric': -1.5,
                'large_metric': 1e6,
                'small_metric': 1e-6
            }
            
            handler.log_aggregated_metrics(metrics, 50)
            
            # Check that all values are serialized correctly
            batch_metrics_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            with open(batch_metrics_file, 'r') as f:
                log_entry = json.loads(f.readline())
            
            assert log_entry['metrics']['float_metric'] == 3.14159
            assert log_entry['metrics']['int_metric'] == 42
            assert log_entry['metrics']['zero_metric'] == 0.0
            assert log_entry['metrics']['negative_metric'] == -1.5
    
    def test_integration_with_existing_methods(self):
        """Test that batched logging integrates with existing FileHandler methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            # Use existing methods
            task_data = {'task_id': 'test_task', 'episode_num': 1}
            handler.log_task_start(task_data)
            
            step_data = {
                'step_num': 1,
                'before_state': None,
                'after_state': None,
                'action': {'operation': 'test'},
                'reward': 1.0,
                'info': {'metrics': {'similarity': 0.8}}
            }
            handler.log_step(step_data)
            
            # Use new batched method
            metrics = {'reward_mean': 2.0}
            handler.log_aggregated_metrics(metrics, 10)
            
            # Both should work without interference
            summary_data = {
                'episode_num': 1,
                'total_steps': 1,
                'total_reward': 1.0,
                'final_similarity': 0.8,
                'success': True
            }
            handler.log_episode_summary(summary_data)
            
            # Check that both regular episode files and batch metrics file exist
            logs_dir = Path(temp_dir) / "logs"
            episode_files = list(logs_dir.glob("episode_*.json"))
            batch_metrics_file = logs_dir / "batch_metrics.jsonl"
            
            assert len(episode_files) == 1
            assert batch_metrics_file.exists()


class TestWandbHandlerBatchedExtensions:
    """Test cases for WandbHandler batched logging extensions."""
    
    def test_log_aggregated_metrics_basic(self):
        """Test basic aggregated metrics logging to wandb."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        metrics = create_test_metrics()
        step = 100
        
        handler.log_aggregated_metrics(metrics, step)
        
        # Check that wandb.log was called with batch/ prefixed metrics
        mock_run.log.assert_called_once()
        call_args = mock_run.log.call_args
        logged_metrics = call_args[0][0]
        logged_step = call_args[1]['step']
        
        assert logged_step == 100
        assert 'batch/reward_mean' in logged_metrics
        assert 'batch/policy_loss' in logged_metrics
        assert 'batch/success_rate' in logged_metrics
        assert logged_metrics['batch/reward_mean'] == 2.5
        assert logged_metrics['batch/policy_loss'] == 0.35
    
    def test_log_aggregated_metrics_no_run(self):
        """Test log_aggregated_metrics when wandb run is not initialized."""
        config = MockWandbConfig(enabled=False)
        handler = WandbHandler(config)
        
        metrics = create_test_metrics()
        
        # Should not crash when run is None
        handler.log_aggregated_metrics(metrics, 100)
    
    def test_log_aggregated_metrics_with_prefix(self):
        """Test that all metrics get batch/ prefix."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        metrics = {
            'reward_mean': 1.0,
            'similarity_max': 0.9,
            'custom_metric': 42.0
        }
        
        handler.log_aggregated_metrics(metrics, 50)
        
        logged_metrics = mock_run.log.call_args[0][0]
        
        # All metrics should have batch/ prefix
        expected_keys = {'batch/reward_mean', 'batch/similarity_max', 'batch/custom_metric'}
        assert set(logged_metrics.keys()) == expected_keys
    
    @patch('jaxarc.utils.logging.wandb_handler.logger')
    def test_log_aggregated_metrics_error_handling(self, mock_logger):
        """Test error handling in log_aggregated_metrics."""
        mock_run = MagicMock()
        mock_run.log.side_effect = Exception("Network error")
        
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        metrics = {'reward_mean': 1.0}
        
        # Should not crash, should log warning
        handler.log_aggregated_metrics(metrics, 10)
        
        mock_logger.warning.assert_called_with("Wandb batch logging failed: Network error")
    
    def test_log_aggregated_metrics_empty_metrics(self):
        """Test logging empty metrics dictionary."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        metrics = {}
        
        handler.log_aggregated_metrics(metrics, 10)
        
        # Should still call wandb.log with empty dict (with batch/ prefix)
        mock_run.log.assert_called_once_with({}, step=10)
    
    def test_integration_with_existing_methods(self):
        """Test that batched logging integrates with existing WandbHandler methods."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        # Use existing methods
        task_data = {'task_id': 'test_task', 'num_train_pairs': 3}
        handler.log_task_start(task_data)
        
        step_data = {
            'step_num': 1,
            'reward': 0.5,
            'info': {'metrics': {'similarity': 0.8}}
        }
        handler.log_step(step_data)
        
        # Use new batched method
        metrics = {'reward_mean': 2.0}
        handler.log_aggregated_metrics(metrics, 10)
        
        summary_data = {
            'episode_num': 1,
            'total_reward': 1.0,
            'success': True
        }
        handler.log_episode_summary(summary_data)
        
        # All methods should have been called
        assert mock_run.log.call_count == 4  # task, step, batch, summary


class TestRichHandlerBatchedExtensions:
    """Test cases for RichHandler batched logging extensions."""
    
    def test_log_aggregated_metrics_basic(self):
        """Test basic aggregated metrics display to console."""
        config = MockConfig()
        handler = RichHandler(config)
        
        # Mock the console to capture output
        mock_console = MagicMock()
        handler.console = mock_console
        
        metrics = create_test_metrics()
        step = 100
        
        handler.log_aggregated_metrics(metrics, step)
        
        # Check that console.print was called
        mock_console.print.assert_called_once()
        
        # Check that the printed object is a Rich Table
        printed_object = mock_console.print.call_args[0][0]
        assert hasattr(printed_object, 'title')
        assert "Step 100" in str(printed_object.title)
    
    def test_log_aggregated_metrics_categorization(self):
        """Test that metrics are properly categorized in display."""
        config = MockConfig()
        handler = RichHandler(config)
        
        # Mock Rich Table to inspect its contents
        with patch('rich.table.Table') as mock_table_class:
            mock_table = MagicMock()
            mock_table_class.return_value = mock_table
            
            metrics = {
                'reward_mean': 2.0,
                'reward_max': 5.0,
                'similarity_mean': 0.8,
                'similarity_std': 0.1,
                'episode_length_mean': 25,
                'policy_loss': 0.3,
                'value_loss': 0.2,
                'gradient_norm': 1.5,
                'custom_metric': 42.0
            }
            
            handler.log_aggregated_metrics(metrics, 50)
            
            # Check that table was created with proper title
            mock_table_class.assert_called_once()
            table_kwargs = mock_table_class.call_args[1]
            assert "Step 50" in table_kwargs['title']
            
            # Check that add_row was called multiple times (for categories and metrics)
            assert mock_table.add_row.call_count > 0
    
    def test_log_aggregated_metrics_formatting(self):
        """Test metric value formatting in display."""
        config = MockConfig()
        handler = RichHandler(config)
        
        mock_console = MagicMock()
        handler.console = mock_console
        
        metrics = {
            'large_value': 1000.123456,
            'small_value': 0.000123,
            'medium_value': 3.14159,
            'integer_value': 42
        }
        
        # Should not crash with various value types and magnitudes
        handler.log_aggregated_metrics(metrics, 25)
        
        mock_console.print.assert_called_once()
    
    @patch('jaxarc.utils.logging.rich_handler.logger')
    def test_log_aggregated_metrics_error_handling(self, mock_logger):
        """Test error handling in log_aggregated_metrics."""
        config = MockConfig()
        handler = RichHandler(config)
        
        # Mock console to raise an error
        mock_console = MagicMock()
        mock_console.print.side_effect = Exception("Console error")
        handler.console = mock_console
        
        metrics = {'reward_mean': 1.0}
        
        # Should not crash, should log warning
        handler.log_aggregated_metrics(metrics, 10)
        
        mock_logger.warning.assert_called_with("Rich batch logging failed: Console error")
    
    def test_log_aggregated_metrics_empty_metrics(self):
        """Test display with empty metrics dictionary."""
        config = MockConfig()
        handler = RichHandler(config)
        
        mock_console = MagicMock()
        handler.console = mock_console
        
        metrics = {}
        
        handler.log_aggregated_metrics(metrics, 10)
        
        # Should still create and print a table
        mock_console.print.assert_called_once()
    
    def test_integration_with_existing_methods(self):
        """Test that batched logging integrates with existing RichHandler methods."""
        config = MockConfig()
        handler = RichHandler(config)
        
        mock_console = MagicMock()
        handler.console = mock_console
        
        # Use existing methods
        task_data = {'task_id': 'test_task', 'episode_num': 1, 'num_train_pairs': 3}
        handler.log_task_start(task_data)
        
        step_data = {
            'step_num': 1,
            'reward': 0.5,
            'info': {'metrics': {'similarity': 0.8}}
        }
        handler.log_step(step_data)
        
        # Use new batched method
        metrics = {'reward_mean': 2.0}
        handler.log_aggregated_metrics(metrics, 10)
        
        summary_data = {
            'episode_num': 1,
            'total_reward': 1.0,
            'success': True
        }
        handler.log_episode_summary(summary_data)
        
        # All methods should have called console.print
        assert mock_console.print.call_count >= 4


class TestHandlerBatchedIntegration:
    """Test integration between handlers and batched logging."""
    
    def test_handler_method_detection(self):
        """Test that handlers are correctly detected for batched logging support."""
        # Handler with batched support
        class BatchedHandler:
            def log_aggregated_metrics(self, metrics, step):
                self.called = True
        
        # Handler without batched support
        class LegacyHandler:
            def log_step(self, data):
                self.called = True
        
        batched_handler = BatchedHandler()
        legacy_handler = LegacyHandler()
        
        # Test hasattr detection
        assert hasattr(batched_handler, 'log_aggregated_metrics')
        assert not hasattr(legacy_handler, 'log_aggregated_metrics')
    
    def test_graceful_degradation_without_batched_support(self):
        """Test that handlers without batched support don't break the system."""
        class LegacyHandler:
            def __init__(self):
                self.calls = []
            
            def log_step(self, data):
                self.calls.append(('step', data))
            
            def log_episode_summary(self, data):
                self.calls.append(('summary', data))
        
        handler = LegacyHandler()
        
        # Should not have batched method
        assert not hasattr(handler, 'log_aggregated_metrics')
        
        # But should still work for regular methods
        handler.log_step({'step_num': 1})
        handler.log_episode_summary({'episode_num': 1})
        
        assert len(handler.calls) == 2
    
    def test_mixed_handler_support(self):
        """Test system with mix of batched and non-batched handlers."""
        class BatchedHandler:
            def __init__(self):
                self.aggregated_calls = []
                self.summary_calls = []
            
            def log_aggregated_metrics(self, metrics, step):
                self.aggregated_calls.append((metrics, step))
            
            def log_episode_summary(self, data):
                self.summary_calls.append(data)
        
        class LegacyHandler:
            def __init__(self):
                self.summary_calls = []
            
            def log_episode_summary(self, data):
                self.summary_calls.append(data)
        
        batched_handler = BatchedHandler()
        legacy_handler = LegacyHandler()
        
        # Simulate ExperimentLogger behavior
        handlers = {'batched': batched_handler, 'legacy': legacy_handler}
        
        metrics = {'reward_mean': 1.0}
        step = 10
        
        # Call log_aggregated_metrics on handlers that support it
        for handler_name, handler in handlers.items():
            if hasattr(handler, 'log_aggregated_metrics'):
                handler.log_aggregated_metrics(metrics, step)
        
        # Only batched handler should have been called
        assert len(batched_handler.aggregated_calls) == 1
        assert batched_handler.aggregated_calls[0] == (metrics, step)
        
        # Both handlers should work for regular methods
        summary_data = {'episode_num': 1}
        for handler in handlers.values():
            handler.log_episode_summary(summary_data)
        
        assert len(batched_handler.summary_calls) == 1
        assert len(legacy_handler.summary_calls) == 1


class TestHandlerSpecificFunctionality:
    """Test handler-specific functionality for batched logging."""
    
    def test_wandb_batch_prefix_uniqueness(self):
        """Test that wandb batch/ prefix distinguishes from regular metrics."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        # Log regular step metrics
        step_data = {
            'step_num': 1,
            'reward': 0.5,
            'info': {'metrics': {'similarity': 0.8}}
        }
        handler.log_step(step_data)
        
        # Log batch metrics
        batch_metrics = {'reward_mean': 2.0, 'similarity_mean': 0.75}
        handler.log_aggregated_metrics(batch_metrics, 10)
        
        # Check that calls are distinct
        assert mock_run.log.call_count == 2
        
        # First call (step): regular metrics
        step_call = mock_run.log.call_args_list[0][0][0]
        assert 'similarity' in step_call  # No prefix
        assert 'reward' in step_call  # No prefix
        
        # Second call (batch): prefixed metrics
        batch_call = mock_run.log.call_args_list[1][0][0]
        assert 'batch/reward_mean' in batch_call
        assert 'batch/similarity_mean' in batch_call
    
    def test_file_handler_jsonl_format(self):
        """Test that FileHandler uses JSONL format for batch metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            # Log multiple batches
            handler.log_aggregated_metrics({'metric1': 1.0}, 10)
            handler.log_aggregated_metrics({'metric2': 2.0}, 20)
            
            # Check JSONL format (one JSON object per line)
            batch_file = Path(temp_dir) / "logs" / "batch_metrics.jsonl"
            with open(batch_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            
            # Each line should be valid JSON
            entry1 = json.loads(lines[0])
            entry2 = json.loads(lines[1])
            
            assert entry1['step'] == 10
            assert entry2['step'] == 20
    
    def test_rich_handler_table_structure(self):
        """Test that RichHandler creates proper table structure."""
        config = MockConfig()
        handler = RichHandler(config)
        
        with patch('rich.table.Table') as mock_table_class:
            mock_table = MagicMock()
            mock_table_class.return_value = mock_table
            
            metrics = {
                'reward_mean': 1.0,
                'similarity_mean': 0.8,
                'policy_loss': 0.3
            }
            
            handler.log_aggregated_metrics(metrics, 100)
            
            # Check table creation
            mock_table_class.assert_called_once()
            table_args = mock_table_class.call_args
            
            # Should have title with step information
            assert 'title' in table_args[1]
            assert 'Step 100' in table_args[1]['title']
            
            # Should add columns
            assert mock_table.add_column.call_count == 2  # Metric and Value columns
            
            # Should add rows for categories and metrics
            assert mock_table.add_row.call_count > 0