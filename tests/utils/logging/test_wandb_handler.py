"""Tests for WandbHandler simplified wandb integration."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, Mock, patch
from typing import Any, Dict

import pytest

from jaxarc.utils.logging.wandb_handler import WandbHandler


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


class TestWandbHandler:
    """Test cases for WandbHandler."""
    
    def test_init_disabled_config(self):
        """Test initialization with disabled wandb config."""
        config = MockWandbConfig(enabled=False)
        handler = WandbHandler(config)
        
        assert handler.config == config
        assert handler.run is None
    
    @patch('jaxarc.utils.logging.wandb_handler.logger')
    def test_init_disabled_logs_message(self, mock_logger):
        """Test that disabled config logs appropriate message."""
        config = MockWandbConfig(enabled=False)
        WandbHandler(config)
        
        mock_logger.info.assert_called_with("Wandb integration disabled in config")
    
    @patch('builtins.__import__')
    @patch('jaxarc.utils.logging.wandb_handler.logger')
    def test_init_wandb_not_available(self, mock_logger, mock_import):
        """Test initialization when wandb is not available."""
        mock_import.side_effect = ImportError("No module named 'wandb'")
        
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        
        assert handler.run is None
        mock_logger.warning.assert_called_with("wandb not available - skipping wandb logging")
    
    @patch('jaxarc.utils.logging.wandb_handler.os.environ', {})
    def test_init_successful(self):
        """Test successful wandb initialization."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_run.id = "test-id"
        mock_wandb.init.return_value = mock_run
        
        config = MockWandbConfig(enabled=True)
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            handler = WandbHandler(config)
        
        assert handler.run == mock_run
        mock_wandb.init.assert_called_once_with(
            project='test-project',
            entity=None,
            tags=['test'],
            notes='Test run',
            group=None,
            job_type='test',
            save_code=True
        )
    
    @patch('jaxarc.utils.logging.wandb_handler.os.environ', {})
    def test_init_offline_mode(self):
        """Test initialization with offline mode."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "test-run"
        mock_run.id = "test-id"
        mock_wandb.init.return_value = mock_run
        
        config = MockWandbConfig(enabled=True, offline_mode=True)
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            handler = WandbHandler(config)
        
        # Check that WANDB_MODE was set to offline
        assert os.environ.get("WANDB_MODE") == "offline"
        assert handler.run == mock_run
    
    @patch('jaxarc.utils.logging.wandb_handler.logger')
    def test_init_wandb_error(self, mock_logger):
        """Test initialization when wandb.init() raises an error."""
        mock_wandb = MagicMock()
        mock_wandb.init.side_effect = Exception("Connection failed")
        
        config = MockWandbConfig(enabled=True)
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            handler = WandbHandler(config)
        
        assert handler.run is None
        mock_logger.warning.assert_called_with("wandb initialization failed: Connection failed")
    
    def test_log_step_no_run(self):
        """Test log_step when wandb run is not initialized."""
        config = MockWandbConfig(enabled=False)
        handler = WandbHandler(config)
        
        # Should not raise an error
        handler.log_step({'step_num': 1, 'reward': 0.5})
    
    def test_log_step_with_metrics(self):
        """Test log_step with info['metrics'] extraction."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        step_data = {
            'step_num': 10,
            'reward': 0.8,
            'info': {
                'metrics': {
                    'similarity': 0.9,
                    'policy_value': 0.7
                }
            }
        }
        
        handler.log_step(step_data)
        
        expected_metrics = {
            'similarity': 0.9,
            'policy_value': 0.7,
            'reward': 0.8,
            'step': 10
        }
        mock_run.log.assert_called_once_with(expected_metrics)
    
    def test_log_step_without_metrics(self):
        """Test log_step without info['metrics']."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        step_data = {
            'step_num': 5,
            'reward': 0.3,
            'some_scalar': 42
        }
        
        handler.log_step(step_data)
        
        expected_metrics = {
            'reward': 0.3,
            'step': 5,
            'some_scalar': 42
        }
        mock_run.log.assert_called_once_with(expected_metrics)
    
    def test_log_step_filters_non_scalar(self):
        """Test that log_step filters out non-scalar values."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        step_data = {
            'step_num': 1,
            'reward': 0.5,
            'before_state': {'grid': [[1, 2], [3, 4]]},  # Should be filtered
            'action': {'type': 'move'},  # Should be filtered
            'info': {'complex_data': [1, 2, 3]},  # Should be filtered (no metrics)
            'scalar_value': 123
        }
        
        handler.log_step(step_data)
        
        expected_metrics = {
            'reward': 0.5,
            'step': 1,
            'scalar_value': 123
        }
        mock_run.log.assert_called_once_with(expected_metrics)
    
    @patch('jaxarc.utils.logging.wandb_handler.logger')
    def test_log_step_error_handling(self, mock_logger):
        """Test error handling in log_step."""
        mock_run = MagicMock()
        mock_run.log.side_effect = Exception("Network error")
        
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        step_data = {'step_num': 1, 'reward': 0.5}
        
        # Should not raise an error
        handler.log_step(step_data)
        
        mock_logger.warning.assert_called_with("wandb step logging failed: Network error")
    
    def test_log_episode_summary_no_run(self):
        """Test log_episode_summary when wandb run is not initialized."""
        config = MockWandbConfig(enabled=False)
        handler = WandbHandler(config)
        
        # Should not raise an error
        handler.log_episode_summary({'episode_num': 1, 'total_reward': 10.0})
    
    def test_log_episode_summary_standard_metrics(self):
        """Test log_episode_summary with standard episode metrics."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        summary_data = {
            'episode_num': 5,
            'total_reward': 15.5,
            'total_steps': 100,
            'final_similarity': 0.85,
            'success': True,
            'extra_metric': 42
        }
        
        handler.log_episode_summary(summary_data)
        
        expected_metrics = {
            'episode_num': 5,
            'total_reward': 15.5,
            'total_steps': 100,
            'final_similarity': 0.85,
            'success': True,
            'extra_metric': 42
        }
        mock_run.log.assert_called_once_with(expected_metrics)
    
    def test_log_episode_summary_with_image(self):
        """Test log_episode_summary with summary image."""
        mock_wandb = MagicMock()
        mock_image = MagicMock()
        mock_wandb.Image.return_value = mock_image
        
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        summary_data = {
            'episode_num': 1,
            'total_reward': 5.0,
            'summary_svg_path': '/path/to/summary.svg'
        }
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            handler.log_episode_summary(summary_data)
        
        # Should log metrics and image separately
        assert mock_run.log.call_count == 2
        
        # First call: metrics
        first_call_args = mock_run.log.call_args_list[0][0][0]
        assert 'episode_num' in first_call_args
        assert 'total_reward' in first_call_args
        
        # Second call: image
        second_call_args = mock_run.log.call_args_list[1][0][0]
        assert 'episode_summary' in second_call_args
        assert second_call_args['episode_summary'] == mock_image
        
        mock_wandb.Image.assert_called_once_with('/path/to/summary.svg')
    
    @patch('jaxarc.utils.logging.wandb_handler.logger')
    def test_log_episode_summary_image_error(self, mock_logger):
        """Test error handling when logging summary image fails."""
        mock_wandb = MagicMock()
        mock_wandb.Image.side_effect = Exception("Image error")
        
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        summary_data = {
            'episode_num': 1,
            'total_reward': 5.0,
            'summary_svg_path': '/path/to/summary.svg'
        }
        
        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            handler.log_episode_summary(summary_data)
        
        # Should still log metrics
        mock_run.log.assert_called_once()
        mock_logger.warning.assert_called_with("Failed to log summary image: Image error")
    
    @patch('jaxarc.utils.logging.wandb_handler.logger')
    def test_log_episode_summary_error_handling(self, mock_logger):
        """Test error handling in log_episode_summary."""
        mock_run = MagicMock()
        mock_run.log.side_effect = Exception("Network error")
        
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        summary_data = {'episode_num': 1, 'total_reward': 5.0}
        
        # Should not raise an error
        handler.log_episode_summary(summary_data)
        
        mock_logger.warning.assert_called_with("wandb episode logging failed: Network error")
    
    def test_close_no_run(self):
        """Test close when wandb run is not initialized."""
        config = MockWandbConfig(enabled=False)
        handler = WandbHandler(config)
        
        # Should not raise an error
        handler.close()
    
    @patch('jaxarc.utils.logging.wandb_handler.logger')
    def test_close_successful(self, mock_logger):
        """Test successful close of wandb run."""
        mock_run = MagicMock()
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        handler.close()
        
        mock_run.finish.assert_called_once()
        mock_logger.info.assert_called_with("Wandb run finished successfully")
        assert handler.run is None
    
    @patch('jaxarc.utils.logging.wandb_handler.logger')
    def test_close_error_handling(self, mock_logger):
        """Test error handling in close."""
        mock_run = MagicMock()
        mock_run.finish.side_effect = Exception("Finish error")
        
        config = MockWandbConfig(enabled=True)
        handler = WandbHandler(config)
        handler.run = mock_run
        
        handler.close()
        
        mock_run.finish.assert_called_once()
        mock_logger.warning.assert_called_with("wandb finish failed: Finish error")
        assert handler.run is None