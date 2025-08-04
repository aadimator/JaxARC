"""Unit tests for RichHandler console output formatting."""

from unittest.mock import Mock, patch
from io import StringIO
import pytest
import numpy as np

from jaxarc.utils.logging.rich_handler import RichHandler


class TestRichHandler:
    """Test suite for RichHandler class."""
    
    def test_initialization(self):
        """Test RichHandler initialization."""
        config = Mock()
        config.environment = Mock()
        config.environment.debug_level = "standard"
        
        handler = RichHandler(config)
        
        assert handler.config == config
        assert handler.console is not None
    
    def test_log_step_verbose_mode(self):
        """Test log_step method in verbose mode displays output."""
        config = Mock()
        config.environment = Mock()
        config.environment.debug_level = "verbose"
        
        handler = RichHandler(config)
        
        # Mock state objects
        mock_before_state = Mock()
        mock_before_state.grid = np.array([[0, 1], [1, 0]])
        
        mock_after_state = Mock()
        mock_after_state.grid = np.array([[1, 0], [0, 1]])
        
        step_data = {
            'step_num': 5,
            'before_state': mock_before_state,
            'after_state': mock_after_state,
            'action': {'type': 'fill', 'position': [0, 0], 'color': 1},
            'reward': 0.5,
            'info': {
                'metrics': {
                    'similarity': 0.75,
                    'learning_rate': 0.001
                }
            }
        }
        
        # Should not raise an exception
        handler.log_step(step_data)
    
    def test_log_step_research_mode(self):
        """Test log_step method in research mode displays output."""
        config = Mock()
        config.environment = Mock()
        config.environment.debug_level = "research"
        
        handler = RichHandler(config)
        
        step_data = {
            'step_num': 1,
            'reward': 0.0,
            'info': {'metrics': {'similarity': 0.5}}
        }
        
        # Should not raise an exception
        handler.log_step(step_data)
    
    def test_log_step_standard_mode_no_output(self):
        """Test log_step method in standard mode does not display step details."""
        config = Mock()
        config.environment = Mock()
        config.environment.debug_level = "standard"
        
        handler = RichHandler(config)
        
        step_data = {
            'step_num': 5,
            'reward': 0.5,
            'info': {'metrics': {'similarity': 0.75}}
        }
        
        # Should not raise an exception and should not display step details
        handler.log_step(step_data)
    
    def test_log_episode_summary(self):
        """Test log_episode_summary method."""
        config = Mock()
        config.environment = Mock()
        config.environment.debug_level = "standard"
        
        handler = RichHandler(config)
        
        summary_data = {
            'episode_num': 10,
            'total_steps': 25,
            'total_reward': 1.5,
            'final_similarity': 0.85,
            'success': True,
            'task_id': 'test_task_001'
        }
        
        # Should not raise an exception
        handler.log_episode_summary(summary_data)
    
    def test_graceful_unknown_keys(self):
        """Test that handler ignores unknown keys gracefully (Requirement 6.4)."""
        config = Mock()
        config.environment = Mock()
        config.environment.debug_level = "verbose"
        
        handler = RichHandler(config)
        
        # Step data with unknown keys
        step_data = {
            'step_num': 1,
            'reward': 0.0,
            'unknown_key': 'unknown_value',
            'another_unknown': {'nested': 'data'},
            'info': {
                'metrics': {'similarity': 0.5},
                'unknown_info': 'should_be_ignored'
            }
        }
        
        # Should not raise an exception
        handler.log_step(step_data)
        
        # Test with episode summary
        summary_data = {
            'episode_num': 1,
            'total_steps': 10,
            'total_reward': 0.0,
            'final_similarity': 0.5,
            'success': False,
            'unknown_summary_key': 'ignored'
        }
        
        # Should not raise an exception
        handler.log_episode_summary(summary_data)
    
    def test_config_with_direct_debug_level(self):
        """Test config with direct debug_level attribute."""
        config = Mock()
        config.debug_level = "research"
        
        handler = RichHandler(config)
        
        step_data = {'step_num': 1, 'reward': 0.0}
        
        # Should work without error
        handler.log_step(step_data)
    
    def test_config_without_debug_level(self):
        """Test config without debug_level (should default to standard)."""
        config = Mock()
        # Ensure no debug_level attributes exist
        if hasattr(config, 'debug_level'):
            delattr(config, 'debug_level')
        if hasattr(config, 'environment'):
            delattr(config, 'environment')
        
        handler = RichHandler(config)
        
        step_data = {'step_num': 1, 'reward': 0.0}
        
        # Should work without error and default to standard mode
        handler.log_step(step_data)
    
    def test_close_method(self):
        """Test close method."""
        config = Mock()
        handler = RichHandler(config)
        
        # Should not raise an exception
        handler.close()
    
    def test_metrics_extraction(self):
        """Test that metrics are properly extracted from info dictionary."""
        config = Mock()
        config.environment = Mock()
        config.environment.debug_level = "verbose"
        
        handler = RichHandler(config)
        
        step_data = {
            'step_num': 1,
            'reward': 0.5,
            'info': {
                'metrics': {
                    'similarity': 0.75,
                    'ppo_policy_value': 0.123,
                    'learning_rate': 0.001
                },
                'other_data': 'should_be_ignored_in_metrics'
            }
        }
        
        # Should not raise an exception and should handle metrics properly
        handler.log_step(step_data)
    
    def test_missing_state_data(self):
        """Test handling of missing state data."""
        config = Mock()
        config.environment = Mock()
        config.environment.debug_level = "verbose"
        
        handler = RichHandler(config)
        
        # Step data without state information
        step_data = {
            'step_num': 1,
            'reward': 0.0,
            'action': {'type': 'fill'},
            'info': {'metrics': {'similarity': 0.5}}
        }
        
        # Should not raise an exception
        handler.log_step(step_data)
    
    def test_missing_episode_data(self):
        """Test handling of missing episode data."""
        config = Mock()
        handler = RichHandler(config)
        
        # Minimal episode summary data
        summary_data = {
            'episode_num': 1
        }
        
        # Should not raise an exception
        handler.log_episode_summary(summary_data)
    
    def test_different_reward_values(self):
        """Test display of different reward values with color coding."""
        config = Mock()
        config.environment = Mock()
        config.environment.debug_level = "verbose"
        
        handler = RichHandler(config)
        
        # Test positive reward
        step_data_positive = {'step_num': 1, 'reward': 1.0}
        handler.log_step(step_data_positive)
        
        # Test negative reward
        step_data_negative = {'step_num': 2, 'reward': -0.5}
        handler.log_step(step_data_negative)
        
        # Test zero reward
        step_data_zero = {'step_num': 3, 'reward': 0.0}
        handler.log_step(step_data_zero)
    
    def test_success_status_display(self):
        """Test display of success status in episode summary."""
        config = Mock()
        handler = RichHandler(config)
        
        # Test successful episode
        summary_success = {
            'episode_num': 1,
            'success': True,
            'total_reward': 1.0,
            'final_similarity': 0.9
        }
        handler.log_episode_summary(summary_success)
        
        # Test failed episode
        summary_failure = {
            'episode_num': 2,
            'success': False,
            'total_reward': -0.5,
            'final_similarity': 0.3
        }
        handler.log_episode_summary(summary_failure)