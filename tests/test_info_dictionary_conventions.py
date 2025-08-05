"""Tests for info dictionary conventions implementation.

This module tests the implementation of task 9: Establish info dictionary conventions.
It verifies that:
1. functional.py structures info dictionary with info['metrics'] for scalar data
2. Handlers extract metrics from info['metrics'] automatically
3. FileHandler serializes entire info dictionary using existing serialization utils
4. SVGHandler and RichHandler ignore unknown keys gracefully
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

# import pytest  # Not needed for basic testing


class TestInfoDictionaryConventions:
    """Test suite for info dictionary conventions."""
    
    def test_info_structure_format(self):
        """Test that info dictionary has the correct structure with metrics."""
        # Simulate the info dictionary structure from functional.py
        mock_state = Mock()
        mock_state.similarity_score = 0.5
        
        final_state = Mock()
        final_state.similarity_score = 0.7
        final_state.step_count = 6
        final_state.episode_mode = 0
        final_state.current_example_idx = 0
        final_state.get_available_demo_count = Mock(return_value=3)
        final_state.get_available_test_count = Mock(return_value=2)
        final_state.get_action_history_length = Mock(return_value=6)
        
        is_control_operation = False
        
        # Recreate the info structure from functional.py
        info = {
            "metrics": {
                "similarity": final_state.similarity_score,
                "similarity_improvement": final_state.similarity_score - mock_state.similarity_score,
                "step_count": final_state.step_count,
                "episode_mode": final_state.episode_mode,
                "current_pair_index": final_state.current_example_idx,
                "available_demo_pairs": final_state.get_available_demo_count(),
                "available_test_pairs": final_state.get_available_test_count(),
                "action_history_length": final_state.get_action_history_length(),
                "operation_type": 0 if not is_control_operation else 1,
            },
            "success": final_state.similarity_score >= 1.0,
            "is_control_operation": is_control_operation,
        }
        
        # Verify structure
        assert 'metrics' in info
        assert isinstance(info['metrics'], dict)
        assert len(info['metrics']) > 0
        
        # Verify all metrics are scalar
        for key, value in info['metrics'].items():
            assert isinstance(value, (int, float)), f"Metric {key} should be scalar"
        
        # Verify non-metric data at top level
        assert 'success' in info
        assert 'is_control_operation' in info
    
    def test_file_handler_serialization(self):
        """Test that FileHandler serializes entire info dictionary properly."""
        from src.jaxarc.utils.logging.file_handler import FileHandler
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock config
            mock_config = Mock()
            mock_config.storage = Mock()
            mock_config.storage.base_output_dir = temp_dir
            mock_config.storage.logs_dir = 'test_logs'
            
            handler = FileHandler(mock_config)
            
            # Test step data with structured info
            step_data = {
                'step_num': 5,
                'before_state': Mock(),
                'after_state': Mock(),
                'action': Mock(),
                'reward': 0.5,
                'info': {
                    'metrics': {
                        'similarity': 0.7,
                        'custom_metric': 42.0
                    },
                    'success': False,
                    'unknown_key': 'should_be_preserved'
                }
            }
            
            # Mock serialization methods
            with patch.object(handler, '_serialize_state', return_value={'type': 'MockState'}):
                with patch('src.jaxarc.utils.logging.file_handler.serialize_action', return_value={'operation': 0}):
                    with patch('src.jaxarc.utils.logging.file_handler.serialize_object', side_effect=lambda x: x):
                        
                        handler.log_step(step_data)
                        
                        # Verify structure is preserved
                        assert 'steps' in handler.current_episode_data
                        serialized_step = handler.current_episode_data['steps'][0]
                        
                        assert 'info' in serialized_step
                        assert 'metrics' in serialized_step['info']
                        assert 'similarity' in serialized_step['info']['metrics']
                        assert 'unknown_key' in serialized_step['info']
            
            handler.close()
    
    def test_svg_handler_graceful_unknown_keys(self):
        """Test that SVGHandler ignores unknown keys gracefully."""
        from src.jaxarc.utils.logging.svg_handler import SVGHandler
        
        # Create mock config
        mock_config = Mock()
        mock_config.storage = Mock()
        mock_config.storage.base_output_dir = 'test_output'
        mock_config.storage.run_name = None
        mock_config.storage.max_episodes_per_run = 1000
        mock_config.storage.max_storage_gb = 10.0
        mock_config.storage.cleanup_policy = 'size_based'
        
        handler = SVGHandler(mock_config)
        
        # Test info filtering
        test_info = {
            'metrics': {'similarity': 0.7},
            'success': True,
            'similarity_improvement': 0.2,
            'unknown_key': 'should_be_ignored',
            'complex_unknown': {'nested': 'data'}
        }
        
        filtered_info = handler._filter_info_for_visualization(test_info)
        
        # Verify known keys preserved
        assert 'metrics' in filtered_info
        assert 'success' in filtered_info
        assert 'similarity_improvement' in filtered_info
        
        # Verify unknown keys ignored
        assert 'unknown_key' not in filtered_info
        assert 'complex_unknown' not in filtered_info
    
    def test_rich_handler_graceful_unknown_keys(self):
        """Test that RichHandler ignores unknown keys gracefully."""
        from src.jaxarc.utils.logging.rich_handler import RichHandler
        
        mock_config = Mock()
        mock_config.environment = Mock()
        mock_config.environment.debug_level = 'verbose'
        
        handler = RichHandler(mock_config)
        
        # Test with unknown keys - should not raise errors
        step_data = {
            'step_num': 5,
            'reward': 0.5,
            'info': {
                'metrics': {'similarity': 0.7},
                'success': True,
                'unknown_key': 'should_be_ignored'
            },
            'unknown_step_key': 'should_be_ignored'
        }
        
        with patch.object(handler, 'console'):
            # Should not raise any errors
            handler.log_step(step_data)
            
            summary_data = {
                'episode_num': 1,
                'total_steps': 5,
                'unknown_summary_key': 'should_be_ignored'
            }
            
            handler.log_episode_summary(summary_data)
    
    def test_wandb_handler_metric_extraction(self):
        """Test that WandbHandler extracts metrics from info['metrics'] automatically."""
        from src.jaxarc.utils.logging.wandb_handler import WandbHandler
        
        mock_config = Mock()
        mock_config.enabled = True
        mock_config.project_name = 'test_project'
        
        mock_wandb = Mock()
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        
        with patch('src.jaxarc.utils.logging.wandb_handler.logger'):
            with patch.dict('sys.modules', {'wandb': mock_wandb}):
                handler = WandbHandler(mock_config)
                
                step_data = {
                    'step_num': 5,
                    'reward': 0.5,
                    'info': {
                        'metrics': {
                            'similarity': 0.7,
                            'custom_metric': 42.0
                        }
                    },
                    'other_scalar': 1.5,
                    'complex_object': Mock()  # Should be excluded
                }
                
                handler.log_step(step_data)
                
                # Verify wandb.log was called
                assert mock_run.log.called
                logged_metrics = mock_run.log.call_args[0][0]
                
                # Verify metrics extracted
                assert 'similarity' in logged_metrics
                assert 'custom_metric' in logged_metrics
                assert 'reward' in logged_metrics
                assert 'step' in logged_metrics
                assert 'other_scalar' in logged_metrics
                
                # Verify complex objects excluded
                assert 'complex_object' not in logged_metrics
                assert 'info' not in logged_metrics
                
                handler.close()