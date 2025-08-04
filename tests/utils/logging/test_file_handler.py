"""Unit tests for FileHandler.

This module tests the FileHandler implementation for synchronous file logging
with JAX array serialization support.
"""

import json
import pickle
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

from jaxarc.utils.logging.file_handler import FileHandler


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else MockStorageConfig()


class MockStorageConfig:
    """Mock storage configuration."""
    def __init__(self, base_output_dir="test_outputs", logs_dir="logs"):
        self.base_output_dir = base_output_dir
        self.logs_dir = logs_dir


class MockState:
    """Mock state object for testing."""
    def __init__(self, working_grid, target_grid, selected, clipboard, 
                 step_count, episode_done, similarity_score, current_example_idx,
                 task_data=None):
        self.working_grid = working_grid
        self.target_grid = target_grid
        self.selected = selected
        self.clipboard = clipboard
        self.step_count = step_count
        self.episode_done = episode_done
        self.similarity_score = similarity_score
        self.current_example_idx = current_example_idx
        self.task_data = task_data if task_data is not None else MockTaskData()


class MockTaskData:
    """Mock task data for testing."""
    def __init__(self, task_index=None):
        self.task_index = task_index if task_index is not None else np.array(0)


def create_test_step_data() -> Dict[str, Any]:
    """Create test step data with numpy arrays (simulating JAX arrays)."""
    before_state = MockState(
        working_grid=np.zeros((5, 5), dtype=np.int32),
        target_grid=np.ones((5, 5), dtype=np.int32),
        selected=np.zeros((5, 5), dtype=np.bool_),
        clipboard=np.zeros((5, 5), dtype=np.int32),
        step_count=0,
        episode_done=False,
        similarity_score=0.0,
        current_example_idx=0
    )
    
    after_state = MockState(
        working_grid=np.ones((5, 5), dtype=np.int32),
        target_grid=np.ones((5, 5), dtype=np.int32),
        selected=np.ones((5, 5), dtype=np.bool_),
        clipboard=np.zeros((5, 5), dtype=np.int32),
        step_count=1,
        episode_done=False,
        similarity_score=1.0,
        current_example_idx=0
    )
    
    return {
        'step_num': 1,
        'before_state': before_state,
        'after_state': after_state,
        'action': {'operation': 'fill', 'color': 1},
        'reward': 1.0,
        'info': {
            'metrics': {
                'similarity': 1.0,
                'step_reward': 1.0
            },
            'debug_info': 'test step'
        }
    }


def create_test_summary_data() -> Dict[str, Any]:
    """Create test episode summary data."""
    return {
        'episode_num': 1,
        'total_steps': 5,
        'total_reward': 5.0,
        'final_similarity': 1.0,
        'task_id': 'test_task_001',
        'success': True
    }


class TestFileHandler:
    """Test cases for FileHandler."""
    
    def test_initialization(self):
        """Test FileHandler initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            assert handler.output_dir == Path(temp_dir) / "logs"
            assert handler.output_dir.exists()
            assert handler.current_episode_data == {}
    
    def test_step_logging(self):
        """Test step data logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            step_data = create_test_step_data()
            
            # Log a step
            handler.log_step(step_data)
            
            # Check that step was added to current episode data
            assert 'steps' in handler.current_episode_data
            assert len(handler.current_episode_data['steps']) == 1
            
            # Check that step data was serialized properly
            logged_step = handler.current_episode_data['steps'][0]
            assert 'step_num' in logged_step
            assert 'before_state' in logged_step
            assert 'after_state' in logged_step
            assert 'action' in logged_step
            assert 'reward' in logged_step
            assert 'info' in logged_step
            
            # Check that state was serialized
            before_state = logged_step['before_state']
            assert 'type' in before_state
            assert before_state['type'] == 'MockState'
    
    def test_episode_summary_logging(self):
        """Test episode summary logging and file creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            # Log some steps first
            for i in range(3):
                step_data = create_test_step_data()
                step_data['step_num'] = i + 1
                handler.log_step(step_data)
            
            # Log episode summary
            summary_data = create_test_summary_data()
            handler.log_episode_summary(summary_data)
            
            # Check that files were created
            output_dir = Path(temp_dir) / "logs"
            json_files = list(output_dir.glob("episode_0001_*.json"))
            pickle_files = list(output_dir.glob("episode_0001_*.pkl"))
            
            assert len(json_files) == 1, f"Expected 1 JSON file, found {len(json_files)}"
            assert len(pickle_files) == 1, f"Expected 1 pickle file, found {len(pickle_files)}"
            
            # Check JSON file content
            with open(json_files[0], 'r') as f:
                json_data = json.load(f)
            
            assert json_data['episode_num'] == 1
            assert json_data['total_steps'] == 5
            assert len(json_data['steps']) == 3
            assert 'timestamp' in json_data
            assert 'config_hash' in json_data
            
            # Check pickle file content
            with open(pickle_files[0], 'rb') as f:
                pickle_data = pickle.load(f)
            
            assert pickle_data['episode_num'] == 1
            assert len(pickle_data['steps']) == 3
            
            # Check that current episode data was reset
            assert handler.current_episode_data == {}
    
    def test_jax_array_serialization(self):
        """Test JAX array serialization specifically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            # Create step data with various numpy array types (simulating JAX arrays)
            step_data = {
                'step_num': 1,
                'before_state': MockState(
                    working_grid=np.array([[1, 2], [3, 4]], dtype=np.int32),
                    target_grid=np.array([[5, 6], [7, 8]], dtype=np.int32),
                    selected=np.array([[True, False], [False, True]], dtype=np.bool_),
                    clipboard=np.zeros((2, 2), dtype=np.int32),
                    step_count=1,
                    episode_done=False,
                    similarity_score=0.5,
                    current_example_idx=0
                ),
                'after_state': None,  # Test None handling
                'action': {'operation': 'test'},
                'reward': np.array(1.0),  # Numpy scalar (simulating JAX scalar)
                'info': {
                    'metrics': {
                        'numpy_metric': np.array([1.0, 2.0, 3.0])  # Numpy array (simulating JAX array)
                    }
                }
            }
            
            handler.log_step(step_data)
            
            # Check serialization
            logged_step = handler.current_episode_data['steps'][0]
            
            # Check that numpy arrays were converted to lists
            before_state = logged_step['before_state']
            assert before_state['arrays']['working_grid'] == [[1, 2], [3, 4]]
            assert before_state['arrays']['selected'] == [[True, False], [False, True]]
            
            # Check None handling
            after_state = logged_step['after_state']
            assert after_state['type'] == 'None'
            
            # Check numpy scalar serialization
            assert isinstance(logged_step['reward'], (int, float))
            
            # Check numpy array in metrics
            metrics = logged_step['info']['metrics']
            assert metrics['numpy_metric'] == [1.0, 2.0, 3.0]
    
    def test_handler_close(self):
        """Test handler close functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            # Add some incomplete episode data
            step_data = create_test_step_data()
            handler.log_step(step_data)
            
            # Close handler
            handler.close()
            
            # Check that incomplete episode data was saved
            output_dir = Path(temp_dir) / "logs"
            incomplete_file = output_dir / "incomplete_episode.json"
            
            assert incomplete_file.exists()
            
            with open(incomplete_file, 'r') as f:
                incomplete_data = json.load(f)
            
            assert 'steps' in incomplete_data
            assert len(incomplete_data['steps']) == 1
    
    def test_error_handling(self):
        """Test error handling in serialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            # Create problematic step data
            class UnserializableObject:
                def __init__(self):
                    self.circular_ref = self
            
            step_data = {
                'step_num': 1,
                'before_state': UnserializableObject(),
                'after_state': None,
                'action': {'operation': 'test'},
                'reward': 1.0,
                'info': {}
            }
            
            # This should not crash, but handle the error gracefully
            handler.log_step(step_data)
            
            # Check that step was logged with error information
            logged_step = handler.current_episode_data['steps'][0]
            before_state = logged_step['before_state']
            
            # Should contain error information
            assert 'serialization_error' in before_state or 'type' in before_state
    
    def test_legacy_config_support(self):
        """Test support for legacy config structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a config with legacy debug.output_dir structure
            class LegacyConfig:
                def __init__(self):
                    self.debug = LegacyDebugConfig(temp_dir)
            
            class LegacyDebugConfig:
                def __init__(self, output_dir):
                    self.output_dir = output_dir
            
            config = LegacyConfig()
            handler = FileHandler(config)
            
            assert handler.output_dir == Path(temp_dir)
            assert handler.output_dir.exists()
    
    def test_metrics_preservation(self):
        """Test that metrics structure is preserved for wandb integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = MockStorageConfig(base_output_dir=temp_dir, logs_dir="logs")
            config = MockConfig(storage=storage_config)
            
            handler = FileHandler(config)
            
            step_data = create_test_step_data()
            step_data['info']['metrics']['custom_metric'] = 42.0
            
            handler.log_step(step_data)
            
            logged_step = handler.current_episode_data['steps'][0]
            assert 'metrics' in logged_step['info']
            assert logged_step['info']['metrics']['custom_metric'] == 42.0
            assert logged_step['info']['metrics']['similarity'] == 1.0