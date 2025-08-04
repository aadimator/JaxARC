"""Tests for SVGHandler functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np

from src.jaxarc.utils.logging.svg_handler import SVGHandler
from src.jaxarc.types import Grid


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self):
        self.visualization = Mock()
        self.visualization.enabled = True
        self.visualization.level = 'verbose'
        self.visualization.step_visualizations = True
        self.visualization.episode_summaries = True
        
        self.storage = Mock()
        self.storage.base_output_dir = 'test_outputs'
        self.storage.run_name = None
        self.storage.max_episodes_per_run = 100
        self.storage.max_storage_gb = 1.0
        self.storage.cleanup_policy = 'manual'
        
        self.environment = Mock()
        self.environment.debug_level = 'verbose'


class MockState:
    """Mock environment state for testing."""
    
    def __init__(self, grid_data, mask_data=None):
        self.working_grid = np.array(grid_data)
        if mask_data is not None:
            self.working_grid_mask = np.array(mask_data)
        else:
            # Create default mask
            self.working_grid_mask = np.ones_like(self.working_grid, dtype=bool)


class MockAction:
    """Mock action for testing."""
    
    def __init__(self, operation_id=0, selection=None):
        self.operation = operation_id
        self.selection = selection if selection is not None else np.zeros((5, 5), dtype=bool)
    
    def __contains__(self, key):
        """Support 'in' operator for compatibility with visualization code."""
        return hasattr(self, key)
    
    def __getitem__(self, key):
        """Support dictionary-style access for compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


@pytest.fixture
def mock_config():
    """Provide mock configuration."""
    return MockConfig()


@pytest.fixture
def temp_dir():
    """Provide temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def svg_handler(mock_config, temp_dir):
    """Provide SVGHandler instance for testing."""
    # Update config to use temp directory
    mock_config.storage.base_output_dir = str(temp_dir)
    return SVGHandler(mock_config)


def test_svg_handler_initialization(svg_handler, mock_config):
    """Test SVGHandler initialization."""
    assert svg_handler.config == mock_config
    assert svg_handler.episode_manager is not None
    assert svg_handler.current_episode_num is None
    assert not svg_handler.current_run_started


def test_start_run(svg_handler, temp_dir):
    """Test starting a new run."""
    run_name = "test_run"
    svg_handler.start_run(run_name)
    
    assert svg_handler.current_run_started
    assert svg_handler.episode_manager.current_run_name == run_name
    
    # Check that run directory was created
    run_dir = temp_dir / run_name
    assert run_dir.exists()


def test_start_episode(svg_handler):
    """Test starting a new episode."""
    episode_num = 5
    svg_handler.start_episode(episode_num)
    
    assert svg_handler.current_episode_num == episode_num
    assert svg_handler.current_run_started  # Should auto-start run


def test_should_generate_step_svg(svg_handler, mock_config):
    """Test step SVG generation decision logic."""
    # Test enabled with verbose level
    mock_config.visualization.enabled = True
    mock_config.visualization.level = 'verbose'
    mock_config.visualization.step_visualizations = True
    assert svg_handler._should_generate_step_svg()
    
    # Test disabled visualization
    mock_config.visualization.enabled = False
    assert not svg_handler._should_generate_step_svg()
    
    # Test off level
    mock_config.visualization.enabled = True
    mock_config.visualization.level = 'off'
    assert not svg_handler._should_generate_step_svg()
    
    # Test standard level (should not generate step SVGs)
    mock_config.visualization.level = 'standard'
    assert not svg_handler._should_generate_step_svg()


def test_should_generate_episode_summary(svg_handler, mock_config):
    """Test episode summary generation decision logic."""
    # Test enabled
    mock_config.visualization.enabled = True
    mock_config.visualization.level = 'standard'
    mock_config.visualization.episode_summaries = True
    assert svg_handler._should_generate_episode_summary()
    
    # Test disabled visualization
    mock_config.visualization.enabled = False
    assert not svg_handler._should_generate_episode_summary()
    
    # Test off level
    mock_config.visualization.enabled = True
    mock_config.visualization.level = 'off'
    assert not svg_handler._should_generate_episode_summary()


def test_extract_grid_from_state(svg_handler):
    """Test grid extraction from different state formats."""
    # Test ArcEnvState format
    grid_data = [[0, 1, 2], [3, 4, 5]]
    mask_data = [[True, True, False], [True, False, True]]
    state = MockState(grid_data, mask_data)
    
    grid = svg_handler._extract_grid_from_state(state)
    assert grid is not None
    assert isinstance(grid, Grid)
    np.testing.assert_array_equal(grid.data, grid_data)
    np.testing.assert_array_equal(grid.mask, mask_data)
    
    # Test Grid object
    existing_grid = Grid(data=np.array(grid_data), mask=np.array(mask_data))
    extracted = svg_handler._extract_grid_from_state(existing_grid)
    assert extracted == existing_grid
    
    # Test raw array
    raw_array = np.array(grid_data)
    grid = svg_handler._extract_grid_from_state(raw_array)
    assert grid is not None
    np.testing.assert_array_equal(grid.data, grid_data)
    
    # Test invalid input
    invalid_state = "invalid"
    grid = svg_handler._extract_grid_from_state(invalid_state)
    assert grid is None


def test_extract_operation_id(svg_handler):
    """Test operation ID extraction from actions."""
    # Test structured action
    action = MockAction(operation_id=5)
    op_id = svg_handler._extract_operation_id(action)
    assert op_id == 5
    
    # Test dictionary action
    action_dict = {'operation': 10, 'selection': np.zeros((3, 3))}
    op_id = svg_handler._extract_operation_id(action_dict)
    assert op_id == 10
    
    # Test invalid action
    invalid_action = "invalid"
    op_id = svg_handler._extract_operation_id(invalid_action)
    assert op_id == 0


@patch('src.jaxarc.utils.logging.svg_handler.draw_rl_step_svg_enhanced')
def test_log_step(mock_draw_step, svg_handler, temp_dir):
    """Test step logging functionality."""
    # Mock the SVG generation function
    mock_draw_step.return_value = "<svg>test step svg</svg>"
    
    # Create test data
    before_state = MockState([[0, 1], [2, 3]])
    after_state = MockState([[1, 1], [2, 4]])
    action = MockAction(operation_id=1)
    
    step_data = {
        'step_num': 3,
        'episode_num': 0,
        'before_state': before_state,
        'after_state': after_state,
        'action': action,
        'reward': 0.5,
        'info': {'similarity': 0.8},
        'task_id': 'test_task',
    }
    
    svg_handler.log_step(step_data)
    
    # Verify SVG generation was called
    mock_draw_step.assert_called_once()
    
    # Verify file was saved
    svg_path = svg_handler.episode_manager.get_step_path(3, "svg")
    assert svg_path.exists()
    
    with open(svg_path, 'r') as f:
        content = f.read()
    assert content == "<svg>test step svg</svg>"


@patch('src.jaxarc.utils.logging.svg_handler.draw_enhanced_episode_summary_svg')
def test_log_episode_summary(mock_draw_summary, svg_handler, temp_dir):
    """Test episode summary logging functionality."""
    # Mock the SVG generation function
    mock_draw_summary.return_value = "<svg>test episode summary</svg>"
    
    # Create test data
    summary_data = {
        'episode_num': 0,
        'total_steps': 10,
        'total_reward': 5.0,
        'final_similarity': 0.9,
        'success': True,
        'task_id': 'test_task',
        'step_data': [],
    }
    
    svg_handler.log_episode_summary(summary_data)
    
    # Verify SVG generation was called
    mock_draw_summary.assert_called_once()
    
    # Verify file was saved
    summary_path = svg_handler.episode_manager.get_episode_summary_path("svg")
    assert summary_path.exists()
    
    with open(summary_path, 'r') as f:
        content = f.read()
    assert content == "<svg>test episode summary</svg>"


def test_log_step_disabled(svg_handler, mock_config):
    """Test that step logging is skipped when disabled."""
    # Disable step visualizations
    mock_config.visualization.level = 'standard'
    
    step_data = {
        'step_num': 1,
        'before_state': MockState([[0, 1]]),
        'after_state': MockState([[1, 1]]),
        'action': MockAction(),
        'reward': 0.0,
        'info': {},
    }
    
    # Should not raise any errors and should not create files
    svg_handler.log_step(step_data)
    
    # No episode should be started
    assert svg_handler.current_episode_num is None


def test_log_episode_summary_disabled(svg_handler, mock_config):
    """Test that episode summary logging is skipped when disabled."""
    # Disable visualization
    mock_config.visualization.enabled = False
    
    summary_data = {
        'episode_num': 0,
        'total_steps': 5,
        'total_reward': 2.0,
        'success': False,
    }
    
    # Should not raise any errors and should not create files
    svg_handler.log_episode_summary(summary_data)


def test_log_step_missing_data(svg_handler):
    """Test step logging with missing required data."""
    # Missing before_state
    step_data = {
        'step_num': 1,
        'after_state': MockState([[1, 1]]),
        'action': MockAction(),
        'reward': 0.0,
    }
    
    # Should not raise errors but should log warning
    svg_handler.log_step(step_data)
    
    # No files should be created
    assert svg_handler.current_episode_num is None


def test_close(svg_handler):
    """Test handler cleanup."""
    # Should not raise any errors
    svg_handler.close()


def test_get_current_run_info(svg_handler):
    """Test getting current run information."""
    info = svg_handler.get_current_run_info()
    assert isinstance(info, dict)
    assert 'run_name' in info
    assert 'run_dir' in info
    assert 'episode_num' in info
    assert 'episode_dir' in info


def test_error_handling_in_log_step(svg_handler):
    """Test error handling during step logging."""
    # Create step data that will cause an error in grid extraction
    step_data = {
        'step_num': 1,
        'before_state': None,  # This will cause an error
        'after_state': MockState([[1, 1]]),
        'action': MockAction(),
        'reward': 0.0,
    }
    
    # Should not raise errors but should handle gracefully
    svg_handler.log_step(step_data)


def test_error_handling_in_log_episode_summary(svg_handler):
    """Test error handling during episode summary logging."""
    # Create summary data that might cause issues
    summary_data = {
        'episode_num': 0,
        # Missing required fields
    }
    
    # Should not raise errors but should handle gracefully
    svg_handler.log_episode_summary(summary_data)


@patch('src.jaxarc.utils.logging.svg_handler.logger')
def test_logging_messages(mock_logger, svg_handler):
    """Test that appropriate log messages are generated."""
    # Test successful run start
    svg_handler.start_run("test_run")
    mock_logger.info.assert_called()
    
    # Test successful episode start
    svg_handler.start_episode(0)
    mock_logger.debug.assert_called()


def test_multiple_episodes(svg_handler, temp_dir):
    """Test handling multiple episodes in sequence."""
    # Start first episode
    svg_handler.start_episode(0)
    assert svg_handler.current_episode_num == 0
    
    # Start second episode
    svg_handler.start_episode(1)
    assert svg_handler.current_episode_num == 1
    
    # Verify both episode directories exist
    run_dir = svg_handler.episode_manager.current_run_dir
    assert (run_dir / "episode_0000").exists()
    assert (run_dir / "episode_0001").exists()


def test_config_fallback_behavior(temp_dir):
    """Test behavior when config attributes are missing."""
    # Create minimal config
    minimal_config = Mock()
    minimal_config.storage = Mock()
    minimal_config.storage.base_output_dir = str(temp_dir)
    
    # Should not raise errors during initialization
    handler = SVGHandler(minimal_config)
    assert handler is not None
    
    # Should handle missing visualization config gracefully
    assert not handler._should_generate_step_svg()
    assert handler._should_generate_episode_summary()  # Default to True