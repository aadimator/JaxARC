"""Comprehensive tests for MiniARC parser implementation.

This test suite covers:
- 5x5 grid constraint validation
- Task loading from flat directory structure
- Performance optimizations
- Error handling for oversized grids
- Task metadata and statistics
- Grid validation and preprocessing
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chex
import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.parsers.mini_arc import MiniArcParser
from jaxarc.types import JaxArcTask


@pytest.fixture
def sample_miniarc_task():
    """Sample MiniARC task data with 5x5 grids."""
    return {
        "train": [
            {
                "input": [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
                "output": [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]
            },
            {
                "input": [[0, 2, 0], [2, 0, 2], [0, 2, 0]],
                "output": [[2, 0, 2], [0, 2, 0], [2, 0, 2]]
            },
        ],
        "test": [
            {
                "input": [[0, 3, 0, 3], [3, 0, 3, 0], [0, 3, 0, 3], [3, 0, 3, 0]],
                "output": [[3, 0, 3, 0], [0, 3, 0, 3], [3, 0, 3, 0], [0, 3, 0, 3]]
            },
        ],
    }


@pytest.fixture
def sample_oversized_task():
    """Sample task with grids exceeding 5x5 constraint."""
    return {
        "train": [
            {
                "input": [[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], 
                         [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]],  # 6x6 grid
                "output": [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
                          [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]
            },
        ],
        "test": [
            {
                "input": [[0, 2, 0, 2, 0, 2], [2, 0, 2, 0, 2, 0]],  # 2x6 grid
            },
        ],
    }


@pytest.fixture
def miniarc_config():
    """Sample MiniARC configuration optimized for 5x5 grids."""
    return DictConfig(
        {
            "grid": {
                "max_grid_height": 5,
                "max_grid_width": 5,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 3,
            "max_test_pairs": 1,
            "tasks": {
                "path": "data/raw/MiniARC/data/MiniARC",
            },
        }
    )


@pytest.fixture
def suboptimal_miniarc_config():
    """MiniARC configuration with suboptimal grid constraints."""
    return DictConfig(
        {
            "grid": {
                "max_grid_height": 30,  # Suboptimal for MiniARC
                "max_grid_width": 30,   # Suboptimal for MiniARC
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 4,
            "max_test_pairs": 3,
            "tasks": {
                "path": "data/raw/MiniARC/data/MiniARC",
            },
        }
    )


@pytest.fixture
def mock_miniarc_directory(sample_miniarc_task):
    """Create a mock MiniARC directory structure with flat file organization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        tasks_dir = Path(temp_dir) / "data" / "MiniARC"
        tasks_dir.mkdir(parents=True)

        # Create sample task files in flat structure
        task_files = [
            "pattern_reversal_001.json",
            "color_swap_002.json", 
            "symmetry_test_003.json",
            "rotation_task_004.json",
            "fill_pattern_005.json",
        ]

        for task_file in task_files:
            task_path = tasks_dir / task_file
            with task_path.open("w") as f:
                json.dump(sample_miniarc_task, f)

        yield temp_dir


class TestMiniArcParser:
    """Test suite for MiniArcParser implementation."""

    def test_miniarc_parser_initialization_optimal_config(self, miniarc_config, mock_miniarc_directory):
        """Test MiniArcParser initialization with optimal 5x5 configuration."""
        # Update config to point to mock directory
        miniarc_config.tasks.path = str(Path(mock_miniarc_directory) / "data" / "MiniARC")

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(miniarc_config.tasks.path)

            parser = MiniArcParser(miniarc_config)

            # Check that tasks were loaded
            task_ids = parser.get_available_task_ids()
            assert len(task_ids) == 5
            
            # Check task IDs match filenames (without .json extension)
            expected_ids = [
                "pattern_reversal_001",
                "color_swap_002", 
                "symmetry_test_003",
                "rotation_task_004",
                "fill_pattern_005",
            ]
            assert set(task_ids) == set(expected_ids)

    def test_miniarc_parser_initialization_suboptimal_config(self, suboptimal_miniarc_config, mock_miniarc_directory):
        """Test MiniArcParser initialization with suboptimal configuration (should warn)."""
        # Update config to point to mock directory
        suboptimal_miniarc_config.tasks.path = str(Path(mock_miniarc_directory) / "data" / "MiniARC")

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(suboptimal_miniarc_config.tasks.path)
            
            with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
                parser = MiniArcParser(suboptimal_miniarc_config)

                # Should have logged a warning about suboptimal configuration
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "MiniARC is optimized for 5x5 grids" in warning_call
                assert "30x30" in warning_call

                # Should still load tasks successfully
                task_ids = parser.get_available_task_ids()
                assert len(task_ids) == 5

    def test_miniarc_parser_5x5_grid_constraint_validation(self, miniarc_config):
        """Test 5x5 grid constraint validation during task loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "data" / "MiniARC"
            tasks_dir.mkdir(parents=True)

            # Create task with oversized grid
            oversized_task = {
                "train": [
                    {
                        "input": [[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], 
                                 [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]],  # 6x6 grid
                        "output": [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]
                    },
                ],
                "test": [{"input": [[0, 2, 0, 2, 0, 2]]}],  # 1x6 grid
            }

            # Write oversized task to file
            oversized_file = tasks_dir / "oversized_task.json"
            with oversized_file.open("w") as f:
                json.dump(oversized_task, f)

            # Update config to point to temp directory
            miniarc_config.tasks.path = str(tasks_dir)

            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = Path(miniarc_config.tasks.path)
                
                with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
                    parser = MiniArcParser(miniarc_config)

                    # Should have logged errors about oversized grids
                    mock_logger.error.assert_called()
                    error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
                    
                    # Check that error mentions grid size constraint violation
                    constraint_errors = [call for call in error_calls if "exceeds MiniARC 5x5 constraint" in call]
                    assert len(constraint_errors) > 0

                    # Task should not be loaded due to constraint violation
                    task_ids = parser.get_available_task_ids()
                    assert "oversized_task" not in task_ids

    def test_miniarc_parser_flat_directory_structure_loading(self, miniarc_config, mock_miniarc_directory):
        """Test task loading from flat directory structure."""
        # Update config to point to mock directory
        miniarc_config.tasks.path = str(Path(mock_miniarc_directory) / "data" / "MiniARC")

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(miniarc_config.tasks.path)

            parser = MiniArcParser(miniarc_config)

            # Test that all tasks were loaded from flat structure
            task_ids = parser.get_available_task_ids()
            assert len(task_ids) == 5

            # Test getting specific task by ID
            task_id = task_ids[0]
            task = parser.get_task_by_id(task_id)
            assert isinstance(task, JaxArcTask)
            assert task.num_train_pairs == 2
            assert task.num_test_pairs == 1

            # Test that task IDs correspond to filenames
            for task_id in task_ids:
                assert not task_id.endswith(".json")  # Should strip extension
                task = parser.get_task_by_id(task_id)
                assert isinstance(task, JaxArcTask)

    def test_miniarc_parser_performance_optimizations(self, miniarc_config, mock_miniarc_directory):
        """Test performance optimizations for 5x5 grids."""
        # Update config to point to mock directory
        miniarc_config.tasks.path = str(Path(mock_miniarc_directory) / "data" / "MiniARC")

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(miniarc_config.tasks.path)

            parser = MiniArcParser(miniarc_config)

            # Test that parser is configured for optimal 5x5 performance
            assert parser.max_grid_height == 5
            assert parser.max_grid_width == 5

            # Get a sample task and verify optimized array shapes
            key = jax.random.PRNGKey(42)
            task = parser.get_random_task(key)

            # Arrays should be padded to 5x5, not larger dimensions
            expected_shape = (miniarc_config.max_train_pairs, 5, 5)
            assert task.input_grids_examples.shape == expected_shape
            assert task.output_grids_examples.shape == expected_shape

            expected_test_shape = (miniarc_config.max_test_pairs, 5, 5)
            assert task.test_input_grids.shape == expected_test_shape
            assert task.true_test_output_grids.shape == expected_test_shape

            # Test dataset statistics show optimization
            stats = parser.get_dataset_statistics()
            assert stats["optimization"] == "5x5 grids"
            assert stats["max_configured_dimensions"] == "5x5"
            assert stats.get("is_5x5_optimized", False) is True

    def test_miniarc_parser_error_handling_oversized_grids(self, miniarc_config):
        """Test error handling for oversized grids."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "data" / "MiniARC"
            tasks_dir.mkdir(parents=True)

            # Create multiple tasks with different constraint violations
            test_cases = [
                {
                    "name": "height_violation.json",
                    "task": {
                        "train": [
                            {
                                "input": [[0], [1], [0], [1], [0], [1]],  # 6x1 (height > 5)
                                "output": [[1], [0], [1], [0], [1], [0]]
                            }
                        ],
                        "test": [{"input": [[0], [1]]}]
                    }
                },
                {
                    "name": "width_violation.json", 
                    "task": {
                        "train": [
                            {
                                "input": [[0, 1, 0, 1, 0, 1]],  # 1x6 (width > 5)
                                "output": [[1, 0, 1, 0, 1, 0]]
                            }
                        ],
                        "test": [{"input": [[0, 1, 0, 1, 0, 1]]}]
                    }
                },
                {
                    "name": "both_violation.json",
                    "task": {
                        "train": [
                            {
                                "input": [[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], 
                                         [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
                                         [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]],  # 6x6 (both > 5)
                                "output": [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]
                            }
                        ],
                        "test": [{"input": [[0, 1]]}]
                    }
                }
            ]

            # Write test cases to files
            for test_case in test_cases:
                task_file = tasks_dir / test_case["name"]
                with task_file.open("w") as f:
                    json.dump(test_case["task"], f)

            # Update config to point to temp directory
            miniarc_config.tasks.path = str(tasks_dir)

            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = Path(miniarc_config.tasks.path)
                
                with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
                    parser = MiniArcParser(miniarc_config)

                    # Should have logged specific errors for each constraint violation
                    error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
                    
                    # Check for specific error messages
                    height_errors = [call for call in error_calls if "6x1" in call and "exceeds MiniARC 5x5 constraint" in call]
                    width_errors = [call for call in error_calls if "1x6" in call and "exceeds MiniARC 5x5 constraint" in call]
                    both_errors = [call for call in error_calls if "6x6" in call and "exceeds MiniARC 5x5 constraint" in call]
                    
                    assert len(height_errors) > 0, "Should log height constraint violation"
                    assert len(width_errors) > 0, "Should log width constraint violation"  
                    assert len(both_errors) > 0, "Should log both dimensions constraint violation"

                    # None of the invalid tasks should be loaded
                    task_ids = parser.get_available_task_ids()
                    assert len(task_ids) == 0

    def test_miniarc_parser_task_preprocessing(self, miniarc_config, sample_miniarc_task):
        """Test task data preprocessing with MiniARC-specific optimizations."""
        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            # Create parser with empty directory (to avoid loading)
            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                parser = MiniArcParser(miniarc_config)

        key = jax.random.PRNGKey(0)
        task_id = "test_miniarc_task"

        # Test preprocessing
        jax_task = parser.preprocess_task_data((task_id, sample_miniarc_task), key)

        assert isinstance(jax_task, JaxArcTask)
        assert jax_task.num_train_pairs == 2
        assert jax_task.num_test_pairs == 1

        # Check optimized array shapes (5x5 instead of 30x30)
        assert jax_task.input_grids_examples.shape == (3, 5, 5)  # max_train_pairs, max_h, max_w
        assert jax_task.test_input_grids.shape == (1, 5, 5)  # max_test_pairs, max_h, max_w

        # Check data types
        assert jax_task.input_grids_examples.dtype == jnp.int32
        assert jax_task.input_masks_examples.dtype == jnp.bool_

        # Check that task index was created
        assert jax_task.task_index.dtype == jnp.int32

        # Verify that grids are properly padded and masked
        # First training pair should have valid data
        assert jax_task.input_masks_examples[0].sum() > 0  # Should have valid cells
        assert jax_task.output_masks_examples[0].sum() > 0

        # Unused training pair slots should be masked as invalid
        if jax_task.num_train_pairs < 3:
            assert jax_task.input_masks_examples[2].sum() == 0  # Should be all False

    def test_miniarc_parser_get_random_task(self, miniarc_config, mock_miniarc_directory):
        """Test getting random task from MiniARC dataset."""
        # Update config to point to mock directory
        miniarc_config.tasks.path = str(Path(mock_miniarc_directory) / "data" / "MiniARC")

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(miniarc_config.tasks.path)

            parser = MiniArcParser(miniarc_config)
            key = jax.random.PRNGKey(123)

            # Test getting random task
            task = parser.get_random_task(key)
            assert isinstance(task, JaxArcTask)
            assert task.num_train_pairs == 2
            assert task.num_test_pairs == 1

            # Test multiple random selections
            keys = jax.random.split(key, 10)
            tasks = [parser.get_random_task(k) for k in keys]
            
            # All should be valid JaxArcTask instances
            assert all(isinstance(task, JaxArcTask) for task in tasks)
            
            # Should potentially get different tasks (though not guaranteed with small dataset)
            task_indices = [task.task_index for task in tasks]
            assert all(isinstance(idx, jnp.ndarray) for idx in task_indices)

    def test_miniarc_parser_get_task_by_id(self, miniarc_config, mock_miniarc_directory):
        """Test getting specific task by ID."""
        # Update config to point to mock directory
        miniarc_config.tasks.path = str(Path(mock_miniarc_directory) / "data" / "MiniARC")

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(miniarc_config.tasks.path)

            parser = MiniArcParser(miniarc_config)

            # Get available task IDs
            task_ids = parser.get_available_task_ids()
            assert len(task_ids) > 0

            # Test getting specific task
            task_id = task_ids[0]
            task = parser.get_task_by_id(task_id)
            assert isinstance(task, JaxArcTask)

            # Test getting all tasks by ID
            for task_id in task_ids:
                task = parser.get_task_by_id(task_id)
                assert isinstance(task, JaxArcTask)
                assert task.num_train_pairs == 2
                assert task.num_test_pairs == 1

            # Test invalid task ID
            with pytest.raises(ValueError, match="Task ID 'invalid_task' not found in MiniARC dataset"):
                parser.get_task_by_id("invalid_task")

    def test_miniarc_parser_empty_directory(self, miniarc_config):
        """Test handling of empty or missing tasks directory."""
        miniarc_config.tasks.path = "/nonexistent/path"

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/nonexistent/path")

            # Should not raise exception during initialization, but log warning
            with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
                parser = MiniArcParser(miniarc_config)

                # Should have logged warning about missing directory
                mock_logger.warning.assert_called()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "MiniARC tasks directory not found" in warning_call

            # Should have no tasks
            assert len(parser.get_available_task_ids()) == 0

            # Should raise error when trying to get random task
            key = jax.random.PRNGKey(0)
            with pytest.raises(RuntimeError, match="No tasks available in MiniARC dataset"):
                parser.get_random_task(key)

    def test_miniarc_parser_load_task_file(self, miniarc_config, sample_miniarc_task):
        """Test loading task file directly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_miniarc_task, f)
            temp_file = f.name

        try:
            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = Path("/mock/path")

                with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                    parser = MiniArcParser(miniarc_config)

                # Test loading valid file
                task_data = parser.load_task_file(temp_file)
                assert task_data == sample_miniarc_task

                # Test loading nonexistent file
                with pytest.raises(FileNotFoundError, match="Task file not found"):
                    parser.load_task_file("/nonexistent/file.json")

        finally:
            Path(temp_file).unlink()

    def test_miniarc_parser_invalid_json_file(self, miniarc_config):
        """Test handling of invalid JSON files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            temp_file = f.name

        try:
            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = Path("/mock/path")

                with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                    parser = MiniArcParser(miniarc_config)

                # Test loading invalid JSON file
                with pytest.raises(ValueError, match="Invalid JSON in file"):
                    parser.load_task_file(temp_file)

        finally:
            Path(temp_file).unlink()

    def test_miniarc_parser_dataset_statistics(self, miniarc_config, mock_miniarc_directory):
        """Test dataset statistics calculation."""
        # Update config to point to mock directory
        miniarc_config.tasks.path = str(Path(mock_miniarc_directory) / "data" / "MiniARC")

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(miniarc_config.tasks.path)

            parser = MiniArcParser(miniarc_config)

            # Test dataset statistics
            stats = parser.get_dataset_statistics()
            
            # Basic statistics
            assert stats["total_tasks"] == 5
            assert stats["optimization"] == "5x5 grids"
            assert stats["max_configured_dimensions"] == "5x5"

            # Training and test pair statistics
            assert "train_pairs" in stats
            assert "test_pairs" in stats
            assert stats["train_pairs"]["min"] == 2
            assert stats["train_pairs"]["max"] == 2
            assert stats["test_pairs"]["min"] == 1
            assert stats["test_pairs"]["max"] == 1

            # Grid dimension statistics
            assert "grid_dimensions" in stats
            grid_dims = stats["grid_dimensions"]
            assert grid_dims["max_height"] <= 5
            assert grid_dims["max_width"] <= 5
            assert grid_dims["avg_height"] <= 5
            assert grid_dims["avg_width"] <= 5

            # Optimization validation
            assert stats["is_5x5_optimized"] is True
            assert "warning" not in stats  # No warning for properly sized grids

    def test_miniarc_parser_dataset_statistics_empty(self, miniarc_config):
        """Test dataset statistics with empty dataset."""
        miniarc_config.tasks.path = "/nonexistent/path"

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/nonexistent/path")

            with patch("jaxarc.parsers.mini_arc.logger"):
                parser = MiniArcParser(miniarc_config)

            # Test statistics for empty dataset
            stats = parser.get_dataset_statistics()
            assert stats["total_tasks"] == 0
            assert stats["optimization"] == "5x5 grids"
            assert stats["max_configured_dimensions"] == "5x5"

    def test_miniarc_parser_grid_validation(self, miniarc_config):
        """Test grid validation functionality."""
        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                parser = MiniArcParser(miniarc_config)

        # Test valid grid (within color range)
        valid_grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        parser._validate_grid_colors(valid_grid)  # Should not raise

        # Test invalid grid (color out of range)
        invalid_grid = jnp.array([[0, 1, 15]], dtype=jnp.int32)  # 15 > max_colors
        with pytest.raises(ValueError, match="Invalid color in grid"):
            parser._validate_grid_colors(invalid_grid)

    def test_miniarc_parser_task_structure_validation(self, miniarc_config):
        """Test task structure validation."""
        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                parser = MiniArcParser(miniarc_config)

        # Test valid task structure
        valid_task = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test": [{"input": [[0, 2]]}]
        }
        parser._validate_task_structure(valid_task, "test_task")  # Should not raise

        # Test invalid task structures
        invalid_tasks = [
            # Missing train section
            {"test": [{"input": [[0, 1]]}]},
            # Missing test section  
            {"train": [{"input": [[0, 1]], "output": [[1, 0]]}]},
            # Empty train section
            {"train": [], "test": [{"input": [[0, 1]]}]},
            # Empty test section
            {"train": [{"input": [[0, 1]], "output": [[1, 0]]}], "test": []},
            # Non-dict structure
            "invalid_structure",
        ]

        for i, invalid_task in enumerate(invalid_tasks):
            with pytest.raises(ValueError):
                parser._validate_task_structure(invalid_task, f"invalid_task_{i}")

    def test_miniarc_parser_missing_config_path(self):
        """Test handling of missing tasks path in configuration."""
        config_without_path = DictConfig({
            "grid": {
                "max_grid_height": 5,
                "max_grid_width": 5,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 3,
            "max_test_pairs": 1,
            # Missing tasks.path
        })

        with pytest.raises(ValueError, match="MiniARC tasks path not specified in configuration"):
            MiniArcParser(config_without_path)

    def test_miniarc_parser_grid_size_validation_methods(self, miniarc_config):
        """Test grid size validation helper methods."""
        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                parser = MiniArcParser(miniarc_config)

        # Test valid grid sizes
        valid_grids = [
            [[0, 1, 2, 3, 4]],  # 1x5
            [[0], [1], [2], [3], [4]],  # 5x1
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],  # 3x3
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],  # 5x5
        ]

        for i, grid in enumerate(valid_grids):
            parser._validate_grid_size(grid, f"valid_grid_{i}")  # Should not raise

        # Test invalid grid sizes
        invalid_grids = [
            [[0, 1, 2, 3, 4, 5]],  # 1x6 (width > 5)
            [[0], [1], [2], [3], [4], [5]],  # 6x1 (height > 5)
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 0, 1], [2, 3, 4, 5, 6, 7], 
             [8, 9, 0, 1, 2, 3], [4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5]],  # 6x6 (both > 5)
        ]

        for i, grid in enumerate(invalid_grids):
            with pytest.raises(ValueError, match="exceeds MiniARC 5x5 constraint"):
                parser._validate_grid_size(grid, f"invalid_grid_{i}")

        # Test empty grid (should not raise)
        parser._validate_grid_size([], "empty_grid")