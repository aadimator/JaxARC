"""Comprehensive tests for MiniArcParser functionality.

This test suite covers 5x5 grid optimization, dataset handling, task creation,
error handling for oversized grids, and performance validation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.parsers.mini_arc import MiniArcParser
from jaxarc.types import JaxArcTask


class TestMiniArcParserComprehensive:
    """Comprehensive test suite for MiniArcParser."""

    @pytest.fixture
    def optimal_config(self):
        """Optimal configuration for MiniArcParser (5x5 grids)."""
        return DictConfig(
            {
                "max_grid_height": 5,
                "max_grid_width": 5,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
                "max_train_pairs": 3,
                "max_test_pairs": 1,
                "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
            }
        )

    @pytest.fixture
    def suboptimal_config(self):
        """Suboptimal configuration for MiniArcParser (larger grids)."""
        return DictConfig(
            {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
                "max_train_pairs": 4,
                "max_test_pairs": 3,
                "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
            }
        )

    @pytest.fixture
    def sample_5x5_task(self):
        """Sample MiniARC task with 5x5 grids."""
        return {
            "train": [
                {
                    "input": [
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0],
                    ],
                    "output": [
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                    ],
                },
                {
                    "input": [[0, 2, 0], [2, 0, 2], [0, 2, 0]],
                    "output": [[2, 0, 2], [0, 2, 0], [2, 0, 2]],
                },
            ],
            "test": [
                {
                    "input": [[3, 4, 3], [4, 5, 4], [3, 4, 3]],
                    "output": [[4, 3, 4], [3, 5, 3], [4, 3, 4]],
                }
            ],
        }

    @pytest.fixture
    def oversized_task(self):
        """Task with grids exceeding 5x5 constraint."""
        return {
            "train": [
                {
                    "input": [
                        [0, 1, 0, 1, 0, 1],  # 6x6 grid
                        [1, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0],
                    ],
                    "output": [
                        [1, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 0, 1],
                    ],
                }
            ],
            "test": [
                {
                    "input": [[0, 1, 0, 1, 0, 1]]  # 1x6 grid
                }
            ],
        }

    @pytest.fixture
    def temp_miniarc_directory(self, sample_5x5_task):
        """Create temporary MiniARC directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "data" / "MiniARC"
            tasks_dir.mkdir(parents=True)

            # Create sample task files
            task_files = {
                "pattern_reversal_001.json": sample_5x5_task,
                "color_swap_002.json": {
                    "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
                    "test": [{"input": [[2, 3]]}],
                },
                "symmetry_test_003.json": {
                    "train": [
                        {"input": [[1, 2, 1]], "output": [[2, 1, 2]]},
                        {"input": [[3, 4]], "output": [[4, 3]]},
                    ],
                    "test": [{"input": [[5, 6, 5]]}],
                },
                "rotation_task_004.json": {
                    "train": [{"input": [[7, 8, 9]], "output": [[9, 8, 7]]}],
                    "test": [{"input": [[1, 2, 3]], "output": [[3, 2, 1]]}],
                },
            }

            for filename, data in task_files.items():
                with (tasks_dir / filename).open("w") as f:
                    json.dump(data, f)

            yield temp_dir

    def test_initialization_optimal_config(
        self, optimal_config, temp_miniarc_directory
    ):
        """Test initialization with optimal 5x5 configuration."""
        optimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(optimal_config.tasks.path)

            parser = MiniArcParser(optimal_config)

            # Check that tasks were loaded
            task_ids = parser.get_available_task_ids()
            assert len(task_ids) == 4

            expected_ids = {
                "pattern_reversal_001",
                "color_swap_002",
                "symmetry_test_003",
                "rotation_task_004",
            }
            assert set(task_ids) == expected_ids

    def test_initialization_suboptimal_config_warning(
        self, suboptimal_config, temp_miniarc_directory
    ):
        """Test initialization with suboptimal configuration logs warning."""
        suboptimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(suboptimal_config.tasks.path)

            with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
                parser = MiniArcParser(suboptimal_config)

                # Should log warning about suboptimal configuration
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "MiniARC is optimized for 5x5 grids" in warning_msg
                assert "30x30" in warning_msg

                # Should still load tasks
                task_ids = parser.get_available_task_ids()
                assert len(task_ids) == 4

    def test_initialization_missing_tasks_path(self, optimal_config):
        """Test initialization failure when tasks path is missing."""
        del optimal_config.tasks["path"]

        with pytest.raises(ValueError, match="MiniARC tasks path not specified"):
            MiniArcParser(optimal_config)

    def test_initialization_nonexistent_directory(self, optimal_config):
        """Test initialization with non-existent tasks directory."""
        optimal_config.tasks.path = "/nonexistent/directory"

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/nonexistent/directory")

            # Should not raise exception but log warning
            with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
                parser = MiniArcParser(optimal_config)

                # Should log warning about missing directory
                mock_logger.warning.assert_called()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "MiniARC tasks directory not found" in warning_msg

                # Should have no tasks
                assert len(parser.get_available_task_ids()) == 0

    def test_5x5_constraint_validation_during_loading(self, optimal_config):
        """Test 5x5 constraint validation during task loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "data" / "MiniARC"
            tasks_dir.mkdir(parents=True)

            # Create tasks with various constraint violations
            test_cases = [
                (
                    "valid_5x5.json",
                    {
                        "train": [
                            {"input": [[1, 2, 3, 4, 5]], "output": [[5, 4, 3, 2, 1]]}
                        ],
                        "test": [{"input": [[6, 7, 8, 9, 0]]}],
                    },
                ),
                (
                    "height_violation.json",
                    {
                        "train": [
                            {
                                "input": [[1], [2], [3], [4], [5], [6]],
                                "output": [[6], [5], [4], [3], [2], [1]],
                            }
                        ],
                        "test": [{"input": [[7], [8]]}],
                    },
                ),
                (
                    "width_violation.json",
                    {
                        "train": [
                            {
                                "input": [[1, 2, 3, 4, 5, 6]],
                                "output": [[6, 5, 4, 3, 2, 1]],
                            }
                        ],
                        "test": [{"input": [[7, 8, 9, 0, 1, 2]]}],
                    },
                ),
                (
                    "both_violation.json",
                    {
                        "train": [
                            {
                                "input": [[1, 2, 3, 4, 5, 6]] * 6,
                                "output": [[6, 5, 4, 3, 2, 1]] * 6,
                            }
                        ],
                        "test": [{"input": [[7, 8, 9, 0, 1, 2]] * 6}],
                    },
                ),
            ]

            for filename, data in test_cases:
                with (tasks_dir / filename).open("w") as f:
                    json.dump(data, f)

            optimal_config.tasks.path = str(tasks_dir)

            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = tasks_dir

                with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
                    parser = MiniArcParser(optimal_config)

                    # Should only load the valid task
                    task_ids = parser.get_available_task_ids()
                    assert len(task_ids) == 1
                    assert "valid_5x5" in task_ids

                    # Should log errors for constraint violations
                    error_calls = [
                        call[0][0] for call in mock_logger.error.call_args_list
                    ]
                    constraint_errors = [
                        call
                        for call in error_calls
                        if "exceeds MiniARC 5x5 constraint" in call
                    ]
                    assert len(constraint_errors) >= 3  # At least 3 violations

    def test_task_structure_validation(self, optimal_config):
        """Test validation of task structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "data" / "MiniARC"
            tasks_dir.mkdir(parents=True)

            # Create tasks with various structure issues
            invalid_tasks = [
                ("missing_train.json", {"test": [{"input": [[1]]}]}),
                ("missing_test.json", {"train": [{"input": [[1]], "output": [[2]]}]}),
                ("empty_train.json", {"train": [], "test": [{"input": [[1]]}]}),
                (
                    "empty_test.json",
                    {"train": [{"input": [[1]], "output": [[2]]}], "test": []},
                ),
                ("non_dict.json", "invalid_structure"),
            ]

            for filename, data in invalid_tasks:
                with (tasks_dir / filename).open("w") as f:
                    json.dump(data, f)

            optimal_config.tasks.path = str(tasks_dir)

            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = tasks_dir

                with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
                    parser = MiniArcParser(optimal_config)

                    # Should have no valid tasks
                    task_ids = parser.get_available_task_ids()
                    assert len(task_ids) == 0

                    # Should log errors for each invalid task
                    error_calls = [
                        call[0][0] for call in mock_logger.error.call_args_list
                    ]
                    assert len(error_calls) >= len(invalid_tasks)

    def test_load_task_file_success(self, optimal_config, sample_5x5_task):
        """Test successful task file loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_5x5_task, f)
            temp_file = f.name

        try:
            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = Path("/mock/path")

                with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                    parser = MiniArcParser(optimal_config)

            data = parser.load_task_file(temp_file)
            assert data == sample_5x5_task

        finally:
            Path(temp_file).unlink()

    def test_load_task_file_errors(self, optimal_config):
        """Test load_task_file error handling."""
        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                parser = MiniArcParser(optimal_config)

        # Test non-existent file
        with pytest.raises(FileNotFoundError, match="Task file not found"):
            parser.load_task_file("/nonexistent/file.json")

        # Test invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                parser.load_task_file(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_preprocess_task_data_success(self, optimal_config, sample_5x5_task):
        """Test successful task data preprocessing."""
        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                parser = MiniArcParser(optimal_config)

        key = jax.random.PRNGKey(42)
        task_id = "test_miniarc_task"

        # Test with tuple input
        task = parser.preprocess_task_data((task_id, sample_5x5_task), key)

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs == 2
        assert task.num_test_pairs == 1

        # Check optimized array shapes (5x5 instead of 30x30)
        assert task.input_grids_examples.shape == (3, 5, 5)  # max_train_pairs, 5, 5
        assert task.test_input_grids.shape == (1, 5, 5)  # max_test_pairs, 5, 5

        # Test with direct task content
        task2 = parser.preprocess_task_data(sample_5x5_task, key)
        assert isinstance(task2, JaxArcTask)

    def test_preprocess_task_data_error_handling(self, optimal_config):
        """Test preprocessing error handling with MiniARC-specific messages."""
        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                parser = MiniArcParser(optimal_config)

        key = jax.random.PRNGKey(42)

        # Test empty training pairs
        invalid_data = {"train": [], "test": [{"input": [[1]]}]}
        with pytest.raises(
            ValueError, match="MiniARC task must have at least one training pair"
        ):
            parser.preprocess_task_data(invalid_data, key)

        # Test empty test pairs
        invalid_data = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        with pytest.raises(
            ValueError, match="MiniARC task must have at least one test pair"
        ):
            parser.preprocess_task_data(invalid_data, key)

    def test_get_random_task_success(self, optimal_config, temp_miniarc_directory):
        """Test successful random task retrieval."""
        optimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(optimal_config.tasks.path)

            parser = MiniArcParser(optimal_config)
            key = jax.random.PRNGKey(42)

            task = parser.get_random_task(key)
            assert isinstance(task, JaxArcTask)
            assert task.num_train_pairs > 0
            assert task.num_test_pairs > 0

    def test_get_random_task_no_tasks(self, optimal_config):
        """Test get_random_task with no available tasks."""
        optimal_config.tasks.path = "/nonexistent/path"

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/nonexistent/path")

            with patch("jaxarc.parsers.mini_arc.logger"):
                parser = MiniArcParser(optimal_config)

        key = jax.random.PRNGKey(42)
        with pytest.raises(RuntimeError, match="No tasks available in MiniARC dataset"):
            parser.get_random_task(key)

    def test_get_task_by_id_success(self, optimal_config, temp_miniarc_directory):
        """Test successful task retrieval by ID."""
        optimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(optimal_config.tasks.path)

            parser = MiniArcParser(optimal_config)

            task = parser.get_task_by_id("pattern_reversal_001")
            assert isinstance(task, JaxArcTask)
            assert task.num_train_pairs == 2
            assert task.num_test_pairs == 1

    def test_get_task_by_id_not_found(self, optimal_config, temp_miniarc_directory):
        """Test get_task_by_id with non-existent task ID."""
        optimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(optimal_config.tasks.path)

            parser = MiniArcParser(optimal_config)

            with pytest.raises(ValueError, match="not found in MiniARC dataset"):
                parser.get_task_by_id("nonexistent_task")

    def test_get_available_task_ids(self, optimal_config, temp_miniarc_directory):
        """Test getting list of available task IDs."""
        optimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(optimal_config.tasks.path)

            parser = MiniArcParser(optimal_config)
            task_ids = parser.get_available_task_ids()

            assert isinstance(task_ids, list)
            assert len(task_ids) == 4

            # Task IDs should be filename stems
            for task_id in task_ids:
                assert not task_id.endswith(".json")

    def test_dataset_statistics_optimal(self, optimal_config, temp_miniarc_directory):
        """Test dataset statistics with optimal configuration."""
        optimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(optimal_config.tasks.path)

            parser = MiniArcParser(optimal_config)
            stats = parser.get_dataset_statistics()

            # Check basic statistics
            assert stats["total_tasks"] == 4
            assert stats["optimization"] == "5x5 grids"
            assert stats["max_configured_dimensions"] == "5x5"
            assert stats["is_5x5_optimized"] is True

            # Check grid dimension statistics
            grid_dims = stats["grid_dimensions"]
            assert grid_dims["max_height"] <= 5
            assert grid_dims["max_width"] <= 5

            # Should not have warning for properly sized grids
            assert "warning" not in stats

    def test_dataset_statistics_empty(self, optimal_config):
        """Test dataset statistics with empty dataset."""
        optimal_config.tasks.path = "/nonexistent/path"

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/nonexistent/path")

            with patch("jaxarc.parsers.mini_arc.logger"):
                parser = MiniArcParser(optimal_config)

        stats = parser.get_dataset_statistics()
        assert stats["total_tasks"] == 0
        assert stats["optimization"] == "5x5 grids"
        assert stats["max_configured_dimensions"] == "5x5"

    def test_dataset_statistics_oversized_grids(self, optimal_config):
        """Test dataset statistics with oversized grids."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "data" / "MiniARC"
            tasks_dir.mkdir(parents=True)

            # Create task with oversized grid
            oversized_task = {
                "train": [
                    {"input": [[1, 2, 3, 4, 5, 6]], "output": [[6, 5, 4, 3, 2, 1]]}
                ],
                "test": [{"input": [[7, 8, 9, 0, 1, 2]]}],
            }

            with (tasks_dir / "oversized.json").open("w") as f:
                json.dump(oversized_task, f)

            optimal_config.tasks.path = str(tasks_dir)

            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = tasks_dir

                with patch("jaxarc.parsers.mini_arc.logger"):
                    parser = MiniArcParser(optimal_config)

                    # Task should be rejected, so no statistics to compute
                    stats = parser.get_dataset_statistics()
                    assert stats["total_tasks"] == 0

    def test_grid_size_validation_methods(self, optimal_config):
        """Test grid size validation helper methods."""
        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(MiniArcParser, "_load_and_cache_tasks"):
                parser = MiniArcParser(optimal_config)

        # Test valid grid sizes
        valid_grids = [
            [[0, 1, 2, 3, 4]],  # 1x5
            [[0], [1], [2], [3], [4]],  # 5x1
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],  # 3x3
            [[0, 1, 2, 3, 4]] * 5,  # 5x5
        ]

        for grid in valid_grids:
            parser._validate_grid_size(grid, "test_grid")  # Should not raise

        # Test invalid grid sizes
        invalid_grids = [
            ([[0, 1, 2, 3, 4, 5]], "1x6 grid"),  # Width > 5
            ([[0], [1], [2], [3], [4], [5]], "6x1 grid"),  # Height > 5
            ([[0, 1, 2, 3, 4, 5]] * 6, "6x6 grid"),  # Both > 5
        ]

        for grid, description in invalid_grids:
            with pytest.raises(ValueError, match="exceeds MiniARC 5x5 constraint"):
                parser._validate_grid_size(grid, description)

    def test_flat_directory_structure_handling(self, optimal_config):
        """Test handling of flat directory structure with various file types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "data" / "MiniARC"
            tasks_dir.mkdir(parents=True)

            # Create JSON files
            json_files = ["task1.json", "task2.json", "task3.json"]
            for filename in json_files:
                with (tasks_dir / filename).open("w") as f:
                    json.dump(
                        {
                            "train": [{"input": [[1]], "output": [[2]]}],
                            "test": [{"input": [[3]]}],
                        },
                        f,
                    )

            # Create non-JSON files (should be ignored)
            non_json_files = ["readme.txt", "script.py", "data.csv", "config.yaml"]
            for filename in non_json_files:
                (tasks_dir / filename).write_text("not a task file")

            optimal_config.tasks.path = str(tasks_dir)

            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = tasks_dir

                parser = MiniArcParser(optimal_config)
                task_ids = parser.get_available_task_ids()

                # Should only load JSON files
                assert len(task_ids) == 3
                expected_ids = {"task1", "task2", "task3"}
                assert set(task_ids) == expected_ids

    def test_malformed_json_handling(self, optimal_config):
        """Test handling of malformed JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "data" / "MiniARC"
            tasks_dir.mkdir(parents=True)

            # Create malformed JSON file
            with (tasks_dir / "malformed.json").open("w") as f:
                f.write("{ invalid json content")

            # Create valid JSON file
            with (tasks_dir / "valid.json").open("w") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            optimal_config.tasks.path = str(tasks_dir)

            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = tasks_dir

                with patch("jaxarc.parsers.mini_arc.logger") as mock_logger:
                    parser = MiniArcParser(optimal_config)

                    # Should skip malformed file and load valid one
                    task_ids = parser.get_available_task_ids()
                    assert len(task_ids) == 1
                    assert "valid" in task_ids

                    # Should log error for malformed file
                    error_calls = [
                        call[0][0] for call in mock_logger.error.call_args_list
                    ]
                    malformed_errors = [
                        call for call in error_calls if "malformed.json" in call
                    ]
                    assert len(malformed_errors) > 0

    def test_jax_compatibility(self, optimal_config, temp_miniarc_directory):
        """Test JAX compatibility of MiniARC parsed data."""
        optimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(optimal_config.tasks.path)

            parser = MiniArcParser(optimal_config)
            key = jax.random.PRNGKey(42)
            task = parser.get_random_task(key)

        # Test JIT compilation
        @jax.jit
        def process_miniarc_task(input_grids, input_masks):
            return jnp.sum(input_grids * input_masks)

        result = process_miniarc_task(
            task.input_grids_examples, task.input_masks_examples
        )
        assert isinstance(result, jnp.ndarray)

        # Test vmap over training pairs
        def process_single_grid(input_grid, input_mask):
            return jnp.mean(input_grid * input_mask)

        vmapped_process = jax.vmap(process_single_grid)
        results = vmapped_process(task.input_grids_examples, task.input_masks_examples)

        assert isinstance(results, jnp.ndarray)
        assert results.shape == (3,)  # max_train_pairs

    def test_performance_optimization_validation(
        self, optimal_config, temp_miniarc_directory
    ):
        """Test that MiniARC parser is optimized for performance."""
        optimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(optimal_config.tasks.path)

            parser = MiniArcParser(optimal_config)

            # Check that parser is configured for optimal performance
            assert parser.max_grid_height == 5
            assert parser.max_grid_width == 5

            # Get a task and verify optimized dimensions
            key = jax.random.PRNGKey(42)
            task = parser.get_random_task(key)

            # Arrays should be 5x5, not larger
            assert task.input_grids_examples.shape[1:] == (5, 5)
            assert task.output_grids_examples.shape[1:] == (5, 5)
            assert task.test_input_grids.shape[1:] == (5, 5)

    def test_edge_case_filenames(self, optimal_config):
        """Test handling of edge case filenames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tasks_dir = Path(temp_dir) / "data" / "MiniARC"
            tasks_dir.mkdir(parents=True)

            # Create files with various filename patterns
            edge_case_files = [
                "task-with-dashes.json",
                "task_with_underscores.json",
                "123numeric_task.json",
                "UPPERCASE_TASK.json",
                "task.with.dots.json",
                "very_long_task_name_that_exceeds_typical_limits.json",
            ]

            for filename in edge_case_files:
                with (tasks_dir / filename).open("w") as f:
                    json.dump(
                        {
                            "train": [{"input": [[1]], "output": [[2]]}],
                            "test": [{"input": [[3]]}],
                        },
                        f,
                    )

            optimal_config.tasks.path = str(tasks_dir)

            with patch("jaxarc.parsers.mini_arc.here") as mock_here:
                mock_here.return_value = tasks_dir

                parser = MiniArcParser(optimal_config)
                task_ids = parser.get_available_task_ids()

                # Should handle all edge case filenames
                assert len(task_ids) == len(edge_case_files)

                # Task IDs should be filename stems
                expected_ids = {
                    filename[:-5] for filename in edge_case_files
                }  # Remove .json
                assert set(task_ids) == expected_ids

    def test_deterministic_task_loading(self, optimal_config, temp_miniarc_directory):
        """Test that task loading is deterministic across parser instances."""
        optimal_config.tasks.path = str(
            Path(temp_miniarc_directory) / "data" / "MiniARC"
        )

        with patch("jaxarc.parsers.mini_arc.here") as mock_here:
            mock_here.return_value = Path(optimal_config.tasks.path)

            # Create two parser instances
            parser1 = MiniArcParser(optimal_config)
            parser2 = MiniArcParser(optimal_config)

            # Task IDs should be in the same order
            tasks1 = parser1.get_available_task_ids()
            tasks2 = parser2.get_available_task_ids()

            assert tasks1 == tasks2

            # Tasks should have identical content
            for task_id in tasks1:
                task1 = parser1.get_task_by_id(task_id)
                task2 = parser2.get_task_by_id(task_id)

                assert jnp.array_equal(
                    task1.input_grids_examples, task2.input_grids_examples
                )
                assert jnp.array_equal(
                    task1.output_grids_examples, task2.output_grids_examples
                )
