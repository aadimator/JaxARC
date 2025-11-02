"""Tests for MiniARC dataset parser.

This module tests the MiniArcParser class to ensure proper loading,
5x5 grid optimization, and JAX compatibility for MiniARC datasets.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest
from chex import assert_trees_all_close

from jaxarc.configs import DatasetConfig
from jaxarc.parsers.mini_arc import MiniArcParser
from jaxarc.types import JaxArcTask


class TestMiniArcParser:
    """Test suite for MiniArcParser class."""

    @pytest.fixture
    def valid_config_5x5(self) -> DatasetConfig:
        """Provide a valid DatasetConfig optimized for 5x5 grids."""
        return DatasetConfig(
            dataset_path="test/data/mini-arc",
            max_grid_height=5,
            max_grid_width=5,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=10,
            max_test_pairs=5,
            task_split="train",
        )

    @pytest.fixture
    def valid_config_large(self) -> DatasetConfig:
        """Provide a valid DatasetConfig with larger dimensions (should trigger warning)."""
        return DatasetConfig(
            dataset_path="test/data/mini-arc",
            max_grid_height=30,
            max_grid_width=30,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=10,
            max_test_pairs=5,
            task_split="train",
        )

    @pytest.fixture
    def sample_5x5_task_data(self) -> dict:
        """Provide sample MiniARC task data with 5x5 grids."""
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
                {"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]},
            ],
            "test": [
                {
                    "input": [[4, 5, 4], [5, 4, 5], [4, 5, 4]],
                    "output": [[5, 4, 5], [4, 5, 4], [5, 4, 5]],
                }
            ],
        }

    @pytest.fixture
    def sample_oversized_task_data(self) -> dict:
        """Provide sample task data that exceeds 5x5 constraints."""
        return {
            "train": [
                {
                    "input": [
                        [0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0],
                    ],  # 2x6 - exceeds width
                    "output": [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]],
                }
            ],
            "test": [
                {
                    "input": [[2], [3], [4], [5], [6], [7]],  # 6x1 - exceeds height
                    "output": [[7], [6], [5], [4], [3], [2]],
                }
            ],
        }

    @pytest.fixture
    def temp_mini_arc_dataset(self, sample_5x5_task_data: dict) -> Path:
        """Create a temporary MiniARC dataset structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create MiniARC directory structure
            data_dir = temp_path / "data" / "MiniARC"
            data_dir.mkdir(parents=True)

            # Create sample task files with descriptive names
            task_files = {
                "flip_pattern.json": sample_5x5_task_data,
                "simple_swap.json": {
                    "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                    "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
                },
                "identity_transform.json": {
                    "train": [{"input": [[0]], "output": [[0]]}],
                    "test": [{"input": [[1]], "output": [[1]]}],
                },
                "color_increment.json": {
                    "train": [{"input": [[0, 1]], "output": [[1, 2]]}],
                    "test": [{"input": [[2, 3]], "output": [[3, 4]]}],
                },
            }

            for filename, data in task_files.items():
                task_file = data_dir / filename
                with task_file.open("w") as f:
                    json.dump(data, f)

            yield temp_path

    @pytest.fixture
    def mock_parser_5x5(
        self, valid_config_5x5: DatasetConfig, temp_mini_arc_dataset: Path
    ) -> MiniArcParser:
        """Create a MiniArcParser with 5x5 optimization and mocked data loading."""
        config = DatasetConfig(
            dataset_path=str(temp_mini_arc_dataset),
            max_grid_height=5,
            max_grid_width=5,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=10,
            max_test_pairs=5,
            task_split="train",
        )

        with patch("pyprojroot.here", return_value=temp_mini_arc_dataset):
            return MiniArcParser(config)

    def test_initialization_success_5x5_optimized(self, temp_mini_arc_dataset: Path):
        """Test successful initialization with 5x5 optimization."""
        config = DatasetConfig(
            dataset_path=str(temp_mini_arc_dataset),
            max_grid_height=5,
            max_grid_width=5,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=10,
            max_test_pairs=5,
            task_split="train",
        )

        with patch("pyprojroot.here", return_value=temp_mini_arc_dataset):
            parser = MiniArcParser(config)

            assert len(parser._task_ids) == 4
            assert "flip_pattern" in parser._task_ids
            assert "simple_swap" in parser._task_ids
            assert "identity_transform" in parser._task_ids
            assert "color_increment" in parser._task_ids

    def test_initialization_with_large_dimensions_warning(
        self, temp_mini_arc_dataset: Path, valid_config_large: DatasetConfig
    ):
        """Test initialization with large dimensions triggers warning."""
        config = DatasetConfig(
            dataset_path=str(temp_mini_arc_dataset),
            max_grid_height=30,
            max_grid_width=30,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=10,
            max_test_pairs=5,
            task_split="train",
        )

        with patch("pyprojroot.here", return_value=temp_mini_arc_dataset):
            with patch("loguru.logger.warning") as mock_warning:
                parser = MiniArcParser(config)

                # Should have logged a warning about non-optimal dimensions
                mock_warning.assert_called()
                warning_call = mock_warning.call_args[0][0]
                assert "optimized for 5x5 grids" in warning_call

    def test_initialization_missing_directory(self, valid_config_5x5: DatasetConfig):
        """Test initialization with missing data directory."""
        with patch("pyprojroot.here", return_value=Path("/nonexistent/path")):
            # With lazy loading, missing directory raises RuntimeError during scan
            with pytest.raises(RuntimeError, match="tasks directory not found"):
                MiniArcParser(valid_config_5x5)

    def test_get_data_path(self, valid_config_5x5: DatasetConfig):
        """Test get_data_path returns correct MiniARC path."""
        with patch.object(MiniArcParser, "_scan_available_tasks"):
            parser = MiniArcParser(valid_config_5x5)
            path = parser.get_data_path()
            assert path == "test/data/mini-arc/data/MiniARC"

    # Removed test_validate_grid_constraints_5x5_optimized - tests specific log message format

    def test_validate_task_structure_valid(self, mock_parser_5x5: MiniArcParser):
        """Test task structure validation with valid data."""
        valid_task = {
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
        }

        # Should not raise any exception
        mock_parser_5x5._validate_task_structure(valid_task, "test_task")

    def test_validate_task_structure_missing_train(
        self, mock_parser_5x5: MiniArcParser
    ):
        """Test task structure validation with missing train section."""
        invalid_task = {"test": [{"input": [[3, 4]], "output": [[4, 3]]}]}

        with pytest.raises(ValueError, match="Missing 'train' section"):
            mock_parser_5x5._validate_task_structure(invalid_task, "test_task")

    def test_validate_task_structure_missing_test(self, mock_parser_5x5: MiniArcParser):
        """Test task structure validation with missing test section."""
        invalid_task = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}]}

        with pytest.raises(ValueError, match="Missing 'test' section"):
            mock_parser_5x5._validate_task_structure(invalid_task, "test_task")

    def test_validate_task_structure_empty_train(self, mock_parser_5x5: MiniArcParser):
        """Test task structure validation with empty train section."""
        invalid_task = {"train": [], "test": [{"input": [[3, 4]], "output": [[4, 3]]}]}

        with pytest.raises(ValueError, match="'train' must be a non-empty list"):
            mock_parser_5x5._validate_task_structure(invalid_task, "test_task")

    def test_validate_miniarc_constraints_valid_5x5(
        self, mock_parser_5x5: MiniArcParser, sample_5x5_task_data: dict
    ):
        """Test MiniARC constraint validation with valid 5x5 data."""
        # Should not raise any exception
        mock_parser_5x5._validate_miniarc_constraints(sample_5x5_task_data, "test_task")

    def test_validate_miniarc_constraints_oversized(
        self, mock_parser_5x5: MiniArcParser, sample_oversized_task_data: dict
    ):
        """Test MiniARC constraint validation with oversized grids."""
        with pytest.raises(ValueError, match="exceeds MiniARC 5x5 constraint"):
            mock_parser_5x5._validate_miniarc_constraints(
                sample_oversized_task_data, "test_task"
            )

    def test_validate_grid_size_valid(self, mock_parser_5x5: MiniArcParser):
        """Test grid size validation with valid 5x5 grid."""
        valid_grid = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [1, 2, 3, 4, 5],
        ]

        # Should not raise any exception
        mock_parser_5x5._validate_grid_size(valid_grid, "test_grid")

    def test_validate_grid_size_oversized_height(self, mock_parser_5x5: MiniArcParser):
        """Test grid size validation with height exceeding 5."""
        oversized_grid = [[1], [2], [3], [4], [5], [6]]  # 6x1

        with pytest.raises(ValueError, match="exceeds MiniARC 5x5 constraint"):
            mock_parser_5x5._validate_grid_size(oversized_grid, "test_grid")

    def test_validate_grid_size_oversized_width(self, mock_parser_5x5: MiniArcParser):
        """Test grid size validation with width exceeding 5."""
        oversized_grid = [[1, 2, 3, 4, 5, 6]]  # 1x6

        with pytest.raises(ValueError, match="exceeds MiniARC 5x5 constraint"):
            mock_parser_5x5._validate_grid_size(oversized_grid, "test_grid")

    def test_load_task_file_success(
        self, mock_parser_5x5: MiniArcParser, temp_mini_arc_dataset: Path
    ):
        """Test successful task file loading."""
        task_file = temp_mini_arc_dataset / "data" / "MiniARC" / "flip_pattern.json"
        result = mock_parser_5x5.load_task_file(str(task_file))

        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result

    def test_load_task_file_not_found(self, mock_parser_5x5: MiniArcParser):
        """Test load_task_file with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Task file not found"):
            mock_parser_5x5.load_task_file("/nonexistent/file.json")

    def test_preprocess_task_data_with_tuple(
        self, mock_parser_5x5: MiniArcParser, sample_5x5_task_data: dict
    ):
        """Test preprocessing with (task_id, task_content) tuple."""
        key = jax.random.PRNGKey(42)
        result = mock_parser_5x5.preprocess_task_data(
            ("test_task", sample_5x5_task_data), key
        )

        assert isinstance(result, JaxArcTask)
        assert result.num_train_pairs == 2
        assert result.num_test_pairs == 1

    def test_preprocess_task_data_direct_content(
        self, mock_parser_5x5: MiniArcParser, sample_5x5_task_data: dict
    ):
        """Test preprocessing with direct task content."""
        key = jax.random.PRNGKey(42)
        result = mock_parser_5x5.preprocess_task_data(sample_5x5_task_data, key)

        assert isinstance(result, JaxArcTask)
        assert result.num_train_pairs == 2
        assert result.num_test_pairs == 1

    def test_get_random_task_success(self, mock_parser_5x5: MiniArcParser):
        """Test successful random task retrieval."""
        key = jax.random.PRNGKey(42)
        task = mock_parser_5x5.get_random_task(key)

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs >= 1
        assert task.num_test_pairs >= 1

    def test_get_random_task_no_tasks(self, valid_config_5x5: DatasetConfig):
        """Test get_random_task with no available tasks."""
        with patch.object(MiniArcParser, "_scan_available_tasks"):
            parser = MiniArcParser(valid_config_5x5)
            parser._task_ids = []  # No tasks available

            key = jax.random.PRNGKey(42)
            with pytest.raises(RuntimeError, match="No tasks available in MiniARC"):
                parser.get_random_task(key)

    def test_get_task_by_id_success(self, mock_parser_5x5: MiniArcParser):
        """Test successful task retrieval by ID."""
        task = mock_parser_5x5.get_task_by_id("flip_pattern")

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs >= 1
        assert task.num_test_pairs >= 1

    def test_get_task_by_id_not_found(self, mock_parser_5x5: MiniArcParser):
        """Test get_task_by_id with non-existent task ID."""
        with pytest.raises(ValueError, match="not found in MiniARC dataset"):
            mock_parser_5x5.get_task_by_id("nonexistent_task")

    def test_get_available_task_ids(self, mock_parser_5x5: MiniArcParser):
        """Test get_available_task_ids returns correct list."""
        task_ids = mock_parser_5x5.get_available_task_ids()

        assert isinstance(task_ids, list)
        assert len(task_ids) == 4
        assert "flip_pattern" in task_ids
        assert "simple_swap" in task_ids
        assert "identity_transform" in task_ids
        assert "color_increment" in task_ids

        # Ensure it returns a copy
        task_ids.append("new_task")
        original_ids = mock_parser_5x5.get_available_task_ids()
        assert "new_task" not in original_ids

    def test_get_dataset_statistics_with_tasks(self, mock_parser_5x5: MiniArcParser):
        """Test get_dataset_statistics with loaded tasks."""
        stats = mock_parser_5x5.get_dataset_statistics()

        assert isinstance(stats, dict)
        assert stats["total_tasks"] == 4
        assert stats["optimization"] == "5x5 grids"
        assert stats["max_configured_dimensions"] == "5x5"
        assert "train_pairs" in stats
        assert "test_pairs" in stats
        assert "grid_dimensions" in stats
        assert "is_5x5_optimized" in stats

    def test_get_dataset_statistics_empty(self, valid_config_5x5: DatasetConfig):
        """Test get_dataset_statistics with no tasks."""
        with patch.object(MiniArcParser, "_scan_available_tasks"):
            parser = MiniArcParser(valid_config_5x5)
            parser._task_ids = []  # No tasks

            stats = parser.get_dataset_statistics()

            assert stats["total_tasks"] == 0
            assert stats["optimization"] == "5x5 grids"
            assert stats["max_configured_dimensions"] == "5x5"

    def test_jax_compatibility(self, mock_parser_5x5: MiniArcParser):
        """Test that parser output is JAX-compatible."""
        key = jax.random.PRNGKey(42)
        task = mock_parser_5x5.get_random_task(key)

        # Test that arrays can be used in JAX operations
        train_sum = jnp.sum(task.input_grids_examples)
        test_sum = jnp.sum(task.test_input_grids)

        assert isinstance(train_sum, jnp.ndarray)
        assert isinstance(test_sum, jnp.ndarray)

        # Test JIT compilation
        @jax.jit
        def process_task(task_data):
            return jnp.sum(task_data.input_grids_examples)

        result = process_task(task)
        assert isinstance(result, jnp.ndarray)

    def test_array_shapes_and_padding_5x5(self, mock_parser_5x5: MiniArcParser):
        """Test that arrays are properly padded for 5x5 optimization."""
        task = mock_parser_5x5.get_task_by_id("flip_pattern")

        # Check that arrays have expected 5x5 dimensions
        max_height = 5
        max_width = 5
        max_train_pairs = mock_parser_5x5.max_train_pairs
        max_test_pairs = mock_parser_5x5.max_test_pairs

        assert task.input_grids_examples.shape == (
            max_train_pairs,
            max_height,
            max_width,
        )
        assert task.output_grids_examples.shape == (
            max_train_pairs,
            max_height,
            max_width,
        )
        assert task.test_input_grids.shape == (max_test_pairs, max_height, max_width)
        assert task.true_test_output_grids.shape == (
            max_test_pairs,
            max_height,
            max_width,
        )

    def test_5x5_optimization_detection(self, mock_parser_5x5: MiniArcParser):
        """Test that 5x5 optimization is correctly detected in statistics."""
        stats = mock_parser_5x5.get_dataset_statistics()

        # Should detect that dataset is 5x5 optimized
        assert stats["is_5x5_optimized"] is True
        assert "warning" not in stats

    def test_error_handling_corrupted_json(
        self, temp_mini_arc_dataset: Path, valid_config_5x5: DatasetConfig
    ):
        """Test error handling with corrupted JSON files."""
        # Create a corrupted JSON file
        data_dir = temp_mini_arc_dataset / "data" / "MiniARC"
        corrupted_file = data_dir / "corrupted.json"
        with corrupted_file.open("w") as f:
            f.write("{ invalid json")

        config = DatasetConfig(
            dataset_path=str(temp_mini_arc_dataset),
            max_grid_height=5,
            max_grid_width=5,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=10,
            max_test_pairs=5,
            task_split="train",
        )

        with patch("pyprojroot.here", return_value=temp_mini_arc_dataset):
            # With lazy loading, corrupted files are detected during scan (all .json files found)
            # but error occurs when trying to load the corrupted task
            parser = MiniArcParser(config)
            assert (
                len(parser._task_ids) == 5
            )  # All 5 JSON files scanned, including corrupted

            # Attempting to load the corrupted task should raise an error
            with pytest.raises(ValueError, match="Invalid JSON"):
                parser.get_task_by_id("corrupted")

    def test_deterministic_preprocessing(self, mock_parser_5x5: MiniArcParser):
        """Test that preprocessing is deterministic for the same input."""
        # Lazy loading: load task first to populate cache
        # Use a task that exists in the fixture
        available_ids = mock_parser_5x5.get_available_task_ids()
        test_task_id = available_ids[0]  # Use first available task
        _ = mock_parser_5x5.get_task_by_id(test_task_id)
        # Access cached task data
        task_data = mock_parser_5x5._cached_tasks[test_task_id]

        # Process the same data multiple times
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(42)  # Same key

        result1 = mock_parser_5x5.preprocess_task_data(("test_task", task_data), key1)
        result2 = mock_parser_5x5.preprocess_task_data(("test_task", task_data), key2)

        # Results should be identical
        assert_trees_all_close(
            result1.input_grids_examples, result2.input_grids_examples
        )
        assert_trees_all_close(
            result1.output_grids_examples, result2.output_grids_examples
        )
        assert result1.num_train_pairs == result2.num_train_pairs
        assert result1.num_test_pairs == result2.num_test_pairs

    def test_from_hydra_class_method(self, valid_config_5x5: DatasetConfig):
        """Test from_hydra class method creates parser correctly."""
        mock_hydra_config = Mock()

        with patch.object(DatasetConfig, "from_hydra", return_value=valid_config_5x5):
            with patch.object(MiniArcParser, "_scan_available_tasks"):
                parser = MiniArcParser.from_hydra(mock_hydra_config)

                assert isinstance(parser, MiniArcParser)
                assert parser.config == valid_config_5x5

    def test_descriptive_task_names(self, mock_parser_5x5: MiniArcParser):
        """Test that MiniARC uses descriptive task names from filenames."""
        task_ids = mock_parser_5x5.get_available_task_ids()

        # Task IDs should be descriptive filenames without extension
        descriptive_names = [
            "flip_pattern",
            "simple_swap",
            "identity_transform",
            "color_increment",
        ]
        for name in descriptive_names:
            assert name in task_ids

    def test_flat_directory_structure_handling(
        self, temp_mini_arc_dataset: Path, valid_config_5x5: DatasetConfig
    ):
        """Test that parser correctly handles flat directory structure."""
        # Add a subdirectory with JSON files (should be ignored)
        subdir = temp_mini_arc_dataset / "data" / "MiniARC" / "subdir"
        subdir.mkdir()
        subdir_file = subdir / "ignored_task.json"
        with subdir_file.open("w") as f:
            json.dump(
                {
                    "train": [{"input": [[1]], "output": [[2]]}],
                    "test": [{"input": [[3]], "output": [[4]]}],
                },
                f,
            )

        config = DatasetConfig(
            dataset_path=str(temp_mini_arc_dataset),
            max_grid_height=5,
            max_grid_width=5,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=10,
            max_test_pairs=5,
            task_split="train",
        )

        with patch("pyprojroot.here", return_value=temp_mini_arc_dataset):
            parser = MiniArcParser(config)

            # Should only load files from the flat directory, not subdirectories
            assert len(parser._task_ids) == 4
            assert "ignored_task" not in parser._task_ids
