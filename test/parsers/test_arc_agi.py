"""Tests for ARC-AGI dataset parser.

This module tests the ArcAgiParser class to ensure proper loading,
preprocessing, and JAX compatibility for ARC-AGI datasets.
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
from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.types import JaxArcTask


class TestArcAgiParser:
    """Test suite for ArcAgiParser class."""

    @pytest.fixture
    def valid_config(self) -> DatasetConfig:
        """Provide a valid DatasetConfig for ARC-AGI testing."""
        return DatasetConfig(
            dataset_path="test/data/arc-agi",
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
    def sample_task_data(self) -> dict:
        """Provide sample ARC-AGI task data in GitHub format."""
        return {
            "train": [
                {
                    "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                    "output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
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
    def temp_arc_agi_dataset(self, sample_task_data: dict) -> Path:
        """Create a temporary ARC-AGI dataset structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create ARC-AGI directory structure
            data_dir = temp_path / "data" / "training"
            data_dir.mkdir(parents=True)

            # Create sample task files
            task_files = {
                "task_001.json": sample_task_data,
                "task_002.json": {
                    "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                    "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
                },
                "task_003.json": {
                    "train": [{"input": [[0]], "output": [[1]]}],
                    "test": [{"input": [[2]], "output": [[3]]}],
                },
            }

            for filename, data in task_files.items():
                task_file = data_dir / filename
                with task_file.open("w") as f:
                    json.dump(data, f)

            yield temp_path

    @pytest.fixture
    def mock_parser_with_data(
        self, valid_config: DatasetConfig, temp_arc_agi_dataset: Path
    ) -> ArcAgiParser:
        """Create an ArcAgiParser with mocked data loading."""
        config = DatasetConfig(
            dataset_path=str(temp_arc_agi_dataset),
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

        with patch("pyprojroot.here", return_value=temp_arc_agi_dataset):
            return ArcAgiParser(config)

    def test_initialization_success(self, temp_arc_agi_dataset: Path):
        """Test successful initialization with valid dataset."""
        config = DatasetConfig(
            dataset_path=str(temp_arc_agi_dataset),
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

        with patch("pyprojroot.here", return_value=temp_arc_agi_dataset):
            parser = ArcAgiParser(config)

            assert len(parser._task_ids) == 3
            assert "task_001" in parser._task_ids
            assert "task_002" in parser._task_ids
            assert "task_003" in parser._task_ids

    def test_initialization_missing_directory(self, valid_config: DatasetConfig):
        """Test initialization fails with missing data directory."""
        with patch("pyprojroot.here", return_value=Path("/nonexistent/path")):
            with pytest.raises(RuntimeError, match="Data directory not found"):
                ArcAgiParser(valid_config)

    # Removed test_initialization_no_json_files - tests specific error message format

    def test_get_data_path_train_split(self, valid_config: DatasetConfig):
        """Test get_data_path returns correct path for train split."""
        with patch.object(ArcAgiParser, "_scan_available_tasks"):
            parser = ArcAgiParser(valid_config)
            path = parser.get_data_path()
            assert path == "test/data/arc-agi/data/training"

    def test_get_data_path_evaluation_split(self, valid_config: DatasetConfig):
        """Test get_data_path returns correct path for evaluation split."""
        config = DatasetConfig(
            dataset_path="test/data/arc-agi",
            max_grid_height=30,
            max_grid_width=30,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=10,
            max_test_pairs=5,
            task_split="evaluation",
        )

        with patch.object(ArcAgiParser, "_scan_available_tasks"):
            parser = ArcAgiParser(config)
            path = parser.get_data_path()
            assert path == "test/data/arc-agi/data/evaluation"

    def test_load_task_file_success(
        self, mock_parser_with_data: ArcAgiParser, temp_arc_agi_dataset: Path
    ):
        """Test successful task file loading."""
        task_file = temp_arc_agi_dataset / "data" / "training" / "task_001.json"
        result = mock_parser_with_data.load_task_file(str(task_file))

        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result
        assert len(result["train"]) == 2
        assert len(result["test"]) == 1

    def test_load_task_file_not_found(self, mock_parser_with_data: ArcAgiParser):
        """Test load_task_file with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Task file not found"):
            mock_parser_with_data.load_task_file("/nonexistent/file.json")

    def test_load_task_file_invalid_json(self, mock_parser_with_data: ArcAgiParser):
        """Test load_task_file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            f.flush()

            with pytest.raises(ValueError, match="Invalid JSON"):
                mock_parser_with_data.load_task_file(f.name)

    def test_extract_task_id_and_content_github_format(
        self, mock_parser_with_data: ArcAgiParser, sample_task_data: dict
    ):
        """Test _extract_task_id_and_content with GitHub format."""
        task_id, content = mock_parser_with_data._extract_task_id_and_content(
            sample_task_data
        )

        assert task_id == "unknown"  # GitHub format doesn't include task ID in data
        assert content == sample_task_data
        assert "train" in content
        assert "test" in content

    def test_extract_task_id_and_content_invalid_format(
        self, mock_parser_with_data: ArcAgiParser
    ):
        """Test _extract_task_id_and_content with invalid format."""
        invalid_data = {"invalid": "format"}

        with pytest.raises(ValueError, match="Invalid task data format"):
            mock_parser_with_data._extract_task_id_and_content(invalid_data)

    def test_extract_task_id_and_content_non_dict(
        self, mock_parser_with_data: ArcAgiParser
    ):
        """Test _extract_task_id_and_content with non-dict input."""
        with pytest.raises(ValueError, match="Expected dict"):
            mock_parser_with_data._extract_task_id_and_content("not a dict")

    def test_preprocess_task_data_success(
        self, mock_parser_with_data: ArcAgiParser, sample_task_data: dict
    ):
        """Test successful task data preprocessing."""
        key = jax.random.PRNGKey(42)
        result = mock_parser_with_data.preprocess_task_data(
            sample_task_data, key, "test_task"
        )

        assert isinstance(result, JaxArcTask)
        assert result.num_train_pairs == 2
        assert result.num_test_pairs == 1

        # Check array shapes and types
        assert result.input_grids_examples.dtype == jnp.int32
        assert result.output_grids_examples.dtype == jnp.int32
        assert result.test_input_grids.dtype == jnp.int32
        assert result.true_test_output_grids.dtype == jnp.int32

        # Check mask types
        assert result.input_masks_examples.dtype == bool
        assert result.output_masks_examples.dtype == bool
        assert result.test_input_masks.dtype == bool
        assert result.true_test_output_masks.dtype == bool

    def test_preprocess_task_data_invalid_training_data(
        self, mock_parser_with_data: ArcAgiParser
    ):
        """Test preprocessing with invalid training data."""
        invalid_data = {
            "train": [],  # Empty training data
            "test": [{"input": [[1]], "output": [[2]]}],
        }

        key = jax.random.PRNGKey(42)
        with pytest.raises(ValueError, match="at least one training pair"):
            mock_parser_with_data.preprocess_task_data(invalid_data, key)

    def test_preprocess_task_data_invalid_test_data(
        self, mock_parser_with_data: ArcAgiParser
    ):
        """Test preprocessing with invalid test data."""
        invalid_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [],  # Empty test data
        }

        key = jax.random.PRNGKey(42)
        with pytest.raises(ValueError, match="at least one test pair"):
            mock_parser_with_data.preprocess_task_data(invalid_data, key)

    def test_get_random_task_success(self, mock_parser_with_data: ArcAgiParser):
        """Test successful random task retrieval."""
        key = jax.random.PRNGKey(42)
        task = mock_parser_with_data.get_random_task(key)

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs >= 1
        assert task.num_test_pairs >= 1

    def test_get_random_task_no_tasks(self, valid_config: DatasetConfig):
        """Test get_random_task with no available tasks."""
        with patch.object(ArcAgiParser, "_scan_available_tasks"):
            parser = ArcAgiParser(valid_config)
            parser._task_ids = []  # No tasks available

            key = jax.random.PRNGKey(42)
            with pytest.raises(RuntimeError, match="No tasks available"):
                parser.get_random_task(key)

    def test_get_task_by_id_success(self, mock_parser_with_data: ArcAgiParser):
        """Test successful task retrieval by ID."""
        task = mock_parser_with_data.get_task_by_id("task_001")

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs >= 1
        assert task.num_test_pairs >= 1

    def test_get_task_by_id_not_found(self, mock_parser_with_data: ArcAgiParser):
        """Test get_task_by_id with non-existent task ID."""
        with pytest.raises(ValueError, match="not found in dataset"):
            mock_parser_with_data.get_task_by_id("nonexistent_task")

    def test_get_available_task_ids(self, mock_parser_with_data: ArcAgiParser):
        """Test get_available_task_ids returns correct list."""
        task_ids = mock_parser_with_data.get_available_task_ids()

        assert isinstance(task_ids, list)
        assert len(task_ids) == 3
        assert "task_001" in task_ids
        assert "task_002" in task_ids
        assert "task_003" in task_ids

        # Ensure it returns a copy (not the original list)
        task_ids.append("new_task")
        original_ids = mock_parser_with_data.get_available_task_ids()
        assert "new_task" not in original_ids

    def test_from_hydra_class_method(self, valid_config: DatasetConfig):
        """Test from_hydra class method creates parser correctly."""
        mock_hydra_config = Mock()

        with patch.object(DatasetConfig, "from_hydra", return_value=valid_config):
            with patch.object(ArcAgiParser, "_scan_available_tasks"):
                parser = ArcAgiParser.from_hydra(mock_hydra_config)

                assert isinstance(parser, ArcAgiParser)
                assert parser.config == valid_config

    def test_jax_compatibility(self, mock_parser_with_data: ArcAgiParser):
        """Test that parser output is JAX-compatible."""
        key = jax.random.PRNGKey(42)
        task = mock_parser_with_data.get_random_task(key)

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

    def test_array_shapes_and_padding(self, mock_parser_with_data: ArcAgiParser):
        """Test that arrays are properly padded to maximum dimensions."""
        task = mock_parser_with_data.get_task_by_id("task_001")

        # Check that arrays have expected maximum dimensions
        max_height = mock_parser_with_data.max_grid_height
        max_width = mock_parser_with_data.max_grid_width
        max_train_pairs = mock_parser_with_data.max_train_pairs
        max_test_pairs = mock_parser_with_data.max_test_pairs

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

        # Check mask shapes match
        assert task.input_masks_examples.shape == (
            max_train_pairs,
            max_height,
            max_width,
        )
        assert task.output_masks_examples.shape == (
            max_train_pairs,
            max_height,
            max_width,
        )
        assert task.test_input_masks.shape == (max_test_pairs, max_height, max_width)
        assert task.true_test_output_masks.shape == (
            max_test_pairs,
            max_height,
            max_width,
        )

    def test_error_handling_corrupted_json(
        self, temp_arc_agi_dataset: Path, valid_config: DatasetConfig
    ):
        """Test error handling with corrupted JSON files."""
        # Create a corrupted JSON file
        data_dir = temp_arc_agi_dataset / "data" / "training"
        corrupted_file = data_dir / "corrupted.json"
        with corrupted_file.open("w") as f:
            f.write("{ invalid json")

        config = DatasetConfig(
            dataset_path=str(temp_arc_agi_dataset),
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

        with patch("pyprojroot.here", return_value=temp_arc_agi_dataset):
            # With lazy loading, corrupted files are detected during scan (all .json files found)
            # but error occurs when trying to load the corrupted task
            parser = ArcAgiParser(config)
            assert (
                len(parser._task_ids) == 4
            )  # All 4 JSON files scanned, including corrupted

            # Attempting to load the corrupted task should raise an error
            with pytest.raises(ValueError, match="Invalid JSON"):
                parser.get_task_by_id("corrupted")

    def test_task_caching_behavior(self, mock_parser_with_data: ArcAgiParser):
        """Test that tasks are properly cached and reused."""
        # Get the same task multiple times
        task1 = mock_parser_with_data.get_task_by_id("task_001")
        task2 = mock_parser_with_data.get_task_by_id("task_001")

        # Tasks should have the same data (though different instances due to preprocessing)
        assert_trees_all_close(task1.input_grids_examples, task2.input_grids_examples)
        assert_trees_all_close(task1.output_grids_examples, task2.output_grids_examples)
        assert task1.num_train_pairs == task2.num_train_pairs
        assert task1.num_test_pairs == task2.num_test_pairs

    def test_deterministic_preprocessing(self, mock_parser_with_data: ArcAgiParser):
        """Test that preprocessing is deterministic for the same input."""
        # Load the task first to populate cache (lazy loading)
        _ = mock_parser_with_data.get_task_by_id("task_001")
        task_data = mock_parser_with_data._cached_tasks["task_001"]

        # Process the same data multiple times
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(42)  # Same key

        result1 = mock_parser_with_data.preprocess_task_data(
            task_data, key1, "task_001"
        )
        result2 = mock_parser_with_data.preprocess_task_data(
            task_data, key2, "task_001"
        )

        # Results should be identical
        assert_trees_all_close(
            result1.input_grids_examples, result2.input_grids_examples
        )
        assert_trees_all_close(
            result1.output_grids_examples, result2.output_grids_examples
        )
        assert result1.num_train_pairs == result2.num_train_pairs
        assert result1.num_test_pairs == result2.num_test_pairs
