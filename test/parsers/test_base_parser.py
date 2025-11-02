"""Tests for the base parser abstract interface.

This module tests the ArcDataParserBase abstract class to ensure proper
interface compliance, validation, and error handling patterns.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest
from chex import assert_trees_all_close

from jaxarc.configs import DatasetConfig
from jaxarc.parsers.base_parser import ArcDataParserBase
from jaxarc.types import JaxArcTask, PRNGKey


class MockParser(ArcDataParserBase):
    """Mock implementation of ArcDataParserBase for testing."""

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self._mock_tasks = {}
        self._available_task_ids = ["task_001", "task_002", "task_003"]

    def load_task_file(self, task_file_path: str) -> Any:
        """Mock implementation that returns test data."""
        return {
            "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
            "test": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
        }

    def preprocess_task_data(self, raw_task_data: Any, key: PRNGKey) -> JaxArcTask:
        """Mock implementation that creates a simple JaxArcTask."""
        # Create simple test data
        train_inputs = jnp.array([[[0, 1], [1, 0]]], dtype=jnp.int32)
        train_outputs = jnp.array([[[1, 0], [0, 1]]], dtype=jnp.int32)
        test_inputs = jnp.array([[[0, 1], [1, 0]]], dtype=jnp.int32)
        test_outputs = jnp.array([[[1, 0], [0, 1]]], dtype=jnp.int32)

        # Create masks (all True for this simple case)
        train_input_masks = jnp.ones((1, 2, 2), dtype=bool)
        train_output_masks = jnp.ones((1, 2, 2), dtype=bool)
        test_input_masks = jnp.ones((1, 2, 2), dtype=bool)
        test_output_masks = jnp.ones((1, 2, 2), dtype=bool)

        return JaxArcTask(
            input_grids_examples=train_inputs,
            input_masks_examples=train_input_masks,
            output_grids_examples=train_outputs,
            output_masks_examples=train_output_masks,
            test_input_grids=test_inputs,
            test_input_masks=test_input_masks,
            true_test_output_grids=test_outputs,
            true_test_output_masks=test_output_masks,
            num_train_pairs=1,
            num_test_pairs=1,
            task_index=jnp.array(0, dtype=jnp.int32),
        )

    def get_random_task(self, key: PRNGKey) -> JaxArcTask:
        """Mock implementation that returns a random task."""
        raw_data = self.load_task_file("mock_path")
        return self.preprocess_task_data(raw_data, key)

    def get_task_by_id(self, task_id: str) -> JaxArcTask:
        """Mock implementation that returns a task by ID."""
        if task_id not in self._available_task_ids:
            raise ValueError(f"Task ID '{task_id}' not found")
        raw_data = self.load_task_file("mock_path")
        key = jax.random.PRNGKey(42)
        return self.preprocess_task_data(raw_data, key)

    def get_available_task_ids(self) -> list[str]:
        """Mock implementation that returns available task IDs."""
        return self._available_task_ids.copy()


class TestArcDataParserBase:
    """Test suite for ArcDataParserBase abstract class."""

    @pytest.fixture
    def valid_config(self) -> DatasetConfig:
        """Provide a valid DatasetConfig for testing."""
        return DatasetConfig(
            dataset_path="test/data",
            max_grid_height=10,
            max_grid_width=10,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=5,
            max_test_pairs=3,
            task_split="train",
        )

    @pytest.fixture
    def invalid_config(self) -> DatasetConfig:
        """Provide an invalid DatasetConfig for testing validation."""
        return DatasetConfig(
            dataset_path="",  # Invalid empty path
            max_grid_height=0,  # Invalid zero height
            max_grid_width=10,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=5,
            max_test_pairs=3,
            task_split="train",
        )

    @pytest.fixture
    def mock_parser(self, valid_config: DatasetConfig) -> MockParser:
        """Provide a mock parser instance for testing."""
        return MockParser(valid_config)

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that ArcDataParserBase cannot be instantiated directly."""
        config = DatasetConfig(
            dataset_path="test/data",
            max_grid_height=10,
            max_grid_width=10,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=5,
            max_test_pairs=3,
            task_split="train",
        )

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ArcDataParserBase(config)

    def test_initialization_with_valid_config(self, valid_config: DatasetConfig):
        """Test successful initialization with valid configuration."""
        parser = MockParser(valid_config)

        assert parser.config == valid_config
        assert parser.max_grid_height == 10
        assert parser.max_grid_width == 10
        assert parser.min_grid_height == 1
        assert parser.min_grid_width == 1
        assert parser.max_colors == 10
        assert parser.background_color == 0
        assert parser.max_train_pairs == 5
        assert parser.max_test_pairs == 3

    def test_initialization_with_invalid_config(self, invalid_config: DatasetConfig):
        """Test initialization fails with invalid configuration."""
        # The invalid_config fixture has max_grid_height=0 which should fail validation
        with pytest.raises(ValueError, match="Configuration validation failed"):
            MockParser(invalid_config)

    def test_get_data_path_default_implementation(self, mock_parser: MockParser):
        """Test default get_data_path implementation."""
        assert mock_parser.get_data_path() == "test/data"

    def test_get_max_dimensions(self, mock_parser: MockParser):
        """Test get_max_dimensions returns correct values."""
        dimensions = mock_parser.get_max_dimensions()
        expected = (
            10,
            10,
            5,
            3,
        )  # max_height, max_width, max_train_pairs, max_test_pairs
        assert dimensions == expected

    def test_get_grid_config(self, mock_parser: MockParser):
        """Test get_grid_config returns correct configuration."""
        config = mock_parser.get_grid_config()
        expected = {
            "max_grid_height": 10,
            "max_grid_width": 10,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
        }
        assert config == expected

    def test_validate_grid_dimensions_valid(self, mock_parser: MockParser):
        """Test validate_grid_dimensions with valid dimensions."""
        # Should not raise any exception
        mock_parser.validate_grid_dimensions(5, 5)
        mock_parser.validate_grid_dimensions(1, 1)  # Minimum
        mock_parser.validate_grid_dimensions(10, 10)  # Maximum

    def test_validate_grid_dimensions_too_small(self, mock_parser: MockParser):
        """Test validate_grid_dimensions with dimensions too small."""
        with pytest.raises(ValueError, match="below minimum"):
            mock_parser.validate_grid_dimensions(0, 5)

        with pytest.raises(ValueError, match="below minimum"):
            mock_parser.validate_grid_dimensions(5, 0)

    def test_validate_grid_dimensions_too_large(self, mock_parser: MockParser):
        """Test validate_grid_dimensions with dimensions too large."""
        with pytest.raises(ValueError, match="exceed maximum"):
            mock_parser.validate_grid_dimensions(11, 5)

        with pytest.raises(ValueError, match="exceed maximum"):
            mock_parser.validate_grid_dimensions(5, 11)

    def test_validate_color_value_valid(self, mock_parser: MockParser):
        """Test validate_color_value with valid colors."""
        # Should not raise any exception
        mock_parser.validate_color_value(0)
        mock_parser.validate_color_value(5)
        mock_parser.validate_color_value(9)

    def test_validate_color_value_invalid(self, mock_parser: MockParser):
        """Test validate_color_value with invalid colors."""
        with pytest.raises(ValueError, match="must be in range"):
            mock_parser.validate_color_value(-1)

        with pytest.raises(ValueError, match="must be in range"):
            mock_parser.validate_color_value(10)

    def test_convert_grid_to_jax_valid(self, mock_parser: MockParser):
        """Test _convert_grid_to_jax with valid grid data."""
        grid_data = [[0, 1, 2], [3, 4, 5]]
        result = mock_parser._convert_grid_to_jax(grid_data)

        expected = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        assert_trees_all_close(result, expected)
        assert result.dtype == jnp.int32

    def test_convert_grid_to_jax_empty_grid(self, mock_parser: MockParser):
        """Test _convert_grid_to_jax with empty grid."""
        with pytest.raises(ValueError, match="Grid data cannot be empty"):
            mock_parser._convert_grid_to_jax([])

    def test_convert_grid_to_jax_invalid_format(self, mock_parser: MockParser):
        """Test _convert_grid_to_jax with invalid format."""
        with pytest.raises(ValueError, match="Grid data must be a list"):
            mock_parser._convert_grid_to_jax("not a list")

    def test_convert_grid_to_jax_inconsistent_rows(self, mock_parser: MockParser):
        """Test _convert_grid_to_jax with inconsistent row lengths."""
        grid_data = [[0, 1], [2, 3, 4]]  # Different row lengths
        with pytest.raises(ValueError, match="same length"):
            mock_parser._convert_grid_to_jax(grid_data)

    def test_convert_grid_to_jax_invalid_cell_type(self, mock_parser: MockParser):
        """Test _convert_grid_to_jax with invalid cell types."""
        grid_data = [[0, 1], [2, "invalid"]]
        with pytest.raises(ValueError, match="must be an integer"):
            mock_parser._convert_grid_to_jax(grid_data)

    def test_convert_grid_to_jax_invalid_cell_value(self, mock_parser: MockParser):
        """Test _convert_grid_to_jax with invalid cell values."""
        grid_data = [[0, 1], [2, 10]]  # 10 is outside valid range 0-9
        with pytest.raises(ValueError, match="must be 0-9"):
            mock_parser._convert_grid_to_jax(grid_data)

    def test_validate_grid_colors_valid(self, mock_parser: MockParser):
        """Test _validate_grid_colors with valid colors."""
        grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        # Should not raise any exception
        mock_parser._validate_grid_colors(grid)

    def test_validate_grid_colors_invalid(self, mock_parser: MockParser):
        """Test _validate_grid_colors with invalid colors."""
        grid = jnp.array([[0, 1, 2], [3, 4, 10]], dtype=jnp.int32)  # 10 is invalid
        with pytest.raises(ValueError, match="Invalid color in grid"):
            mock_parser._validate_grid_colors(grid)

    def test_from_hydra_class_method(self, valid_config: DatasetConfig):
        """Test from_hydra class method creates parser correctly."""
        mock_hydra_config = Mock()

        with patch.object(DatasetConfig, "from_hydra", return_value=valid_config):
            parser = MockParser.from_hydra(mock_hydra_config)

            assert isinstance(parser, MockParser)
            assert parser.config == valid_config

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented by subclasses."""

        class IncompleteParser(ArcDataParserBase):
            pass

        config = DatasetConfig(
            dataset_path="test/data",
            max_grid_height=10,
            max_grid_width=10,
            min_grid_height=1,
            min_grid_width=1,
            max_colors=10,
            background_color=0,
            max_train_pairs=5,
            max_test_pairs=3,
            task_split="train",
        )

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteParser(config)

    def test_task_index_mapping_methods(self, mock_parser: MockParser):
        """Test task index mapping and validation methods."""
        # Test get_available_task_ids
        task_ids = mock_parser.get_available_task_ids()
        assert task_ids == ["task_001", "task_002", "task_003"]

        # Test get_task_by_id with valid ID
        task = mock_parser.get_task_by_id("task_001")
        assert isinstance(task, JaxArcTask)

        # Test get_task_by_id with invalid ID
        with pytest.raises(ValueError, match="not found"):
            mock_parser.get_task_by_id("invalid_task")

    @patch("jaxarc.utils.task_manager.get_task_id_globally")
    def test_validate_task_index_mapping(
        self, mock_get_task_id, mock_parser: MockParser
    ):
        """Test validate_task_index_mapping method."""
        # Test valid task index
        mock_get_task_id.return_value = "task_001"
        assert mock_parser.validate_task_index_mapping(0) is True

        # Test invalid task index (not in global manager)
        mock_get_task_id.return_value = None
        assert mock_parser.validate_task_index_mapping(999) is False

        # Test invalid task index (not in this parser)
        mock_get_task_id.return_value = "unknown_task"
        assert mock_parser.validate_task_index_mapping(0) is False

    @patch("jaxarc.utils.task_manager.get_task_id_globally")
    def test_reconstruct_task_from_index(
        self, mock_get_task_id, mock_parser: MockParser
    ):
        """Test reconstruct_task_from_index method."""
        # Test successful reconstruction
        mock_get_task_id.return_value = "task_001"
        task = mock_parser.reconstruct_task_from_index(0)
        assert isinstance(task, JaxArcTask)

        # Test reconstruction with invalid index
        mock_get_task_id.return_value = None
        with pytest.raises(ValueError, match="not found in global task manager"):
            mock_parser.reconstruct_task_from_index(999)

        # Test reconstruction with unknown task ID
        mock_get_task_id.return_value = "unknown_task"
        with pytest.raises(ValueError, match="Cannot reconstruct task"):
            mock_parser.reconstruct_task_from_index(0)

    @patch("jaxarc.utils.task_manager.register_task_globally")
    def test_get_task_index_for_id(self, mock_register_task, mock_parser: MockParser):
        """Test get_task_index_for_id method."""
        # Test successful index retrieval
        mock_register_task.return_value = 42
        index = mock_parser.get_task_index_for_id("task_001")
        assert index == 42
        mock_register_task.assert_called_once_with("task_001")

        # Test with unavailable task ID
        with pytest.raises(ValueError, match="not available in this parser"):
            mock_parser.get_task_index_for_id("unknown_task")
