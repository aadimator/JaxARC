"""Comprehensive tests for ArcDataParserBase functionality.

This test suite provides comprehensive coverage of the base parser class,
including configuration validation, abstract method enforcement, utility methods,
and error handling scenarios.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.parsers.base_parser import ArcDataParserBase
from jaxarc.types import JaxArcTask


class MockConcreteParser(ArcDataParserBase):
    """Mock concrete implementation for testing base parser functionality."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.mock_tasks = {}
        self.mock_task_ids = []

    def load_task_file(self, task_file_path: str) -> Any:
        """Mock implementation of load_task_file."""
        if task_file_path == "/nonexistent/file.json":
            raise FileNotFoundError(f"Task file not found: {task_file_path}")
        if task_file_path == "/invalid/file.json":
            raise ValueError(f"Invalid JSON in file {task_file_path}")

        return {
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[3, 4]]}],
        }

    def preprocess_task_data(self, raw_task_data: Any, key: chex.PRNGKey) -> JaxArcTask:
        """Mock implementation of preprocess_task_data."""
        # Create a minimal valid JaxArcTask
        return JaxArcTask(
            input_grids_examples=jnp.zeros(
                (self.max_train_pairs, self.max_grid_height, self.max_grid_width),
                dtype=jnp.int32,
            ),
            input_masks_examples=jnp.ones(
                (self.max_train_pairs, self.max_grid_height, self.max_grid_width),
                dtype=jnp.bool_,
            ),
            output_grids_examples=jnp.zeros(
                (self.max_train_pairs, self.max_grid_height, self.max_grid_width),
                dtype=jnp.int32,
            ),
            output_masks_examples=jnp.ones(
                (self.max_train_pairs, self.max_grid_height, self.max_grid_width),
                dtype=jnp.bool_,
            ),
            num_train_pairs=1,
            test_input_grids=jnp.zeros(
                (self.max_test_pairs, self.max_grid_height, self.max_grid_width),
                dtype=jnp.int32,
            ),
            test_input_masks=jnp.ones(
                (self.max_test_pairs, self.max_grid_height, self.max_grid_width),
                dtype=jnp.bool_,
            ),
            true_test_output_grids=jnp.zeros(
                (self.max_test_pairs, self.max_grid_height, self.max_grid_width),
                dtype=jnp.int32,
            ),
            true_test_output_masks=jnp.ones(
                (self.max_test_pairs, self.max_grid_height, self.max_grid_width),
                dtype=jnp.bool_,
            ),
            num_test_pairs=1,
            task_index=jnp.array(42, dtype=jnp.int32),
        )

    def get_random_task(self, key: chex.PRNGKey) -> JaxArcTask:
        """Mock implementation of get_random_task."""
        if not self.mock_task_ids:
            raise RuntimeError("No tasks available in dataset")
        return self.preprocess_task_data({}, key)


class TestArcDataParserBaseComprehensive:
    """Comprehensive test suite for ArcDataParserBase."""

    @pytest.fixture
    def valid_config(self):
        """Valid configuration for testing."""
        return DictConfig(
            {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
                "max_train_pairs": 5,
                "max_test_pairs": 3,
            }
        )

    @pytest.fixture
    def parser(self, valid_config):
        """Create a mock parser instance."""
        return MockConcreteParser(valid_config)

    def test_initialization_with_valid_config(self, valid_config):
        """Test successful initialization with valid configuration."""
        parser = MockConcreteParser(valid_config)

        assert parser.max_grid_height == 30
        assert parser.max_grid_width == 30
        assert parser.min_grid_height == 1
        assert parser.min_grid_width == 1
        assert parser.max_colors == 10
        assert parser.background_color == 0
        assert parser.max_train_pairs == 5
        assert parser.max_test_pairs == 3

    def test_initialization_missing_required_fields(self):
        """Test initialization fails with missing required configuration fields."""
        required_fields = [
            "max_grid_height",
            "max_grid_width",
            "min_grid_height",
            "min_grid_width",
            "max_colors",
            "background_color",
            "max_train_pairs",
            "max_test_pairs",
        ]

        for missing_field in required_fields:
            config = DictConfig(
                {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                    "max_train_pairs": 5,
                    "max_test_pairs": 3,
                }
            )
            del config[missing_field]

            with pytest.raises(KeyError, match="Missing required configuration field"):
                MockConcreteParser(config)

    def test_initialization_invalid_grid_dimensions(self):
        """Test initialization fails with invalid grid dimensions."""
        base_config = {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
            "max_train_pairs": 5,
            "max_test_pairs": 3,
        }

        # Test negative max dimensions
        invalid_configs = [
            {**base_config, "max_grid_height": -1},
            {**base_config, "max_grid_width": -1},
            {**base_config, "max_grid_height": 0},
            {**base_config, "max_grid_width": 0},
        ]

        for config in invalid_configs:
            with pytest.raises(ValueError, match="Grid dimensions must be positive"):
                MockConcreteParser(DictConfig(config))

    def test_initialization_invalid_min_dimensions(self):
        """Test initialization fails with invalid minimum dimensions."""
        base_config = {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
            "max_train_pairs": 5,
            "max_test_pairs": 3,
        }

        # Test negative min dimensions
        invalid_configs = [
            {**base_config, "min_grid_height": -1},
            {**base_config, "min_grid_width": -1},
            {**base_config, "min_grid_height": 0},
            {**base_config, "min_grid_width": 0},
        ]

        for config in invalid_configs:
            with pytest.raises(
                ValueError, match="Minimum grid dimensions must be positive"
            ):
                MockConcreteParser(DictConfig(config))

    def test_initialization_min_exceeds_max_dimensions(self):
        """Test initialization fails when min dimensions exceed max dimensions."""
        base_config = {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
            "max_train_pairs": 5,
            "max_test_pairs": 3,
        }

        invalid_configs = [
            {**base_config, "min_grid_height": 31},  # min_height > max_height
            {**base_config, "min_grid_width": 31},  # min_width > max_width
        ]

        for config in invalid_configs:
            with pytest.raises(
                ValueError, match="Minimum dimensions.*cannot exceed maximum"
            ):
                MockConcreteParser(DictConfig(config))

    def test_initialization_invalid_pair_counts(self):
        """Test initialization fails with invalid pair counts."""
        base_config = {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
            "max_train_pairs": 5,
            "max_test_pairs": 3,
        }

        invalid_configs = [
            {**base_config, "max_train_pairs": -1},
            {**base_config, "max_train_pairs": 0},
            {**base_config, "max_test_pairs": -1},
            {**base_config, "max_test_pairs": 0},
        ]

        for config in invalid_configs:
            with pytest.raises(ValueError, match="Max pairs must be positive"):
                MockConcreteParser(DictConfig(config))

    def test_initialization_invalid_color_config(self):
        """Test initialization fails with invalid color configuration."""
        base_config = {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
            "max_train_pairs": 5,
            "max_test_pairs": 3,
        }

        # Test invalid max_colors
        with pytest.raises(ValueError, match="Max colors must be positive"):
            MockConcreteParser(DictConfig({**base_config, "max_colors": -1}))

        with pytest.raises(ValueError, match="Max colors must be positive"):
            MockConcreteParser(DictConfig({**base_config, "max_colors": 0}))

        # Test invalid background_color
        with pytest.raises(ValueError, match="Background color.*must be in range"):
            MockConcreteParser(DictConfig({**base_config, "background_color": -1}))

        with pytest.raises(ValueError, match="Background color.*must be in range"):
            MockConcreteParser(DictConfig({**base_config, "background_color": 10}))

    def test_get_max_dimensions(self, parser):
        """Test get_max_dimensions method."""
        dims = parser.get_max_dimensions()
        assert dims == (30, 30, 5, 3)  # height, width, train_pairs, test_pairs

    def test_get_grid_config(self, parser):
        """Test get_grid_config method."""
        config = parser.get_grid_config()
        expected = {
            "max_grid_height": 30,
            "max_grid_width": 30,
            "min_grid_height": 1,
            "min_grid_width": 1,
            "max_colors": 10,
            "background_color": 0,
        }
        assert config == expected

    def test_validate_grid_dimensions_valid(self, parser):
        """Test validate_grid_dimensions with valid dimensions."""
        # Should not raise for valid dimensions
        parser.validate_grid_dimensions(1, 1)  # minimum
        parser.validate_grid_dimensions(30, 30)  # maximum
        parser.validate_grid_dimensions(15, 20)  # within range

    def test_validate_grid_dimensions_below_minimum(self, parser):
        """Test validate_grid_dimensions with dimensions below minimum."""
        with pytest.raises(ValueError, match="below minimum"):
            parser.validate_grid_dimensions(0, 1)

        with pytest.raises(ValueError, match="below minimum"):
            parser.validate_grid_dimensions(1, 0)

    def test_validate_grid_dimensions_above_maximum(self, parser):
        """Test validate_grid_dimensions with dimensions above maximum."""
        with pytest.raises(ValueError, match="exceed maximum"):
            parser.validate_grid_dimensions(31, 30)

        with pytest.raises(ValueError, match="exceed maximum"):
            parser.validate_grid_dimensions(30, 31)

    def test_validate_color_value_valid(self, parser):
        """Test validate_color_value with valid colors."""
        # Should not raise for valid colors
        for color in range(10):  # 0-9 are valid
            parser.validate_color_value(color)

    def test_validate_color_value_invalid(self, parser):
        """Test validate_color_value with invalid colors."""
        with pytest.raises(ValueError, match="must be in range"):
            parser.validate_color_value(-1)

        with pytest.raises(ValueError, match="must be in range"):
            parser.validate_color_value(10)

    def test_process_training_pairs_valid(self, parser):
        """Test _process_training_pairs with valid data."""
        task_content = {
            "train": [
                {"input": [[1, 2]], "output": [[2, 1]]},
                {"input": [[3, 4]], "output": [[4, 3]]},
            ]
        }

        with patch.object(parser, "_validate_grid_colors"):
            train_inputs, train_outputs = parser._process_training_pairs(task_content)

        assert len(train_inputs) == 2
        assert len(train_outputs) == 2
        assert isinstance(train_inputs[0], jnp.ndarray)
        assert isinstance(train_outputs[0], jnp.ndarray)

    def test_process_training_pairs_empty(self, parser):
        """Test _process_training_pairs with empty training data."""
        task_content = {"train": []}

        with pytest.raises(ValueError, match="at least one training pair"):
            parser._process_training_pairs(task_content)

    def test_process_training_pairs_missing_data(self, parser):
        """Test _process_training_pairs with missing input/output."""
        # Missing input
        task_content = {"train": [{"output": [[1, 2]]}]}
        with pytest.raises(ValueError, match="missing input or output"):
            parser._process_training_pairs(task_content)

        # Missing output
        task_content = {"train": [{"input": [[1, 2]]}]}
        with pytest.raises(ValueError, match="missing input or output"):
            parser._process_training_pairs(task_content)

    def test_process_test_pairs_valid(self, parser):
        """Test _process_test_pairs with valid data."""
        task_content = {
            "test": [
                {"input": [[1, 2]], "output": [[2, 1]]},
                {"input": [[3, 4]]},  # No output (typical for test)
            ]
        }

        with patch.object(parser, "_validate_grid_colors"):
            test_inputs, test_outputs = parser._process_test_pairs(task_content)

        assert len(test_inputs) == 2
        assert len(test_outputs) == 2
        assert isinstance(test_inputs[0], jnp.ndarray)
        assert isinstance(test_outputs[0], jnp.ndarray)

    def test_process_test_pairs_empty(self, parser):
        """Test _process_test_pairs with empty test data."""
        task_content = {"test": []}

        with pytest.raises(ValueError, match="at least one test pair"):
            parser._process_test_pairs(task_content)

    def test_process_test_pairs_missing_input(self, parser):
        """Test _process_test_pairs with missing input."""
        task_content = {"test": [{"output": [[1, 2]]}]}  # Missing input

        with pytest.raises(ValueError, match="missing input"):
            parser._process_test_pairs(task_content)

    def test_pad_and_create_masks(self, parser):
        """Test _pad_and_create_masks functionality."""
        train_inputs = [jnp.array([[1, 2]], dtype=jnp.int32)]
        train_outputs = [jnp.array([[2, 1]], dtype=jnp.int32)]
        test_inputs = [jnp.array([[3, 4]], dtype=jnp.int32)]
        test_outputs = [jnp.array([[4, 3]], dtype=jnp.int32)]

        with patch("jaxarc.parsers.utils.pad_array_sequence") as mock_pad:
            mock_pad.return_value = (
                jnp.zeros((5, 30, 30), dtype=jnp.int32),
                jnp.ones((5, 30, 30), dtype=jnp.bool_),
            )

            result = parser._pad_and_create_masks(
                train_inputs, train_outputs, test_inputs, test_outputs
            )

        # Check that pad_array_sequence was called correctly
        assert mock_pad.call_count == 4  # Called for each array type

        # Check result structure
        expected_keys = [
            "train_inputs",
            "train_input_masks",
            "train_outputs",
            "train_output_masks",
            "test_inputs",
            "test_input_masks",
            "test_outputs",
            "test_output_masks",
        ]
        for key in expected_keys:
            assert key in result

    def test_validate_grid_colors_valid(self, parser):
        """Test _validate_grid_colors with valid colors."""
        valid_grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        # Should not raise
        parser._validate_grid_colors(valid_grid)

    def test_validate_grid_colors_invalid(self, parser):
        """Test _validate_grid_colors with invalid colors."""
        invalid_grid = jnp.array([[0, 1, 15]], dtype=jnp.int32)  # 15 > max_colors

        with pytest.raises(ValueError, match="Invalid color in grid"):
            parser._validate_grid_colors(invalid_grid)

    def test_log_parsing_stats(self, parser):
        """Test _log_parsing_stats functionality."""
        train_inputs = [jnp.array([[1, 2]], dtype=jnp.int32)]
        train_outputs = [jnp.array([[2, 1]], dtype=jnp.int32)]
        test_inputs = [jnp.array([[3, 4]], dtype=jnp.int32)]
        test_outputs = [jnp.array([[4, 3]], dtype=jnp.int32)]

        with patch("jaxarc.parsers.utils.log_parsing_stats") as mock_log:
            parser._log_parsing_stats(
                train_inputs, train_outputs, test_inputs, test_outputs, "test_task"
            )

        # Should have called the logging function
        mock_log.assert_called_once()

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated directly."""
        config = DictConfig(
            {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
                "max_train_pairs": 5,
                "max_test_pairs": 3,
            }
        )

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ArcDataParserBase(config)  # type: ignore[abstract]

    def test_abstract_methods_enforcement(self):
        """Test that concrete implementations must implement all abstract methods."""

        class IncompleteParser(ArcDataParserBase):
            """Incomplete parser missing abstract method implementations."""

        config = DictConfig(
            {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
                "max_train_pairs": 5,
                "max_test_pairs": 3,
            }
        )

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteParser(config)  # type: ignore[abstract]

    def test_load_task_file_success(self, parser):
        """Test successful load_task_file call."""
        result = parser.load_task_file("/valid/file.json")
        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result

    def test_load_task_file_not_found(self, parser):
        """Test load_task_file with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Task file not found"):
            parser.load_task_file("/nonexistent/file.json")

    def test_load_task_file_invalid_json(self, parser):
        """Test load_task_file with invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parser.load_task_file("/invalid/file.json")

    def test_preprocess_task_data_success(self, parser):
        """Test successful preprocess_task_data call."""
        raw_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        }
        key = jax.random.PRNGKey(42)

        result = parser.preprocess_task_data(raw_data, key)
        assert isinstance(result, JaxArcTask)

    def test_get_random_task_success(self, parser):
        """Test successful get_random_task call."""
        parser.mock_task_ids = ["task1", "task2"]  # Add some mock tasks
        key = jax.random.PRNGKey(42)

        result = parser.get_random_task(key)
        assert isinstance(result, JaxArcTask)

    def test_get_random_task_no_tasks(self, parser):
        """Test get_random_task with no available tasks."""
        key = jax.random.PRNGKey(42)

        with pytest.raises(RuntimeError, match="No tasks available"):
            parser.get_random_task(key)

    def test_configuration_edge_cases(self):
        """Test edge cases in configuration validation."""
        # Test with minimum valid values
        min_config = DictConfig(
            {
                "max_grid_height": 1,
                "max_grid_width": 1,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 1,
                "background_color": 0,
                "max_train_pairs": 1,
                "max_test_pairs": 1,
            }
        )

        parser = MockConcreteParser(min_config)
        assert parser.max_grid_height == 1
        assert parser.max_colors == 1

    def test_jax_compatibility(self, parser):
        """Test that parser methods work with JAX transformations."""
        key = jax.random.PRNGKey(42)
        parser.mock_task_ids = ["task1"]

        # Test JIT compilation of get_random_task
        @jax.jit
        def get_task_jitted(key):
            return parser.get_random_task(key)

        # This should work without errors
        task = get_task_jitted(key)
        assert isinstance(task, JaxArcTask)

    def test_error_message_quality(self):
        """Test that error messages are informative and helpful."""
        # Test missing field error
        config = DictConfig({"max_grid_height": 30})  # Missing other required fields

        try:
            MockConcreteParser(config)
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert "Missing required configuration field" in str(e)

        # Test dimension validation error
        config = DictConfig(
            {
                "max_grid_height": -1,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
                "max_train_pairs": 5,
                "max_test_pairs": 3,
            }
        )

        try:
            MockConcreteParser(config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Grid dimensions must be positive" in str(e)
            assert "-1" in str(e)  # Should include the invalid value
