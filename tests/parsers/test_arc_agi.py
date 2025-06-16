"""Tests for the updated ArcAgiParser class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.types import ParsedTaskData


@pytest.fixture
def sample_task_data():
    """Sample ARC task data in JSON format."""
    return {
        "test_task_001": {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                {"input": [[5, 6]], "output": [[6, 5]]},
            ],
            "test": [{"input": [[7, 8]], "output": [[8, 7]]}],
        }
    }


@pytest.fixture
def sample_challenge_data():
    """Sample challenge data (test inputs without outputs)."""
    return {
        "test_task_001": {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                {"input": [[5, 6]], "output": [[6, 5]]},
            ],
            "test": [
                {"input": [[7, 8]]}  # No output for challenge format
            ],
        }
    }


@pytest.fixture
def temp_challenges_file(sample_challenge_data):
    """Create a temporary JSON file with challenge data."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(sample_challenge_data, f)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_solutions_file():
    """Create a temporary JSON file with solutions data."""
    solutions_data = {
        "test_task_001": [[[8, 7]]]  # Solutions for test inputs
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(solutions_data, f)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def parser_config(temp_challenges_file, temp_solutions_file):
    """Create parser configuration."""
    return DictConfig(
        {
            "default_split": "training",
            "training": {
                "challenges": str(temp_challenges_file),
                "solutions": str(temp_solutions_file),
            },
        }
    )


@pytest.fixture
def parser(parser_config):
    """Create ArcAgiParser instance."""
    # Add max dimensions to the config
    config_with_dims = parser_config.copy()
    config_with_dims["max_grid_height"] = 30
    config_with_dims["max_grid_width"] = 30
    config_with_dims["max_train_pairs"] = 5
    config_with_dims["max_test_pairs"] = 5

    return ArcAgiParser(cfg=config_with_dims)


class TestArcAgiParser:
    """Test suite for ArcAgiParser."""

    def test_parser_initialization(self, parser_config):
        """Test parser initializes correctly."""
        # Add max dimensions to the config
        config_with_dims = parser_config.copy()
        config_with_dims["max_grid_height"] = 30
        config_with_dims["max_grid_width"] = 30
        config_with_dims["max_train_pairs"] = 5
        config_with_dims["max_test_pairs"] = 5

        parser = ArcAgiParser(cfg=config_with_dims)

        assert parser.max_grid_height == 30
        assert parser.max_grid_width == 30
        assert parser.max_train_pairs == 5
        assert parser.max_test_pairs == 5
        assert len(parser.get_available_task_ids()) > 0

    def test_load_task_file(self, parser, temp_challenges_file):
        """Test loading task file."""
        data = parser.load_task_file(str(temp_challenges_file))

        assert isinstance(data, dict)
        assert "test_task_001" in data
        assert "train" in data["test_task_001"]
        assert "test" in data["test_task_001"]

    def test_load_nonexistent_file(self, parser):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.load_task_file("/nonexistent/file.json")

    def test_load_invalid_json(self, parser):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                parser.load_task_file(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

    def test_preprocess_task_data(self, parser, sample_task_data):
        """Test preprocessing task data."""
        key = jax.random.PRNGKey(42)
        parsed_data = parser.preprocess_task_data(sample_task_data, key)

        assert isinstance(parsed_data, ParsedTaskData)
        assert parsed_data.task_id == "test_task_001"
        assert parsed_data.num_train_pairs == 2
        assert parsed_data.num_test_pairs == 1

        # Check shapes
        assert parsed_data.input_grids_examples.shape == (5, 30, 30)  # padded
        assert parsed_data.output_grids_examples.shape == (5, 30, 30)
        assert parsed_data.test_input_grids.shape == (5, 30, 30)
        assert parsed_data.true_test_output_grids.shape == (5, 30, 30)

        # Check masks
        assert parsed_data.input_masks_examples.dtype == jnp.bool_
        assert parsed_data.output_masks_examples.dtype == jnp.bool_
        assert parsed_data.test_input_masks.dtype == jnp.bool_
        assert parsed_data.true_test_output_masks.dtype == jnp.bool_

    def test_preprocess_invalid_task_data(self, parser):
        """Test preprocessing invalid task data."""
        key = jax.random.PRNGKey(42)

        # Test non-dict input
        with pytest.raises(ValueError, match="Expected dict"):
            parser.preprocess_task_data("invalid", key)

        # Test multiple tasks
        with pytest.raises(ValueError, match="Expected single task"):
            parser.preprocess_task_data({"task1": {}, "task2": {}}, key)

        # Test task without training pairs
        with pytest.raises(ValueError, match="at least one training pair"):
            parser.preprocess_task_data(
                {"task": {"train": [], "test": [{"input": [[1]]}]}}, key
            )

    def test_preprocess_validates_grid_dimensions(self, parser_config):
        """Test that preprocessing validates grid dimensions."""
        # Create parser with small max dimensions
        config_with_dims = parser_config.copy()
        config_with_dims["max_grid_height"] = 2
        config_with_dims["max_grid_width"] = 2
        config_with_dims["max_train_pairs"] = 5
        config_with_dims["max_test_pairs"] = 5

        parser = ArcAgiParser(cfg=config_with_dims)

        # Create task with grid exceeding max dimensions
        large_grid_task = {
            "task": {
                "train": [
                    {
                        "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        "output": [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                    }
                ],
                "test": [{"input": [[1, 2, 3]]}],
            }
        }

        key = jax.random.PRNGKey(42)
        with pytest.raises(ValueError, match="exceed maximum"):
            parser.preprocess_task_data(large_grid_task, key)

    def test_get_random_task(self, parser):
        """Test getting random task."""
        key = jax.random.PRNGKey(42)
        parsed_data = parser.get_random_task(key)

        assert isinstance(parsed_data, ParsedTaskData)
        assert parsed_data.task_id is not None
        assert parsed_data.num_train_pairs > 0
        assert parsed_data.num_test_pairs > 0

    def test_get_task_by_id(self, parser):
        """Test getting specific task by ID."""
        task_ids = parser.get_available_task_ids()
        assert len(task_ids) > 0

        task_id = task_ids[0]
        parsed_data = parser.get_task_by_id(task_id)

        assert isinstance(parsed_data, ParsedTaskData)
        assert parsed_data.task_id == task_id

    def test_get_task_by_invalid_id(self, parser):
        """Test getting task with invalid ID."""
        with pytest.raises(ValueError, match="not found in dataset"):
            parser.get_task_by_id("nonexistent_task")

    def test_get_available_task_ids(self, parser):
        """Test getting available task IDs."""
        task_ids = parser.get_available_task_ids()

        assert isinstance(task_ids, list)
        assert len(task_ids) > 0
        assert "test_task_001" in task_ids

    def test_parser_with_empty_dataset(self, temp_challenges_file):
        """Test parser behavior with empty dataset."""
        # Create empty dataset
        with temp_challenges_file.open("w") as f:
            json.dump({}, f)
        config = DictConfig(
            {
                "default_split": "training",
                "training": {
                    "challenges": str(temp_challenges_file),
                },
                "max_grid_height": 30,
                "max_grid_width": 30,
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )

        parser = ArcAgiParser(cfg=config)

        # Should have no tasks
        assert len(parser.get_available_task_ids()) == 0

        # Should raise error when trying to get random task
        key = jax.random.PRNGKey(42)
        with pytest.raises(RuntimeError, match="No tasks available"):
            parser.get_random_task(key)

    def test_jax_compatibility(self, parser):
        """Test that parsed data is JAX-compatible."""
        key = jax.random.PRNGKey(42)
        parsed_data = parser.get_random_task(key)

        # Test that we can use JAX transformations on the JAX array fields
        def sum_grids(input_grids: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(input_grids)

        jitted_sum = jax.jit(sum_grids)
        result = jitted_sum(parsed_data.input_grids_examples)

        assert isinstance(result, jnp.ndarray)
        assert result.dtype in [jnp.int32, jnp.int64]

        # Test that we can also jit over the whole structure by extracting JAX-compatible parts
        def process_arrays(
            input_grids: jnp.ndarray,
            input_masks: jnp.ndarray,
            output_grids: jnp.ndarray,
            output_masks: jnp.ndarray,
        ) -> jnp.ndarray:
            # Sum of valid input pixels
            valid_inputs = jnp.sum(input_grids * input_masks)
            valid_outputs = jnp.sum(output_grids * output_masks)
            return valid_inputs + valid_outputs

        jitted_process = jax.jit(process_arrays)
        result2 = jitted_process(
            parsed_data.input_grids_examples,
            parsed_data.input_masks_examples,
            parsed_data.output_grids_examples,
            parsed_data.output_masks_examples,
        )

        assert isinstance(result2, jnp.ndarray)

    def test_padding_and_masking(self, parser, sample_task_data):
        """Test that padding and masking work correctly."""
        key = jax.random.PRNGKey(42)
        parsed_data = parser.preprocess_task_data(sample_task_data, key)

        # Check that masks correctly identify valid data
        # First training pair: input is 2x2, should have True mask in top-left 2x2 region
        first_input_mask = parsed_data.input_masks_examples[0]
        assert jnp.all(first_input_mask[:2, :2])  # Valid region
        assert not jnp.any(first_input_mask[2:, 2:])  # Padded region should be False

        # Check that actual data is preserved in valid regions
        first_input_grid = parsed_data.input_grids_examples[0]
        expected_values = jnp.array([[1, 2], [3, 4]])
        assert jnp.array_equal(first_input_grid[:2, :2], expected_values)

        # Check that padded regions use -1 as fill value
        assert jnp.all(first_input_grid[2:, :] == -1)  # Bottom rows
        assert jnp.all(first_input_grid[:, 2:] == -1)  # Right columns

    def test_different_grid_sizes(self, parser_config):
        """Test handling of different grid sizes within max limits."""
        config_with_dims = parser_config.copy()
        config_with_dims["max_grid_height"] = 10
        config_with_dims["max_grid_width"] = 10
        config_with_dims["max_train_pairs"] = 3
        config_with_dims["max_test_pairs"] = 2

        parser = ArcAgiParser(cfg=config_with_dims)

        mixed_size_task = {
            "mixed_task": {
                "train": [
                    {"input": [[1]], "output": [[2]]},  # 1x1
                    {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},  # 2x2
                ],
                "test": [
                    {"input": [[1, 2, 3]], "output": [[3, 2, 1]]}  # 1x3
                ],
            }
        }

        key = jax.random.PRNGKey(42)
        parsed_data = parser.preprocess_task_data(mixed_size_task, key)

        # All grids should be padded to 10x10
        assert parsed_data.input_grids_examples.shape == (3, 10, 10)
        assert parsed_data.output_grids_examples.shape == (3, 10, 10)
        assert parsed_data.test_input_grids.shape == (2, 10, 10)

        # Check that different sized grids are correctly masked
        # First training pair is 1x1
        assert jnp.sum(parsed_data.input_masks_examples[0]) == 1
        # Second training pair is 2x2
        assert jnp.sum(parsed_data.input_masks_examples[1]) == 4
        # Test pair is 1x3
        assert jnp.sum(parsed_data.test_input_masks[0]) == 3

    def test_solutions_merging(self):
        """Test that solutions are properly merged with challenges."""
        # Create sample challenge data
        challenge_data = {
            "test_task": {
                "train": [
                    {"input": [[1, 2]], "output": [[2, 1]]},
                ],
                "test": [
                    {"input": [[3, 4]]},  # No output initially
                    {"input": [[5, 6]]},  # No output initially
                ],
            }
        }

        # Create sample solutions data
        solutions_data = {
            "test_task": [
                [[4, 3]],  # Solution for first test pair
                [[6, 5]],  # Solution for second test pair
            ]
        }

        # Create temporary files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as challenges_file:
            json.dump(challenge_data, challenges_file)
            challenges_path = challenges_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as solutions_file:
            json.dump(solutions_data, solutions_file)
            solutions_path = solutions_file.name

        try:
            # Create parser configuration
            cfg = DictConfig(
                {
                    "default_split": "training",
                    "training": {
                        "challenges": challenges_path,
                        "solutions": solutions_path,
                    },
                    "max_grid_height": 10,
                    "max_grid_width": 10,
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )

            # Create parser
            parser = ArcAgiParser(cfg=cfg)

            # Get the task
            parsed_task = parser.get_task_by_id("test_task")

            # Verify the solutions were merged correctly
            assert parsed_task.task_id == "test_task"
            assert parsed_task.num_test_pairs == 2

            # Check that test outputs are now present and correct
            # Note: The outputs should be padded to max dimensions but contain the solution data
            test_output_1 = parsed_task.true_test_output_grids[0]
            test_output_2 = parsed_task.true_test_output_grids[1]

            # Check the actual values in the non-padded region
            assert test_output_1[0, 0] == 4  # First solution: [[4, 3]]
            assert test_output_1[0, 1] == 3
            assert test_output_2[0, 0] == 6  # Second solution: [[6, 5]]
            assert test_output_2[0, 1] == 5

            # Also verify that the masks are correct
            test_output_mask_1 = parsed_task.true_test_output_masks[0]
            test_output_mask_2 = parsed_task.true_test_output_masks[1]

            # First test output is 1x2, so 2 valid cells
            assert jnp.sum(test_output_mask_1) == 2
            # Second test output is 1x2, so 2 valid cells
            assert jnp.sum(test_output_mask_2) == 2

        finally:
            # Clean up temporary files
            Path(challenges_path).unlink()
            Path(solutions_path).unlink()
