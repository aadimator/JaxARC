"""Tests for ArcAgiParser GitHub format parsing functionality."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.types import JaxArcTask


class TestArcAgiParserGitHubFormat:
    """Test suite for ArcAgiParser GitHub format parsing."""

    @pytest.fixture
    def sample_github_task_data(self):
        """Sample task data in GitHub format (direct task content)."""
        return {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                {"input": [[5, 6]], "output": [[6, 5]]},
            ],
            "test": [{"input": [[7, 8]], "output": [[8, 7]]}],
        }

    @pytest.fixture
    def sample_github_task_no_test_output(self):
        """Sample task data in GitHub format without test outputs."""
        return {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                {"input": [[5, 6]], "output": [[6, 5]]},
            ],
            "test": [{"input": [[7, 8]]}],  # No output for test
        }

    @pytest.fixture
    def temp_github_directory(self, sample_github_task_data):
        """Create a temporary directory with GitHub format JSON files."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create individual JSON files
        task_files = {
            "007bbfb7.json": sample_github_task_data,
            "00d62c1b.json": {
                "train": [
                    {"input": [[0, 1]], "output": [[1, 0]]},
                ],
                "test": [{"input": [[2, 3]], "output": [[3, 2]]}],
            },
            "025d127b.json": {
                "train": [
                    {"input": [[9, 8, 7]], "output": [[7, 8, 9]]},
                    {"input": [[6, 5]], "output": [[5, 6]]},
                ],
                "test": [{"input": [[4, 3, 2, 1]]}],  # No test output
            },
        }

        for filename, task_data in task_files.items():
            task_file = temp_dir / filename
            with task_file.open("w", encoding="utf-8") as f:
                json.dump(task_data, f)

        yield temp_dir

        # Cleanup
        for file in temp_dir.glob("*.json"):
            file.unlink()
        temp_dir.rmdir()

    @pytest.fixture
    def github_parser_config(self, temp_github_directory):
        """Create parser configuration for GitHub format."""
        return DictConfig(
            {
                "default_split": "training",
                "training": {
                    "path": str(temp_github_directory),
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )

    @pytest.fixture
    def github_parser(self, github_parser_config):
        """Create ArcAgiParser instance for GitHub format."""
        return ArcAgiParser(cfg=github_parser_config)

    def test_individual_json_file_loading(self, github_parser):
        """Test that individual JSON files are loaded correctly."""
        task_ids = github_parser.get_available_task_ids()

        # Should have loaded all 3 task files
        assert len(task_ids) == 3
        assert "007bbfb7" in task_ids
        assert "00d62c1b" in task_ids
        assert "025d127b" in task_ids

    def test_task_id_extraction_from_filenames(self, github_parser):
        """Test that task IDs are correctly extracted from filenames."""
        task_ids = github_parser.get_available_task_ids()

        # Task IDs should be the filename stems (without .json extension)
        expected_ids = {"007bbfb7", "00d62c1b", "025d127b"}
        assert set(task_ids) == expected_ids

    def test_github_format_task_parsing(self, github_parser, sample_github_task_data):
        """Test parsing of GitHub format task data."""
        key = jax.random.PRNGKey(42)
        parsed_task = github_parser.preprocess_task_data(
            sample_github_task_data, key, task_id="test_task"
        )

        assert isinstance(parsed_task, JaxArcTask)
        assert parsed_task.num_train_pairs == 2
        assert parsed_task.num_test_pairs == 1

        # Check that data is correctly parsed
        # First training input should be [[1, 2], [3, 4]]
        first_input = parsed_task.input_grids_examples[0]
        assert first_input[0, 0] == 1
        assert first_input[0, 1] == 2
        assert first_input[1, 0] == 3
        assert first_input[1, 1] == 4

        # First training output should be [[2, 1], [4, 3]]
        first_output = parsed_task.output_grids_examples[0]
        assert first_output[0, 0] == 2
        assert first_output[0, 1] == 1
        assert first_output[1, 0] == 4
        assert first_output[1, 1] == 3

    def test_github_format_without_test_outputs(
        self, github_parser, sample_github_task_no_test_output
    ):
        """Test parsing GitHub format task without test outputs."""
        key = jax.random.PRNGKey(42)
        parsed_task = github_parser.preprocess_task_data(
            sample_github_task_no_test_output, key, task_id="test_task"
        )

        assert isinstance(parsed_task, JaxArcTask)
        assert parsed_task.num_train_pairs == 2
        assert parsed_task.num_test_pairs == 1

        # Test input should be parsed correctly
        test_input = parsed_task.test_input_grids[0]
        assert test_input[0, 0] == 7
        assert test_input[0, 1] == 8

        # Test output should be dummy (zeros) since no output provided
        test_output = parsed_task.true_test_output_grids[0]
        # The dummy output should be zeros with same shape as input
        assert test_output[0, 0] == 0
        assert test_output[0, 1] == 0

    def test_missing_json_files_error_handling(self):
        """Test error handling when no JSON files are found."""
        # Create empty directory
        temp_dir = Path(tempfile.mkdtemp())

        try:
            config = DictConfig(
                {
                    "default_split": "training",
                    "training": {
                        "path": str(temp_dir),
                    },
                    "grid": {
                        "max_grid_height": 30,
                        "max_grid_width": 30,
                        "min_grid_height": 1,
                        "min_grid_width": 1,
                        "max_colors": 10,
                        "background_color": 0,
                    },
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )

            with pytest.raises(RuntimeError, match="No JSON files found"):
                ArcAgiParser(cfg=config)
        finally:
            temp_dir.rmdir()

    def test_malformed_json_file_handling(self):
        """Test error handling for malformed JSON files."""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Create a malformed JSON file
            malformed_file = temp_dir / "malformed.json"
            with malformed_file.open("w", encoding="utf-8") as f:
                f.write("{ invalid json content")

            # Create a valid JSON file
            valid_file = temp_dir / "valid.json"
            with valid_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            config = DictConfig(
                {
                    "default_split": "training",
                    "training": {
                        "path": str(temp_dir),
                    },
                    "grid": {
                        "max_grid_height": 30,
                        "max_grid_width": 30,
                        "min_grid_height": 1,
                        "min_grid_width": 1,
                        "max_colors": 10,
                        "background_color": 0,
                    },
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )

            # Parser should skip malformed file and load valid one
            parser = ArcAgiParser(cfg=config)
            task_ids = parser.get_available_task_ids()

            # Should only have the valid task
            assert len(task_ids) == 1
            assert "valid" in task_ids

        finally:
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()

    def test_directory_structure_validation(self):
        """Test validation of directory structure."""
        # Test with non-existent directory
        config = DictConfig(
            {
                "default_split": "training",
                "training": {
                    "path": "/nonexistent/directory",
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )

        with pytest.raises(RuntimeError, match="Data directory not found"):
            ArcAgiParser(cfg=config)

    def test_missing_data_path_configuration(self):
        """Test error handling when data path is not specified."""
        config = DictConfig(
            {
                "default_split": "training",
                "training": {
                    # Missing 'path' key
                },
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )

        with pytest.raises(RuntimeError, match="No data path specified"):
            ArcAgiParser(cfg=config)

    def test_invalid_task_data_format(self, github_parser):
        """Test error handling for invalid task data format."""
        key = jax.random.PRNGKey(42)

        # Test with missing 'train' key
        invalid_data = {"test": [{"input": [[1]]}]}
        with pytest.raises(ValueError, match="Invalid task data format"):
            github_parser.preprocess_task_data(invalid_data, key)

        # Test with missing 'test' key
        invalid_data = {"train": [{"input": [[1]], "output": [[2]]}]}
        with pytest.raises(ValueError, match="Invalid task data format"):
            github_parser.preprocess_task_data(invalid_data, key)

    def test_task_data_validation(self, github_parser):
        """Test validation of task data structure."""
        key = jax.random.PRNGKey(42)

        # Test with empty training pairs
        invalid_data = {
            "train": [],
            "test": [{"input": [[1]]}],
        }
        with pytest.raises(ValueError, match="at least one training pair"):
            github_parser.preprocess_task_data(invalid_data, key)

        # Test with empty test pairs
        invalid_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [],
        }
        with pytest.raises(ValueError, match="at least one test pair"):
            github_parser.preprocess_task_data(invalid_data, key)

        # Test with missing input in training pair
        invalid_data = {
            "train": [{"output": [[2]]}],  # Missing input
            "test": [{"input": [[1]]}],
        }
        with pytest.raises(ValueError, match="missing input or output"):
            github_parser.preprocess_task_data(invalid_data, key)

        # Test with missing input in test pair
        invalid_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"output": [[3]]}],  # Missing input
        }
        with pytest.raises(ValueError, match="missing input"):
            github_parser.preprocess_task_data(invalid_data, key)

    def test_load_task_file_method(self, temp_github_directory):
        """Test the load_task_file method for individual JSON files."""
        parser_config = DictConfig(
            {
                "default_split": "training",
                "training": {"path": str(temp_github_directory)},
                "grid": {
                    "max_grid_height": 30,
                    "max_grid_width": 30,
                    "min_grid_height": 1,
                    "min_grid_width": 1,
                    "max_colors": 10,
                    "background_color": 0,
                },
                "max_train_pairs": 5,
                "max_test_pairs": 5,
            }
        )
        parser = ArcAgiParser(cfg=parser_config)

        # Test loading existing file
        task_file = temp_github_directory / "007bbfb7.json"
        task_data = parser.load_task_file(str(task_file))

        assert isinstance(task_data, dict)
        assert "train" in task_data
        assert "test" in task_data
        assert len(task_data["train"]) == 2
        assert len(task_data["test"]) == 1

        # Test loading non-existent file
        with pytest.raises(FileNotFoundError, match="Task file not found"):
            parser.load_task_file(str(temp_github_directory / "nonexistent.json"))

        # Test loading invalid JSON file
        invalid_file = temp_github_directory / "invalid.json"
        with invalid_file.open("w", encoding="utf-8") as f:
            f.write("{ invalid json")

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                parser.load_task_file(str(invalid_file))
        finally:
            invalid_file.unlink()

    def test_get_task_by_id_github_format(self, github_parser):
        """Test getting specific tasks by ID in GitHub format."""
        # Test getting existing task
        task = github_parser.get_task_by_id("007bbfb7")
        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs == 2
        assert task.num_test_pairs == 1

        # Test getting non-existent task
        with pytest.raises(ValueError, match="not found in dataset"):
            github_parser.get_task_by_id("nonexistent_task")

    def test_get_random_task_github_format(self, github_parser):
        """Test getting random tasks in GitHub format."""
        key = jax.random.PRNGKey(42)
        task = github_parser.get_random_task(key)

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs > 0
        assert task.num_test_pairs > 0

        # Test that we get different tasks with different keys
        key2 = jax.random.PRNGKey(123)
        task2 = github_parser.get_random_task(key2)

        # Tasks might be the same due to small dataset, but method should work
        assert isinstance(task2, JaxArcTask)

    def test_github_format_padding_and_masking(self, github_parser):
        """Test that padding and masking work correctly with GitHub format."""
        task = github_parser.get_task_by_id("007bbfb7")

        # Check shapes are padded correctly
        assert task.input_grids_examples.shape == (
            5,
            30,
            30,
        )  # max_train_pairs, max_height, max_width
        assert task.output_grids_examples.shape == (5, 30, 30)
        assert task.test_input_grids.shape == (
            5,
            30,
            30,
        )  # max_test_pairs, max_height, max_width

        # Check masks
        assert task.input_masks_examples.dtype == jnp.bool_
        assert task.output_masks_examples.dtype == jnp.bool_
        assert task.test_input_masks.dtype == jnp.bool_

        # Check that valid regions are correctly masked
        # First training input is 2x2
        first_input_mask = task.input_masks_examples[0]
        assert jnp.sum(first_input_mask) == 4  # 2x2 = 4 valid cells

        # Second training input is 1x2
        second_input_mask = task.input_masks_examples[1]
        assert jnp.sum(second_input_mask) == 2  # 1x2 = 2 valid cells

        # Test input is 1x2
        test_input_mask = task.test_input_masks[0]
        assert jnp.sum(test_input_mask) == 2  # 1x2 = 2 valid cells

    def test_github_format_jax_compatibility(self, github_parser):
        """Test that GitHub format parsed data is JAX-compatible."""
        task = github_parser.get_task_by_id("007bbfb7")

        # Test JIT compilation
        @jax.jit
        def sum_valid_inputs(input_grids, input_masks):
            return jnp.sum(input_grids * input_masks)

        result = sum_valid_inputs(task.input_grids_examples, task.input_masks_examples)
        assert isinstance(result, jnp.ndarray)

        # Test vmap over tasks
        def process_single_task(input_grid, input_mask):
            return jnp.sum(input_grid * input_mask)

        vmapped_process = jax.vmap(process_single_task)
        results = vmapped_process(task.input_grids_examples, task.input_masks_examples)

        assert isinstance(results, jnp.ndarray)
        assert results.shape == (5,)  # max_train_pairs

    def test_mixed_file_types_in_directory(self):
        """Test handling directory with mixed file types (only JSON should be loaded)."""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Create JSON file
            json_file = temp_dir / "task.json"
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            # Create non-JSON files
            txt_file = temp_dir / "readme.txt"
            with txt_file.open("w", encoding="utf-8") as f:
                f.write("This is not a JSON file")

            py_file = temp_dir / "script.py"
            with py_file.open("w", encoding="utf-8") as f:
                f.write("print('hello')")

            config = DictConfig(
                {
                    "default_split": "training",
                    "training": {"path": str(temp_dir)},
                    "grid": {
                        "max_grid_height": 30,
                        "max_grid_width": 30,
                        "min_grid_height": 1,
                        "min_grid_width": 1,
                        "max_colors": 10,
                        "background_color": 0,
                    },
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )

            parser = ArcAgiParser(cfg=config)
            task_ids = parser.get_available_task_ids()

            # Should only load the JSON file
            assert len(task_ids) == 1
            assert "task" in task_ids

        finally:
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()

    def test_large_dataset_loading_performance(self):
        """Test parser performance with a larger number of JSON files."""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Create multiple JSON files to simulate a larger dataset
            num_tasks = 50
            for i in range(num_tasks):
                task_file = temp_dir / f"task_{i:03d}.json"
                with task_file.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "train": [
                                {"input": [[i % 10]], "output": [[(i + 1) % 10]]},
                            ],
                            "test": [{"input": [[(i + 2) % 10]]}],
                        },
                        f,
                    )

            config = DictConfig(
                {
                    "default_split": "training",
                    "training": {"path": str(temp_dir)},
                    "grid": {
                        "max_grid_height": 30,
                        "max_grid_width": 30,
                        "min_grid_height": 1,
                        "min_grid_width": 1,
                        "max_colors": 10,
                        "background_color": 0,
                    },
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )

            # Test that parser can handle larger datasets efficiently
            start_time = time.time()
            parser = ArcAgiParser(cfg=config)
            load_time = time.time() - start_time

            # Should load all tasks
            task_ids = parser.get_available_task_ids()
            assert len(task_ids) == num_tasks

            # Loading should be reasonably fast (less than 5 seconds for 50 tasks)
            assert load_time < 5.0, f"Loading took too long: {load_time:.2f}s"

            # Test random task access
            key = jax.random.PRNGKey(42)
            task = parser.get_random_task(key)
            assert isinstance(task, JaxArcTask)

        finally:
            for file in temp_dir.glob("*.json"):
                file.unlink()
            temp_dir.rmdir()

    def test_unicode_and_special_characters_in_filenames(self):
        """Test handling of JSON files with unicode and special characters in filenames."""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Create files with various filename patterns
            test_files = {
                "normal_task.json": {
                    "train": [{"input": [[1]], "output": [[2]]}],
                    "test": [{"input": [[3]]}],
                },
                "task-with-dashes.json": {
                    "train": [{"input": [[4]], "output": [[5]]}],
                    "test": [{"input": [[6]]}],
                },
                "task_with_underscores.json": {
                    "train": [{"input": [[7]], "output": [[8]]}],
                    "test": [{"input": [[9]]}],
                },
                "123numeric.json": {
                    "train": [{"input": [[0]], "output": [[1]]}],
                    "test": [{"input": [[2]]}],
                },
            }

            for filename, task_data in test_files.items():
                task_file = temp_dir / filename
                with task_file.open("w", encoding="utf-8") as f:
                    json.dump(task_data, f)

            config = DictConfig(
                {
                    "default_split": "training",
                    "training": {"path": str(temp_dir)},
                    "grid": {
                        "max_grid_height": 30,
                        "max_grid_width": 30,
                        "min_grid_height": 1,
                        "min_grid_width": 1,
                        "max_colors": 10,
                        "background_color": 0,
                    },
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )

            parser = ArcAgiParser(cfg=config)
            task_ids = parser.get_available_task_ids()

            # Should load all files
            assert len(task_ids) == 4
            expected_ids = {
                "normal_task",
                "task-with-dashes",
                "task_with_underscores",
                "123numeric",
            }
            assert set(task_ids) == expected_ids

            # Test that we can retrieve tasks by their IDs
            for task_id in task_ids:
                task = parser.get_task_by_id(task_id)
                assert isinstance(task, JaxArcTask)

        finally:
            for file in temp_dir.glob("*.json"):
                file.unlink()
            temp_dir.rmdir()

    def test_empty_json_file_handling(self):
        """Test handling of empty JSON files."""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Create empty JSON file
            empty_file = temp_dir / "empty.json"
            with empty_file.open("w", encoding="utf-8") as f:
                json.dump({}, f)

            # Create valid JSON file
            valid_file = temp_dir / "valid.json"
            with valid_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            config = DictConfig(
                {
                    "default_split": "training",
                    "training": {"path": str(temp_dir)},
                    "grid": {
                        "max_grid_height": 30,
                        "max_grid_width": 30,
                        "min_grid_height": 1,
                        "min_grid_width": 1,
                        "max_colors": 10,
                        "background_color": 0,
                    },
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )

            # Parser should load both files (validation happens during preprocessing)
            parser = ArcAgiParser(cfg=config)
            task_ids = parser.get_available_task_ids()

            # Should have both tasks loaded
            assert len(task_ids) == 2
            assert "valid" in task_ids
            assert "empty" in task_ids

            # Valid task should work fine
            valid_task = parser.get_task_by_id("valid")
            assert isinstance(valid_task, JaxArcTask)

            # Empty task should fail during preprocessing
            with pytest.raises(ValueError, match="Invalid task data format"):
                parser.get_task_by_id("empty")

        finally:
            for file in temp_dir.glob("*.json"):
                file.unlink()
            temp_dir.rmdir()

    def test_file_permission_error_handling(self):
        """Test handling of files with permission issues."""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Create a valid JSON file
            valid_file = temp_dir / "valid.json"
            with valid_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            # Create a file with restricted permissions (if possible on this system)
            restricted_file = temp_dir / "restricted.json"
            with restricted_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "train": [{"input": [[4]], "output": [[5]]}],
                        "test": [{"input": [[6]]}],
                    },
                    f,
                )

            # Try to restrict permissions (may not work on all systems)
            try:
                restricted_file.chmod(0o000)  # No permissions
                permission_restricted = True
            except (OSError, PermissionError):
                permission_restricted = False

            config = DictConfig(
                {
                    "default_split": "training",
                    "training": {"path": str(temp_dir)},
                    "grid": {
                        "max_grid_height": 30,
                        "max_grid_width": 30,
                        "min_grid_height": 1,
                        "min_grid_width": 1,
                        "max_colors": 10,
                        "background_color": 0,
                    },
                    "max_train_pairs": 5,
                    "max_test_pairs": 5,
                }
            )

            # Parser should handle permission errors gracefully
            parser = ArcAgiParser(cfg=config)
            task_ids = parser.get_available_task_ids()

            if permission_restricted:
                # Should only load the accessible file
                assert len(task_ids) == 1
                assert "valid" in task_ids
            else:
                # If permission restriction didn't work, both files should be loaded
                assert len(task_ids) >= 1
                assert "valid" in task_ids

        finally:
            # Restore permissions before cleanup
            try:
                restricted_file.chmod(0o644)
            except (OSError, PermissionError, NameError):
                pass

            for file in temp_dir.glob("*.json"):
                try:
                    file.unlink()
                except (OSError, PermissionError):
                    pass
            temp_dir.rmdir()
