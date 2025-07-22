"""Comprehensive tests for ArcAgiParser functionality.

This test suite covers data loading, JaxArcTask creation, error handling,
edge cases, and JAX compatibility for the ArcAgiParser implementation.
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

from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.types import JaxArcTask


class TestArcAgiParserComprehensive:
    """Comprehensive test suite for ArcAgiParser."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for ArcAgiParser."""
        return DictConfig(
            {
                "default_split": "training",
                "training": {"path": "data/arc-prize-2024"},
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
    def sample_task_data(self):
        """Sample ARC task data in GitHub format."""
        return {
            "train": [
                {
                    "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                    "output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                },
                {"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]},
            ],
            "test": [
                {"input": [[4, 5, 4], [5, 4, 5]], "output": [[5, 4, 5], [4, 5, 4]]},
                {
                    "input": [[6, 7], [7, 6]]
                    # No output for second test case
                },
            ],
        }

    @pytest.fixture
    def temp_dataset_directory(self, sample_task_data):
        """Create temporary dataset directory with sample tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "arc-dataset"
            dataset_dir.mkdir()

            # Create sample task files
            task_files = {
                "00576224.json": sample_task_data,
                "007bbfb7.json": {
                    "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                    "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
                },
                "00d62c1b.json": {
                    "train": [
                        {"input": [[0, 1, 0]], "output": [[1, 0, 1]]},
                        {"input": [[2, 3, 2]], "output": [[3, 2, 3]]},
                    ],
                    "test": [{"input": [[4, 5, 4]]}],
                },
            }

            for filename, data in task_files.items():
                with (dataset_dir / filename).open("w") as f:
                    json.dump(data, f)

            yield dataset_dir

    def test_initialization_success(self, base_config, temp_dataset_directory):
        """Test successful ArcAgiParser initialization."""
        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)

            # Check that tasks were loaded
            task_ids = parser.get_available_task_ids()
            assert len(task_ids) == 3
            assert "00576224" in task_ids
            assert "007bbfb7" in task_ids
            assert "00d62c1b" in task_ids

    def test_initialization_missing_data_path(self, base_config):
        """Test initialization failure when data path is missing."""
        del base_config.training["path"]

        with pytest.raises(RuntimeError, match="No data path specified"):
            ArcAgiParser(base_config)

    def test_initialization_nonexistent_directory(self, base_config):
        """Test initialization failure with non-existent directory."""
        base_config.training.path = "/nonexistent/directory"

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/nonexistent/directory")

            with pytest.raises(RuntimeError, match="Data directory not found"):
                ArcAgiParser(base_config)

    def test_initialization_empty_directory(self, base_config):
        """Test initialization failure with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / "empty"
            empty_dir.mkdir()

            base_config.training.path = str(empty_dir)

            with patch("jaxarc.parsers.arc_agi.here") as mock_here:
                mock_here.return_value = empty_dir

                with pytest.raises(RuntimeError, match="No JSON files found"):
                    ArcAgiParser(base_config)

    def test_load_task_file_success(self, base_config, temp_dataset_directory):
        """Test successful task file loading."""
        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)

            # Test loading existing file
            task_file = temp_dataset_directory / "00576224.json"
            data = parser.load_task_file(str(task_file))

            assert isinstance(data, dict)
            assert "train" in data
            assert "test" in data
            assert len(data["train"]) == 2
            assert len(data["test"]) == 2

    def test_load_task_file_not_found(self, base_config, temp_dataset_directory):
        """Test load_task_file with non-existent file."""
        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)

            with pytest.raises(FileNotFoundError, match="Task file not found"):
                parser.load_task_file("/nonexistent/file.json")

    def test_load_task_file_invalid_json(self, base_config, temp_dataset_directory):
        """Test load_task_file with invalid JSON."""
        # Create invalid JSON file
        invalid_file = temp_dataset_directory / "invalid.json"
        with invalid_file.open("w") as f:
            f.write("{ invalid json content")

        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)

            with pytest.raises(ValueError, match="Invalid JSON"):
                parser.load_task_file(str(invalid_file))

    def test_preprocess_task_data_success(self, base_config, sample_task_data):
        """Test successful task data preprocessing."""
        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ArcAgiParser, "_load_and_cache_tasks"):
                parser = ArcAgiParser(base_config)

        key = jax.random.PRNGKey(42)
        task = parser.preprocess_task_data(sample_task_data, key, "test_task")

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs == 2
        assert task.num_test_pairs == 2

        # Check array shapes
        assert task.input_grids_examples.shape == (5, 30, 30)
        assert task.output_grids_examples.shape == (5, 30, 30)
        assert task.test_input_grids.shape == (3, 30, 30)
        assert task.true_test_output_grids.shape == (3, 30, 30)

        # Check data types
        assert task.input_grids_examples.dtype == jnp.int32
        assert task.input_masks_examples.dtype == jnp.bool_
        assert task.output_masks_examples.dtype == jnp.bool_
        assert task.test_input_masks.dtype == jnp.bool_

    def test_preprocess_task_data_invalid_format(self, base_config):
        """Test preprocessing with invalid task data format."""
        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ArcAgiParser, "_load_and_cache_tasks"):
                parser = ArcAgiParser(base_config)

        key = jax.random.PRNGKey(42)

        # Test with missing 'train' key
        invalid_data = {"test": [{"input": [[1]]}]}
        with pytest.raises(ValueError, match="Invalid task data format"):
            parser.preprocess_task_data(invalid_data, key)

        # Test with missing 'test' key
        invalid_data = {"train": [{"input": [[1]], "output": [[2]]}]}
        with pytest.raises(ValueError, match="Invalid task data format"):
            parser.preprocess_task_data(invalid_data, key)

        # Test with non-dict input
        with pytest.raises(ValueError, match="Expected dict"):
            parser.preprocess_task_data("invalid", key)

    def test_preprocess_task_data_empty_sections(self, base_config):
        """Test preprocessing with empty train/test sections."""
        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ArcAgiParser, "_load_and_cache_tasks"):
                parser = ArcAgiParser(base_config)

        key = jax.random.PRNGKey(42)

        # Empty training pairs
        invalid_data = {"train": [], "test": [{"input": [[1]]}]}
        with pytest.raises(ValueError, match="at least one training pair"):
            parser.preprocess_task_data(invalid_data, key)

        # Empty test pairs
        invalid_data = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        with pytest.raises(ValueError, match="at least one test pair"):
            parser.preprocess_task_data(invalid_data, key)

    def test_preprocess_task_data_missing_input_output(self, base_config):
        """Test preprocessing with missing input/output data."""
        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ArcAgiParser, "_load_and_cache_tasks"):
                parser = ArcAgiParser(base_config)

        key = jax.random.PRNGKey(42)

        # Missing input in training pair
        invalid_data = {"train": [{"output": [[1]]}], "test": [{"input": [[2]]}]}
        with pytest.raises(ValueError, match="missing input or output"):
            parser.preprocess_task_data(invalid_data, key)

        # Missing output in training pair
        invalid_data = {"train": [{"input": [[1]]}], "test": [{"input": [[2]]}]}
        with pytest.raises(ValueError, match="missing input or output"):
            parser.preprocess_task_data(invalid_data, key)

        # Missing input in test pair
        invalid_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"output": [[3]]}],
        }
        with pytest.raises(ValueError, match="missing input"):
            parser.preprocess_task_data(invalid_data, key)

    def test_get_random_task_success(self, base_config, temp_dataset_directory):
        """Test successful random task retrieval."""
        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)
            key = jax.random.PRNGKey(42)

            task = parser.get_random_task(key)
            assert isinstance(task, JaxArcTask)
            assert task.num_train_pairs > 0
            assert task.num_test_pairs > 0

    def test_get_random_task_no_tasks(self, base_config):
        """Test get_random_task with no available tasks."""
        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ArcAgiParser, "_load_and_cache_tasks"):
                parser = ArcAgiParser(base_config)
                parser._task_ids = []  # No tasks available

        key = jax.random.PRNGKey(42)
        with pytest.raises(RuntimeError, match="No tasks available"):
            parser.get_random_task(key)

    def test_get_task_by_id_success(self, base_config, temp_dataset_directory):
        """Test successful task retrieval by ID."""
        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)

            task = parser.get_task_by_id("00576224")
            assert isinstance(task, JaxArcTask)
            assert task.num_train_pairs == 2
            assert task.num_test_pairs == 2

    def test_get_task_by_id_not_found(self, base_config, temp_dataset_directory):
        """Test get_task_by_id with non-existent task ID."""
        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)

            with pytest.raises(ValueError, match="not found in dataset"):
                parser.get_task_by_id("nonexistent_task")

    def test_get_available_task_ids(self, base_config, temp_dataset_directory):
        """Test getting list of available task IDs."""
        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)
            task_ids = parser.get_available_task_ids()

            assert isinstance(task_ids, list)
            assert len(task_ids) == 3
            assert "00576224" in task_ids
            assert "007bbfb7" in task_ids
            assert "00d62c1b" in task_ids

    def test_task_id_extraction_from_filename(
        self, base_config, temp_dataset_directory
    ):
        """Test that task IDs are correctly extracted from filenames."""
        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)
            task_ids = parser.get_available_task_ids()

            # Task IDs should be filename stems (without .json extension)
            for task_id in task_ids:
                assert not task_id.endswith(".json")
                assert len(task_id) == 8  # ARC task IDs are typically 8 characters

    def test_malformed_json_handling(self, base_config):
        """Test handling of malformed JSON files during initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()

            # Create malformed JSON file
            malformed_file = dataset_dir / "malformed.json"
            with malformed_file.open("w") as f:
                f.write("{ invalid json content")

            # Create valid JSON file
            valid_file = dataset_dir / "valid.json"
            with valid_file.open("w") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            base_config.training.path = str(dataset_dir)

            with patch("jaxarc.parsers.arc_agi.here") as mock_here:
                mock_here.return_value = dataset_dir

                # Should skip malformed file and load valid one
                parser = ArcAgiParser(base_config)
                task_ids = parser.get_available_task_ids()

                assert len(task_ids) == 1
                assert "valid" in task_ids

    def test_grid_dimension_validation(self, base_config):
        """Test validation of grid dimensions during preprocessing."""
        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ArcAgiParser, "_load_and_cache_tasks"):
                parser = ArcAgiParser(base_config)

        key = jax.random.PRNGKey(42)

        # Create task with oversized grid
        oversized_task = {
            "train": [
                {
                    "input": [[0] * 35] * 35,  # 35x35 grid (exceeds 30x30 limit)
                    "output": [[1] * 35] * 35,
                }
            ],
            "test": [{"input": [[2] * 35] * 35}],
        }

        with pytest.raises(ValueError, match="exceed maximum"):
            parser.preprocess_task_data(oversized_task, key)

    def test_color_validation(self, base_config):
        """Test validation of color values during preprocessing."""
        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ArcAgiParser, "_load_and_cache_tasks"):
                parser = ArcAgiParser(base_config)

        key = jax.random.PRNGKey(42)

        # Create task with invalid color values
        invalid_color_task = {
            "train": [
                {
                    "input": [[15, 20]],  # Colors > max_colors (10)
                    "output": [[25, 30]],
                }
            ],
            "test": [{"input": [[35, 40]]}],
        }

        with pytest.raises(ValueError, match="Invalid color in grid"):
            parser.preprocess_task_data(invalid_color_task, key)

    def test_padding_and_masking(self, base_config, sample_task_data):
        """Test that grids are properly padded and masked."""
        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ArcAgiParser, "_load_and_cache_tasks"):
                parser = ArcAgiParser(base_config)

        key = jax.random.PRNGKey(42)
        task = parser.preprocess_task_data(sample_task_data, key, "test_task")

        # Check that arrays are padded to maximum dimensions
        assert task.input_grids_examples.shape == (5, 30, 30)
        assert task.output_grids_examples.shape == (5, 30, 30)
        assert task.test_input_grids.shape == (3, 30, 30)

        # Check that masks indicate valid regions
        # First training input is 3x3
        first_input_mask = task.input_masks_examples[0]
        assert jnp.sum(first_input_mask) == 9  # 3x3 = 9 valid cells

        # Second training input is 2x2
        second_input_mask = task.input_masks_examples[1]
        assert jnp.sum(second_input_mask) == 4  # 2x2 = 4 valid cells

        # Unused training slots should be completely masked
        unused_mask = task.input_masks_examples[2]
        assert jnp.sum(unused_mask) == 0  # No valid cells

    def test_jax_compatibility(self, base_config, temp_dataset_directory):
        """Test JAX compatibility of parsed data."""
        base_config.training.path = str(temp_dataset_directory)

        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = temp_dataset_directory

            parser = ArcAgiParser(base_config)
            key = jax.random.PRNGKey(42)
            task = parser.get_random_task(key)

        # Test JIT compilation
        @jax.jit
        def sum_valid_inputs(input_grids, input_masks):
            return jnp.sum(input_grids * input_masks)

        result = sum_valid_inputs(task.input_grids_examples, task.input_masks_examples)
        assert isinstance(result, jnp.ndarray)

        # Test vmap over training pairs
        def process_single_pair(input_grid, input_mask):
            return jnp.sum(input_grid * input_mask)

        vmapped_process = jax.vmap(process_single_pair)
        results = vmapped_process(task.input_grids_examples, task.input_masks_examples)

        assert isinstance(results, jnp.ndarray)
        assert results.shape == (5,)  # max_train_pairs

    def test_deterministic_preprocessing(self, base_config, sample_task_data):
        """Test that preprocessing is deterministic for the same input."""
        with patch("jaxarc.parsers.arc_agi.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ArcAgiParser, "_load_and_cache_tasks"):
                parser = ArcAgiParser(base_config)

        key = jax.random.PRNGKey(42)

        # Process the same data twice
        task1 = parser.preprocess_task_data(sample_task_data, key, "test_task")
        task2 = parser.preprocess_task_data(sample_task_data, key, "test_task")

        # Results should be identical
        assert jnp.array_equal(task1.input_grids_examples, task2.input_grids_examples)
        assert jnp.array_equal(task1.output_grids_examples, task2.output_grids_examples)
        assert jnp.array_equal(task1.input_masks_examples, task2.input_masks_examples)

    def test_legacy_kaggle_format_detection(self, base_config):
        """Test detection and rejection of legacy Kaggle format."""
        # Create config that looks like legacy Kaggle format
        legacy_config = base_config.copy()
        legacy_config.training = {
            "challenges": "path/to/challenges.json",
            "solutions": "path/to/solutions.json",
        }

        with pytest.raises(RuntimeError, match="Legacy Kaggle format detected"):
            ArcAgiParser(legacy_config)

    def test_unicode_filename_handling(self, base_config):
        """Test handling of files with unicode characters in names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()

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
                "123numeric.json": {
                    "train": [{"input": [[7]], "output": [[8]]}],
                    "test": [{"input": [[9]]}],
                },
            }

            for filename, data in test_files.items():
                with (dataset_dir / filename).open("w") as f:
                    json.dump(data, f)

            base_config.training.path = str(dataset_dir)

            with patch("jaxarc.parsers.arc_agi.here") as mock_here:
                mock_here.return_value = dataset_dir

                parser = ArcAgiParser(base_config)
                task_ids = parser.get_available_task_ids()

                assert len(task_ids) == 3
                expected_ids = {"normal_task", "task-with-dashes", "123numeric"}
                assert set(task_ids) == expected_ids

    def test_large_dataset_performance(self, base_config):
        """Test parser performance with larger number of tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()

            # Create multiple task files
            num_tasks = 100
            for i in range(num_tasks):
                task_file = dataset_dir / f"task_{i:03d}.json"
                with task_file.open("w") as f:
                    json.dump(
                        {
                            "train": [
                                {"input": [[i % 10]], "output": [[(i + 1) % 10]]}
                            ],
                            "test": [{"input": [[(i + 2) % 10]]}],
                        },
                        f,
                    )

            base_config.training.path = str(dataset_dir)

            with patch("jaxarc.parsers.arc_agi.here") as mock_here:
                mock_here.return_value = dataset_dir

                # Should handle large dataset efficiently
                parser = ArcAgiParser(base_config)
                task_ids = parser.get_available_task_ids()

                assert len(task_ids) == num_tasks

                # Test random access
                key = jax.random.PRNGKey(42)
                task = parser.get_random_task(key)
                assert isinstance(task, JaxArcTask)

    def test_mixed_file_types_in_directory(self, base_config):
        """Test that only JSON files are loaded from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()

            # Create JSON file
            json_file = dataset_dir / "task.json"
            with json_file.open("w") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            # Create non-JSON files
            (dataset_dir / "readme.txt").write_text("This is not a JSON file")
            (dataset_dir / "script.py").write_text("print('hello')")
            (dataset_dir / "data.csv").write_text("col1,col2\n1,2")

            base_config.training.path = str(dataset_dir)

            with patch("jaxarc.parsers.arc_agi.here") as mock_here:
                mock_here.return_value = dataset_dir

                parser = ArcAgiParser(base_config)
                task_ids = parser.get_available_task_ids()

                # Should only load the JSON file
                assert len(task_ids) == 1
                assert "task" in task_ids

    def test_empty_json_file_handling(self, base_config):
        """Test handling of empty JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()

            # Create empty JSON file
            empty_file = dataset_dir / "empty.json"
            with empty_file.open("w") as f:
                json.dump({}, f)

            # Create valid JSON file
            valid_file = dataset_dir / "valid.json"
            with valid_file.open("w") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            base_config.training.path = str(dataset_dir)

            with patch("jaxarc.parsers.arc_agi.here") as mock_here:
                mock_here.return_value = dataset_dir

                parser = ArcAgiParser(base_config)
                task_ids = parser.get_available_task_ids()

                # Both files should be loaded (validation happens during preprocessing)
                assert len(task_ids) == 2
                assert "valid" in task_ids
                assert "empty" in task_ids

                # Valid task should work
                valid_task = parser.get_task_by_id("valid")
                assert isinstance(valid_task, JaxArcTask)

                # Empty task should fail during preprocessing
                with pytest.raises(ValueError, match="Invalid task data format"):
                    parser.get_task_by_id("empty")
