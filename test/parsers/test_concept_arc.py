"""Tests for ConceptARC dataset parser.

This module tests the ConceptArcParser class to ensure proper loading,
concept group organization, and JAX compatibility for ConceptARC datasets.
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
from jaxarc.parsers.concept_arc import ConceptArcParser
from jaxarc.types import JaxArcTask


class TestConceptArcParser:
    """Test suite for ConceptArcParser class."""

    @pytest.fixture
    def valid_config(self) -> DatasetConfig:
        """Provide a valid DatasetConfig for ConceptARC testing."""
        return DatasetConfig(
            dataset_path="test/data/concept-arc",
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
        """Provide sample ConceptARC task data."""
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
    def temp_concept_arc_dataset(self, sample_task_data: dict) -> Path:
        """Create a temporary ConceptARC dataset structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create ConceptARC directory structure
            corpus_dir = temp_path / "corpus"
            corpus_dir.mkdir(parents=True)

            # Create concept group directories with tasks
            concept_groups = {
                "AboveBelow": {
                    "task_001.json": sample_task_data,
                    "task_002.json": {
                        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                        "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
                    },
                },
                "Center": {
                    "task_003.json": {
                        "train": [{"input": [[0]], "output": [[1]]}],
                        "test": [{"input": [[2]], "output": [[3]]}],
                    }
                },
                "Copy": {
                    "task_004.json": {
                        "train": [{"input": [[5, 6]], "output": [[5, 6]]}],
                        "test": [{"input": [[7, 8]], "output": [[7, 8]]}],
                    }
                },
            }

            for concept_name, tasks in concept_groups.items():
                concept_dir = corpus_dir / concept_name
                concept_dir.mkdir()

                for filename, data in tasks.items():
                    task_file = concept_dir / filename
                    with task_file.open("w") as f:
                        json.dump(data, f)

            yield temp_path

    @pytest.fixture
    def mock_parser_with_data(
        self, valid_config: DatasetConfig, temp_concept_arc_dataset: Path
    ) -> ConceptArcParser:
        """Create a ConceptArcParser with mocked data loading."""
        config = DatasetConfig(
            dataset_path=str(temp_concept_arc_dataset),
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

        with patch("pyprojroot.here", return_value=temp_concept_arc_dataset):
            return ConceptArcParser(config)

    def test_initialization_success(self, temp_concept_arc_dataset: Path):
        """Test successful initialization with valid dataset."""
        config = DatasetConfig(
            dataset_path=str(temp_concept_arc_dataset),
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

        with patch("pyprojroot.here", return_value=temp_concept_arc_dataset):
            parser = ConceptArcParser(config)

            assert len(parser._all_task_ids) == 4
            assert len(parser._concept_groups) == 3
            assert "AboveBelow" in parser._concept_groups
            assert "Center" in parser._concept_groups
            assert "Copy" in parser._concept_groups

    def test_initialization_missing_corpus_directory(self, valid_config: DatasetConfig):
        """Test initialization with missing corpus directory."""
        with patch("pyprojroot.here", return_value=Path("/nonexistent/path")):
            # With lazy loading, missing directory raises RuntimeError during scan
            with pytest.raises(RuntimeError, match="corpus directory not found"):
                ConceptArcParser(valid_config)

    def test_get_data_path(self, valid_config: DatasetConfig):
        """Test get_data_path returns correct corpus path."""
        with patch.object(ConceptArcParser, "_scan_available_tasks"):
            parser = ConceptArcParser(valid_config)
            path = parser.get_data_path()
            assert path == "test/data/concept-arc/corpus"

    def test_discover_concept_groups(self, mock_parser_with_data: ConceptArcParser):
        """Test concept group discovery from directory structure."""
        concept_groups = mock_parser_with_data.get_concept_groups()

        assert len(concept_groups) == 3
        assert "AboveBelow" in concept_groups
        assert "Center" in concept_groups
        assert "Copy" in concept_groups

    def test_get_tasks_in_concept(self, mock_parser_with_data: ConceptArcParser):
        """Test getting tasks from specific concept groups."""
        above_below_tasks = mock_parser_with_data.get_tasks_in_concept("AboveBelow")
        assert len(above_below_tasks) == 2
        assert "AboveBelow/task_001" in above_below_tasks
        assert "AboveBelow/task_002" in above_below_tasks

        center_tasks = mock_parser_with_data.get_tasks_in_concept("Center")
        assert len(center_tasks) == 1
        assert "Center/task_003" in center_tasks

    def test_get_tasks_in_concept_invalid(
        self, mock_parser_with_data: ConceptArcParser
    ):
        """Test getting tasks from non-existent concept group."""
        with pytest.raises(ValueError, match="not found"):
            mock_parser_with_data.get_tasks_in_concept("NonExistentConcept")

    def test_load_task_file_success(
        self, mock_parser_with_data: ConceptArcParser, temp_concept_arc_dataset: Path
    ):
        """Test successful task file loading."""
        task_file = temp_concept_arc_dataset / "corpus" / "AboveBelow" / "task_001.json"
        result = mock_parser_with_data.load_task_file(str(task_file))

        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result

    def test_load_task_file_not_found(self, mock_parser_with_data: ConceptArcParser):
        """Test load_task_file with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Task file not found"):
            mock_parser_with_data.load_task_file("/nonexistent/file.json")

    def test_preprocess_task_data_with_tuple(
        self, mock_parser_with_data: ConceptArcParser, sample_task_data: dict
    ):
        """Test preprocessing with (task_id, task_content) tuple."""
        key = jax.random.PRNGKey(42)
        result = mock_parser_with_data.preprocess_task_data(
            ("test_task", sample_task_data), key
        )

        assert isinstance(result, JaxArcTask)
        assert result.num_train_pairs == 2
        assert result.num_test_pairs == 1

    def test_preprocess_task_data_direct_content(
        self, mock_parser_with_data: ConceptArcParser, sample_task_data: dict
    ):
        """Test preprocessing with direct task content."""
        key = jax.random.PRNGKey(42)
        result = mock_parser_with_data.preprocess_task_data(sample_task_data, key)

        assert isinstance(result, JaxArcTask)
        assert result.num_train_pairs == 2
        assert result.num_test_pairs == 1

    def test_get_random_task_success(self, mock_parser_with_data: ConceptArcParser):
        """Test successful random task retrieval."""
        key = jax.random.PRNGKey(42)
        task = mock_parser_with_data.get_random_task(key)

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs >= 1
        assert task.num_test_pairs >= 1

    def test_get_random_task_no_tasks(self, valid_config: DatasetConfig):
        """Test get_random_task with no available tasks."""
        with patch.object(ConceptArcParser, "_scan_available_tasks"):
            parser = ConceptArcParser(valid_config)
            parser._all_task_ids = []  # No tasks available

            key = jax.random.PRNGKey(42)
            with pytest.raises(RuntimeError, match="No tasks available"):
                parser.get_random_task(key)

    def test_get_random_task_from_concept_success(
        self, mock_parser_with_data: ConceptArcParser
    ):
        """Test successful random task retrieval from specific concept."""
        key = jax.random.PRNGKey(42)
        task = mock_parser_with_data.get_random_task_from_concept("AboveBelow", key)

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs >= 1
        assert task.num_test_pairs >= 1

    def test_get_random_task_from_concept_invalid(
        self, mock_parser_with_data: ConceptArcParser
    ):
        """Test get_random_task_from_concept with invalid concept."""
        key = jax.random.PRNGKey(42)
        with pytest.raises(ValueError, match="not found"):
            mock_parser_with_data.get_random_task_from_concept("InvalidConcept", key)

    def test_get_random_task_from_concept_no_tasks(
        self, mock_parser_with_data: ConceptArcParser
    ):
        """Test get_random_task_from_concept with concept having no tasks."""
        # Manually create empty concept group
        mock_parser_with_data._concept_groups["EmptyConcept"] = []

        key = jax.random.PRNGKey(42)
        with pytest.raises(RuntimeError, match="No tasks available in concept"):
            mock_parser_with_data.get_random_task_from_concept("EmptyConcept", key)

    def test_get_task_by_id_success(self, mock_parser_with_data: ConceptArcParser):
        """Test successful task retrieval by ID."""
        task = mock_parser_with_data.get_task_by_id("AboveBelow/task_001")

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs >= 1
        assert task.num_test_pairs >= 1

    def test_get_task_by_id_not_found(self, mock_parser_with_data: ConceptArcParser):
        """Test get_task_by_id with non-existent task ID."""
        with pytest.raises(ValueError, match="not found in ConceptARC dataset"):
            mock_parser_with_data.get_task_by_id("NonExistent/task")

    def test_get_available_task_ids(self, mock_parser_with_data: ConceptArcParser):
        """Test get_available_task_ids returns correct list."""
        task_ids = mock_parser_with_data.get_available_task_ids()

        assert isinstance(task_ids, list)
        assert len(task_ids) == 4
        assert "AboveBelow/task_001" in task_ids
        assert "AboveBelow/task_002" in task_ids
        assert "Center/task_003" in task_ids
        assert "Copy/task_004" in task_ids

        # Ensure it returns a copy
        task_ids.append("new_task")
        original_ids = mock_parser_with_data.get_available_task_ids()
        assert "new_task" not in original_ids

    def test_get_task_metadata(self, mock_parser_with_data: ConceptArcParser):
        """Test get_task_metadata returns correct information."""
        # Lazy loading: Load task first to populate full metadata (num_demonstrations, num_test_inputs)
        # get_task_by_id triggers _load_task_from_disk which populates these fields
        _ = mock_parser_with_data.get_task_by_id("AboveBelow/task_001")

        metadata = mock_parser_with_data.get_task_metadata("AboveBelow/task_001")

        assert isinstance(metadata, dict)
        assert metadata["concept_group"] == "AboveBelow"
        assert metadata["task_name"] == "task_001"
        assert "file_path" in metadata
        assert "num_demonstrations" in metadata
        assert "num_test_inputs" in metadata

    def test_get_task_metadata_not_found(self, mock_parser_with_data: ConceptArcParser):
        """Test get_task_metadata with non-existent task ID."""
        with pytest.raises(ValueError, match="not found in ConceptARC dataset"):
            mock_parser_with_data.get_task_metadata("NonExistent/task")

    def test_get_dataset_statistics(self, mock_parser_with_data: ConceptArcParser):
        """Test get_dataset_statistics returns comprehensive information."""
        stats = mock_parser_with_data.get_dataset_statistics()

        assert isinstance(stats, dict)
        assert stats["total_tasks"] == 4
        assert stats["total_concept_groups"] == 3
        assert "concept_groups" in stats

        # Check concept group statistics
        concept_stats = stats["concept_groups"]
        assert "AboveBelow" in concept_stats
        assert "Center" in concept_stats
        assert "Copy" in concept_stats

        # Check AboveBelow statistics
        above_below_stats = concept_stats["AboveBelow"]
        assert above_below_stats["num_tasks"] == 2
        assert len(above_below_stats["tasks"]) == 2

    def test_jax_compatibility(self, mock_parser_with_data: ConceptArcParser):
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

    def test_array_shapes_and_padding(self, mock_parser_with_data: ConceptArcParser):
        """Test that arrays are properly padded to maximum dimensions."""
        task = mock_parser_with_data.get_task_by_id("AboveBelow/task_001")

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

    def test_concept_based_sampling_consistency(
        self, mock_parser_with_data: ConceptArcParser
    ):
        """Test that concept-based sampling returns tasks from correct concept."""
        key = jax.random.PRNGKey(42)

        # Sample multiple tasks from AboveBelow concept
        for _ in range(5):
            task = mock_parser_with_data.get_random_task_from_concept("AboveBelow", key)
            assert isinstance(task, JaxArcTask)
            # We can't directly check which task was selected, but we can verify it's valid
            assert task.num_train_pairs >= 1
            assert task.num_test_pairs >= 1

    def test_error_handling_corrupted_json(
        self, temp_concept_arc_dataset: Path, valid_config: DatasetConfig
    ):
        """Test error handling with corrupted JSON files."""
        # Create a corrupted JSON file in a concept directory
        concept_dir = temp_concept_arc_dataset / "corpus" / "AboveBelow"
        corrupted_file = concept_dir / "corrupted.json"
        with corrupted_file.open("w") as f:
            f.write("{ invalid json")

        config = DatasetConfig(
            dataset_path=str(temp_concept_arc_dataset),
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

        with patch("pyprojroot.here", return_value=temp_concept_arc_dataset):
            # With lazy loading, corrupted files are detected during scan (all .json files found)
            # but error occurs when trying to load the corrupted task
            parser = ConceptArcParser(config)
            # All JSON files in AboveBelow directory are scanned: task_001, task_002, corrupted
            # Plus task_003 in Center and task_004 in Copy = 5 total
            assert len(parser._all_task_ids) == 5

            # Attempting to load the corrupted task should raise an error
            with pytest.raises(ValueError, match="Invalid JSON"):
                parser.get_task_by_id("AboveBelow/corrupted")

    def test_deterministic_preprocessing(self, mock_parser_with_data: ConceptArcParser):
        """Test that preprocessing is deterministic for the same input."""
        # Lazy loading: load task first to populate cache
        _ = mock_parser_with_data.get_task_by_id("AboveBelow/task_001")
        task_data = mock_parser_with_data._cached_tasks["AboveBelow/task_001"]

        # Process the same data multiple times
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(42)  # Same key

        result1 = mock_parser_with_data.preprocess_task_data(
            ("test_task", task_data), key1
        )
        result2 = mock_parser_with_data.preprocess_task_data(
            ("test_task", task_data), key2
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

    def test_from_hydra_class_method(self, valid_config: DatasetConfig):
        """Test from_hydra class method creates parser correctly."""
        mock_hydra_config = Mock()

        with patch.object(DatasetConfig, "from_hydra", return_value=valid_config):
            with patch.object(ConceptArcParser, "_scan_available_tasks"):
                parser = ConceptArcParser.from_hydra(mock_hydra_config)

                assert isinstance(parser, ConceptArcParser)
                assert parser.config == valid_config
