"""Comprehensive tests for ConceptArcParser functionality.

This test suite covers concept group organization, task loading from hierarchical
structure, concept-based sampling, error handling, and format compatibility validation.
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

from jaxarc.parsers.concept_arc import ConceptArcParser
from jaxarc.types import JaxArcTask


class TestConceptArcParserComprehensive:
    """Comprehensive test suite for ConceptArcParser."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for ConceptArcParser."""
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
                "corpus": {
                    "path": "data/raw/ConceptARC/corpus",
                    "concept_groups": [
                        "AboveBelow",
                        "Center",
                        "CleanUp",
                        "CompleteShape",
                        "Copy",
                        "Count",
                        "ExtendToBoundary",
                        "ExtractObjects",
                    ],
                },
            }
        )

    @pytest.fixture
    def sample_concept_task(self):
        """Sample ConceptARC task data."""
        return {
            "train": [
                {
                    "input": [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
                    "output": [[1, 0, 1], [0, 2, 0], [1, 0, 1]],
                },
                {"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]},
            ],
            "test": [
                {
                    "input": [[4, 5, 4], [5, 6, 5], [4, 5, 4]],
                    "output": [[5, 4, 5], [4, 6, 4], [5, 4, 5]],
                },
                {
                    "input": [[7, 8], [8, 7]]
                    # No output for second test case
                },
            ],
        }

    @pytest.fixture
    def temp_concept_arc_directory(self, sample_concept_task):
        """Create temporary ConceptARC directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"

            # Create concept group directories
            concept_groups = ["AboveBelow", "Center", "Copy", "Count"]

            for concept in concept_groups:
                concept_dir = corpus_dir / concept
                concept_dir.mkdir(parents=True)

                # Create task files for each concept
                for i in range(3):  # 3 tasks per concept
                    task_file = concept_dir / f"task_{i:03d}.json"
                    with task_file.open("w") as f:
                        json.dump(sample_concept_task, f)

            yield temp_dir

    def test_initialization_success(self, base_config, temp_concept_arc_directory):
        """Test successful ConceptArcParser initialization."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)

            # Check concept groups were discovered
            concept_groups = parser.get_concept_groups()
            assert len(concept_groups) == 4
            assert "AboveBelow" in concept_groups
            assert "Center" in concept_groups
            assert "Copy" in concept_groups
            assert "Count" in concept_groups

            # Check tasks were loaded
            all_tasks = parser.get_available_task_ids()
            assert len(all_tasks) == 12  # 4 concepts × 3 tasks each

    def test_initialization_missing_corpus_path(self, base_config):
        """Test initialization failure when corpus path is missing."""
        del base_config.corpus["path"]

        with pytest.raises(ValueError, match="ConceptARC corpus path not specified"):
            ConceptArcParser(base_config)

    def test_initialization_nonexistent_corpus(self, base_config):
        """Test initialization with non-existent corpus directory."""
        base_config.corpus.path = "/nonexistent/corpus"

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path("/nonexistent/corpus")

            # Should not raise exception but log warning
            parser = ConceptArcParser(base_config)

            # Should have no tasks or concept groups
            assert len(parser.get_available_task_ids()) == 0
            assert len(parser.get_concept_groups()) == 0

    def test_concept_group_discovery(self, base_config, temp_concept_arc_directory):
        """Test automatic discovery of concept groups from directory structure."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)

            # Check discovered concept groups
            concept_groups = parser.get_concept_groups()
            expected_groups = {"AboveBelow", "Center", "Copy", "Count"}
            assert set(concept_groups) == expected_groups

            # Check tasks in each concept group
            for concept in concept_groups:
                tasks = parser.get_tasks_in_concept(concept)
                assert len(tasks) == 3
                for task_id in tasks:
                    assert task_id.startswith(f"{concept}/")

    def test_concept_group_validation_against_expected(
        self, base_config, temp_concept_arc_directory
    ):
        """Test validation of discovered concept groups against expected list."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        # Set expected concept groups (some missing, some extra)
        base_config.corpus.concept_groups = [
            "AboveBelow",
            "Center",
            "Copy",
            "Count",
            "MissingConcept1",
            "MissingConcept2",  # These don't exist
        ]

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            # Should still initialize successfully but log warnings
            parser = ConceptArcParser(base_config)

            # Should have found the existing concept groups
            concept_groups = parser.get_concept_groups()
            assert len(concept_groups) == 4

    def test_task_id_format(self, base_config, temp_concept_arc_directory):
        """Test that task IDs follow concept_group/task_name format."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)
            task_ids = parser.get_available_task_ids()

            for task_id in task_ids:
                # Should be in format "ConceptGroup/task_name"
                assert "/" in task_id
                concept, task_name = task_id.split("/", 1)
                assert concept in ["AboveBelow", "Center", "Copy", "Count"]
                assert task_name.startswith("task_")

    def test_get_random_task_from_concept(
        self, base_config, temp_concept_arc_directory
    ):
        """Test getting random task from specific concept group."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)
            key = jax.random.PRNGKey(42)

            # Test getting task from specific concept
            task = parser.get_random_task_from_concept("AboveBelow", key)
            assert isinstance(task, JaxArcTask)
            assert task.num_train_pairs == 2
            assert task.num_test_pairs == 2

    def test_get_random_task_from_invalid_concept(
        self, base_config, temp_concept_arc_directory
    ):
        """Test error handling for invalid concept group."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)
            key = jax.random.PRNGKey(42)

            with pytest.raises(
                ValueError, match="Concept group 'InvalidConcept' not found"
            ):
                parser.get_random_task_from_concept("InvalidConcept", key)

    def test_get_tasks_in_concept(self, base_config, temp_concept_arc_directory):
        """Test getting all tasks in a specific concept group."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)

            # Test getting tasks from existing concept
            tasks = parser.get_tasks_in_concept("Center")
            assert len(tasks) == 3
            assert all(task_id.startswith("Center/") for task_id in tasks)

            # Test getting tasks from invalid concept
            with pytest.raises(
                ValueError, match="Concept group 'InvalidConcept' not found"
            ):
                parser.get_tasks_in_concept("InvalidConcept")

    def test_get_random_task_global(self, base_config, temp_concept_arc_directory):
        """Test getting random task from entire dataset."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)
            key = jax.random.PRNGKey(123)

            task = parser.get_random_task(key)
            assert isinstance(task, JaxArcTask)
            assert task.num_train_pairs == 2
            assert task.num_test_pairs == 2

    def test_get_random_task_no_tasks(self, base_config):
        """Test get_random_task with no available tasks."""
        base_config.corpus.path = "/nonexistent/corpus"

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path("/nonexistent/corpus")

            parser = ConceptArcParser(base_config)
            key = jax.random.PRNGKey(42)

            with pytest.raises(
                RuntimeError, match="No tasks available in ConceptARC dataset"
            ):
                parser.get_random_task(key)

    def test_get_task_by_id(self, base_config, temp_concept_arc_directory):
        """Test getting specific task by ID."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)
            task_ids = parser.get_available_task_ids()

            # Test getting existing task
            task_id = task_ids[0]
            task = parser.get_task_by_id(task_id)
            assert isinstance(task, JaxArcTask)

            # Test getting non-existent task
            with pytest.raises(ValueError, match="not found in ConceptARC dataset"):
                parser.get_task_by_id("NonExistent/task")

    def test_task_metadata(self, base_config, temp_concept_arc_directory):
        """Test task metadata functionality."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)
            task_ids = parser.get_available_task_ids()

            # Test getting metadata for existing task
            task_id = task_ids[0]
            metadata = parser.get_task_metadata(task_id)

            assert "concept_group" in metadata
            assert "task_name" in metadata
            assert "file_path" in metadata
            assert "num_demonstrations" in metadata
            assert "num_test_inputs" in metadata

            # Check metadata values
            concept, task_name = task_id.split("/", 1)
            assert metadata["concept_group"] == concept
            assert metadata["task_name"] == task_name
            assert metadata["num_demonstrations"] == 2
            assert metadata["num_test_inputs"] == 2

            # Test getting metadata for non-existent task
            with pytest.raises(ValueError, match="not found in ConceptARC dataset"):
                parser.get_task_metadata("NonExistent/task")

    def test_dataset_statistics(self, base_config, temp_concept_arc_directory):
        """Test dataset statistics calculation."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)
            stats = parser.get_dataset_statistics()

            # Check overall statistics
            assert stats["total_tasks"] == 12
            assert stats["total_concept_groups"] == 4
            assert "concept_groups" in stats

            # Check concept group statistics
            for concept in ["AboveBelow", "Center", "Copy", "Count"]:
                assert concept in stats["concept_groups"]
                concept_stats = stats["concept_groups"][concept]

                assert concept_stats["num_tasks"] == 3
                assert len(concept_stats["tasks"]) == 3
                assert concept_stats["avg_demonstrations"] == 2.0
                assert concept_stats["min_demonstrations"] == 2
                assert concept_stats["max_demonstrations"] == 2
                assert concept_stats["avg_test_inputs"] == 2.0

    def test_load_task_file_success(self, base_config, sample_concept_task):
        """Test successful task file loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_concept_task, f)
            temp_file = f.name

        try:
            with patch("jaxarc.parsers.concept_arc.here") as mock_here:
                mock_here.return_value = Path("/mock/path")

                with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
                    parser = ConceptArcParser(base_config)

            # Test loading valid file
            data = parser.load_task_file(temp_file)
            assert data == sample_concept_task

        finally:
            Path(temp_file).unlink()

    def test_load_task_file_errors(self, base_config):
        """Test load_task_file error handling."""
        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
                parser = ConceptArcParser(base_config)

        # Test non-existent file
        with pytest.raises(FileNotFoundError, match="Task file not found"):
            parser.load_task_file("/nonexistent/file.json")

    def test_preprocess_task_data_success(self, base_config, sample_concept_task):
        """Test successful task data preprocessing."""
        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
                parser = ConceptArcParser(base_config)

        key = jax.random.PRNGKey(42)
        task_id = "TestConcept/test_task"

        # Test with tuple input (task_id, task_content)
        task = parser.preprocess_task_data((task_id, sample_concept_task), key)

        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs == 2
        assert task.num_test_pairs == 2

        # Check array shapes
        assert task.input_grids_examples.shape == (4, 30, 30)  # max_train_pairs
        assert task.test_input_grids.shape == (3, 30, 30)  # max_test_pairs

        # Test with direct task content
        task2 = parser.preprocess_task_data(sample_concept_task, key)
        assert isinstance(task2, JaxArcTask)

    def test_preprocess_task_data_error_handling(self, base_config):
        """Test preprocessing error handling with ConceptARC-specific messages."""
        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
                parser = ConceptArcParser(base_config)

        key = jax.random.PRNGKey(42)

        # Test empty training pairs
        invalid_data = {"train": [], "test": [{"input": [[1]]}]}
        with pytest.raises(
            ValueError, match="ConceptARC task must have at least one training pair"
        ):
            parser.preprocess_task_data(invalid_data, key)

        # Test empty test pairs
        invalid_data = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        with pytest.raises(
            ValueError, match="ConceptARC task must have at least one test pair"
        ):
            parser.preprocess_task_data(invalid_data, key)

    def test_empty_concept_directory_handling(self, base_config):
        """Test handling of concept directories with no JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"

            # Create concept directories but no JSON files
            for concept in ["EmptyConcept1", "EmptyConcept2"]:
                concept_dir = corpus_dir / concept
                concept_dir.mkdir(parents=True)
                # Create non-JSON files
                (concept_dir / "readme.txt").write_text("No tasks here")

            base_config.corpus.path = str(corpus_dir)

            with patch("jaxarc.parsers.concept_arc.here") as mock_here:
                mock_here.return_value = corpus_dir

                # Should raise ValueError when no concept groups found
                with pytest.raises(ValueError, match="No concept groups found"):
                    ConceptArcParser(base_config)

    def test_malformed_json_handling(self, base_config):
        """Test handling of malformed JSON files in concept directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"
            concept_dir = corpus_dir / "TestConcept"
            concept_dir.mkdir(parents=True)

            # Create malformed JSON file
            malformed_file = concept_dir / "malformed.json"
            with malformed_file.open("w") as f:
                f.write("{ invalid json content")

            # Create valid JSON file
            valid_file = concept_dir / "valid.json"
            with valid_file.open("w") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            base_config.corpus.path = str(corpus_dir)

            with patch("jaxarc.parsers.concept_arc.here") as mock_here:
                mock_here.return_value = corpus_dir

                # Should skip malformed file and load valid one
                parser = ConceptArcParser(base_config)

                concept_groups = parser.get_concept_groups()
                assert "TestConcept" in concept_groups

                tasks = parser.get_tasks_in_concept("TestConcept")
                # Should only have valid task (malformed should be skipped during loading)
                valid_tasks = [task for task in tasks if "valid" in task]
                assert len(valid_tasks) == 1
                assert "TestConcept/valid" in valid_tasks

    def test_concept_based_sampling_distribution(
        self, base_config, temp_concept_arc_directory
    ):
        """Test that concept-based sampling works correctly across multiple calls."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)

            # Test sampling from specific concept multiple times
            concept = "AboveBelow"
            keys = jax.random.split(jax.random.PRNGKey(42), 10)

            tasks = []
            for key in keys:
                task = parser.get_random_task_from_concept(concept, key)
                tasks.append(task)

            # All tasks should be valid
            assert all(isinstance(task, JaxArcTask) for task in tasks)

            # Should potentially get different tasks (though not guaranteed with small dataset)
            task_indices = [task.task_index for task in tasks]
            assert all(isinstance(idx, jnp.ndarray) for idx in task_indices)

    def test_jax_compatibility(self, base_config, temp_concept_arc_directory):
        """Test JAX compatibility of ConceptARC parsed data."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            parser = ConceptArcParser(base_config)
            key = jax.random.PRNGKey(42)
            task = parser.get_random_task(key)

        # Test JIT compilation
        @jax.jit
        def process_task(input_grids, input_masks):
            return jnp.sum(input_grids * input_masks)

        result = process_task(task.input_grids_examples, task.input_masks_examples)
        assert isinstance(result, jnp.ndarray)

        # Test vmap over concept-based tasks
        def process_single_input(input_grid, input_mask):
            return jnp.mean(input_grid * input_mask)

        vmapped_process = jax.vmap(process_single_input)
        results = vmapped_process(task.input_grids_examples, task.input_masks_examples)

        assert isinstance(results, jnp.ndarray)
        assert results.shape == (4,)  # max_train_pairs

    def test_hierarchical_directory_structure_validation(self, base_config):
        """Test validation of hierarchical directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"

            # Create nested directory structure (should be ignored)
            nested_concept = corpus_dir / "ValidConcept" / "SubConcept"
            nested_concept.mkdir(parents=True)

            # Create task in nested directory (should be ignored)
            nested_task = nested_concept / "nested_task.json"
            with nested_task.open("w") as f:
                json.dump(
                    {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    },
                    f,
                )

            # Create task in valid location
            valid_concept = corpus_dir / "ValidConcept"
            valid_task = valid_concept / "valid_task.json"
            with valid_task.open("w") as f:
                json.dump(
                    {
                        "train": [{"input": [[4]], "output": [[5]]}],
                        "test": [{"input": [[6]]}],
                    },
                    f,
                )

            base_config.corpus.path = str(corpus_dir)

            with patch("jaxarc.parsers.concept_arc.here") as mock_here:
                mock_here.return_value = corpus_dir

                parser = ConceptArcParser(base_config)

                # Should only find tasks in direct concept directories
                concept_groups = parser.get_concept_groups()
                assert "ValidConcept" in concept_groups

                tasks = parser.get_tasks_in_concept("ValidConcept")
                assert len(tasks) == 1
                assert "ValidConcept/valid_task" in tasks

    def test_deterministic_task_ordering(self, base_config, temp_concept_arc_directory):
        """Test that task ordering is deterministic across parser instances."""
        base_config.corpus.path = str(Path(temp_concept_arc_directory) / "corpus")

        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path(base_config.corpus.path)

            # Create two parser instances
            parser1 = ConceptArcParser(base_config)
            parser2 = ConceptArcParser(base_config)

            # Task IDs should be in the same order
            tasks1 = parser1.get_available_task_ids()
            tasks2 = parser2.get_available_task_ids()

            assert tasks1 == tasks2

            # Concept groups should be in the same order
            concepts1 = parser1.get_concept_groups()
            concepts2 = parser2.get_concept_groups()

            assert concepts1 == concepts2

    def test_edge_case_concept_names(self, base_config):
        """Test handling of edge case concept directory names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir) / "corpus"

            # Create concept directories with edge case names
            edge_case_concepts = [
                "Concept-With-Dashes",
                "Concept_With_Underscores",
                "123NumericConcept",
                "ConceptWithVeryLongNameThatExceedsTypicalLimits",
            ]

            for concept in edge_case_concepts:
                concept_dir = corpus_dir / concept
                concept_dir.mkdir(parents=True)

                # Create a task file
                task_file = concept_dir / "task.json"
                with task_file.open("w") as f:
                    json.dump(
                        {
                            "train": [{"input": [[1]], "output": [[2]]}],
                            "test": [{"input": [[3]]}],
                        },
                        f,
                    )

            base_config.corpus.path = str(corpus_dir)

            with patch("jaxarc.parsers.concept_arc.here") as mock_here:
                mock_here.return_value = corpus_dir

                parser = ConceptArcParser(base_config)

                # Should handle all edge case concept names
                concept_groups = parser.get_concept_groups()
                assert len(concept_groups) == len(edge_case_concepts)

                for concept in edge_case_concepts:
                    assert concept in concept_groups
                    tasks = parser.get_tasks_in_concept(concept)
                    assert len(tasks) == 1
                    assert f"{concept}/task" in tasks
