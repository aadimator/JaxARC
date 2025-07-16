"""Comprehensive tests for ConceptARC parser implementation.

This test suite covers:
- Concept group discovery and organization
- Task loading from hierarchical structure
- Concept-based random sampling
- Error handling for missing concept groups
- Task metadata and statistics
- Grid validation and preprocessing
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chex
import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.parsers.concept_arc import ConceptArcParser
from jaxarc.types import JaxArcTask


@pytest.fixture
def sample_concept_arc_task():
    """Sample ConceptARC task data."""
    return {
        "train": [
            {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
            {"input": [[0, 2], [2, 0]], "output": [[2, 0], [0, 2]]},
        ],
        "test": [
            {"input": [[0, 3], [3, 0]], "output": [[3, 0], [0, 3]]},
            {
                "input": [[0, 4], [4, 0]]
                # Note: no output for second test case (typical for challenge format)
            },
        ],
    }


@pytest.fixture
def concept_arc_config():
    """Sample ConceptARC configuration."""
    return DictConfig(
        {
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
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
                ],
            },
        }
    )


@pytest.fixture
def mock_concept_arc_directory(sample_concept_arc_task):
    """Create a mock ConceptARC directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        corpus_dir = Path(temp_dir) / "corpus"

        # Create concept group directories with sample tasks
        concept_groups = ["AboveBelow", "Center", "Copy"]

        for concept in concept_groups:
            concept_dir = corpus_dir / concept
            concept_dir.mkdir(parents=True)

            # Create sample task files
            for i in range(2):  # 2 tasks per concept
                task_file = concept_dir / f"task_{i:03d}.json"
                with task_file.open("w") as f:
                    json.dump(sample_concept_arc_task, f)

        yield temp_dir


def test_concept_arc_parser_initialization(
    concept_arc_config, mock_concept_arc_directory
):
    """Test ConceptArcParser initialization."""
    # Update config to point to mock directory
    concept_arc_config.corpus.path = str(Path(mock_concept_arc_directory) / "corpus")

    with patch("jaxarc.parsers.concept_arc.here") as mock_here:
        mock_here.return_value = Path(concept_arc_config.corpus.path)

        parser = ConceptArcParser(concept_arc_config)

        # Check that concept groups were discovered
        concept_groups = parser.get_concept_groups()
        assert len(concept_groups) == 3
        assert "AboveBelow" in concept_groups
        assert "Center" in concept_groups
        assert "Copy" in concept_groups

        # Check that tasks were loaded
        all_tasks = parser.get_available_task_ids()
        assert len(all_tasks) == 6  # 3 concepts × 2 tasks each


def test_concept_arc_parser_concept_based_sampling(
    concept_arc_config, mock_concept_arc_directory
):
    """Test concept-based random sampling functionality."""
    # Update config to point to mock directory
    concept_arc_config.corpus.path = str(Path(mock_concept_arc_directory) / "corpus")

    with patch("jaxarc.parsers.concept_arc.here") as mock_here:
        mock_here.return_value = Path(concept_arc_config.corpus.path)

        parser = ConceptArcParser(concept_arc_config)
        key = jax.random.PRNGKey(42)

        # Test getting random task from specific concept
        task = parser.get_random_task_from_concept("AboveBelow", key)
        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs == 2
        assert task.num_test_pairs == 2

        # Test getting tasks in concept
        tasks_in_concept = parser.get_tasks_in_concept("Center")
        assert len(tasks_in_concept) == 2
        assert all(task_id.startswith("Center/") for task_id in tasks_in_concept)


def test_concept_arc_parser_invalid_concept(
    concept_arc_config, mock_concept_arc_directory
):
    """Test error handling for invalid concept groups."""
    # Update config to point to mock directory
    concept_arc_config.corpus.path = str(Path(mock_concept_arc_directory) / "corpus")

    with patch("jaxarc.parsers.concept_arc.here") as mock_here:
        mock_here.return_value = Path(concept_arc_config.corpus.path)

        parser = ConceptArcParser(concept_arc_config)
        key = jax.random.PRNGKey(42)

        # Test invalid concept group
        with pytest.raises(
            ValueError, match="Concept group 'InvalidConcept' not found"
        ):
            parser.get_random_task_from_concept("InvalidConcept", key)

        with pytest.raises(
            ValueError, match="Concept group 'InvalidConcept' not found"
        ):
            parser.get_tasks_in_concept("InvalidConcept")


def test_concept_arc_parser_task_preprocessing(
    concept_arc_config, sample_concept_arc_task
):
    """Test task data preprocessing."""
    with patch("jaxarc.parsers.concept_arc.here") as mock_here:
        mock_here.return_value = Path("/mock/path")

        # Create parser with empty directory (to avoid loading)
        with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
            parser = ConceptArcParser(concept_arc_config)

        key = jax.random.PRNGKey(0)
        task_id = "TestConcept/test_task"

        # Test preprocessing
        jax_task = parser.preprocess_task_data((task_id, sample_concept_arc_task), key)

        assert isinstance(jax_task, JaxArcTask)
        assert jax_task.num_train_pairs == 2
        assert jax_task.num_test_pairs == 2

        # Check array shapes
        assert jax_task.input_grids_examples.shape == (
            4,
            30,
            30,
        )  # max_train_pairs, max_h, max_w
        assert jax_task.test_input_grids.shape == (
            3,
            30,
            30,
        )  # max_test_pairs, max_h, max_w

        # Check data types
        assert jax_task.input_grids_examples.dtype == jnp.int32
        assert jax_task.input_masks_examples.dtype == jnp.bool_

        # Check that task index was created
        assert jax_task.task_index.dtype == jnp.int32


def test_concept_arc_parser_get_random_task(
    concept_arc_config, mock_concept_arc_directory
):
    """Test getting random task from entire dataset."""
    # Update config to point to mock directory
    concept_arc_config.corpus.path = str(Path(mock_concept_arc_directory) / "corpus")

    with patch("jaxarc.parsers.concept_arc.here") as mock_here:
        mock_here.return_value = Path(concept_arc_config.corpus.path)

        parser = ConceptArcParser(concept_arc_config)
        key = jax.random.PRNGKey(123)

        # Test getting random task
        task = parser.get_random_task(key)
        assert isinstance(task, JaxArcTask)
        assert task.num_train_pairs == 2
        assert task.num_test_pairs == 2


def test_concept_arc_parser_get_task_by_id(
    concept_arc_config, mock_concept_arc_directory
):
    """Test getting specific task by ID."""
    # Update config to point to mock directory
    concept_arc_config.corpus.path = str(Path(mock_concept_arc_directory) / "corpus")

    with patch("jaxarc.parsers.concept_arc.here") as mock_here:
        mock_here.return_value = Path(concept_arc_config.corpus.path)

        parser = ConceptArcParser(concept_arc_config)

        # Get available task IDs
        task_ids = parser.get_available_task_ids()
        assert len(task_ids) > 0

        # Test getting specific task
        task_id = task_ids[0]
        task = parser.get_task_by_id(task_id)
        assert isinstance(task, JaxArcTask)

        # Test invalid task ID
        with pytest.raises(ValueError, match="Task ID 'invalid_task' not found"):
            parser.get_task_by_id("invalid_task")


def test_concept_arc_parser_metadata_and_statistics(
    concept_arc_config, mock_concept_arc_directory
):
    """Test task metadata and dataset statistics."""
    # Update config to point to mock directory
    concept_arc_config.corpus.path = str(Path(mock_concept_arc_directory) / "corpus")

    with patch("jaxarc.parsers.concept_arc.here") as mock_here:
        mock_here.return_value = Path(concept_arc_config.corpus.path)

        parser = ConceptArcParser(concept_arc_config)

        # Test task metadata
        task_ids = parser.get_available_task_ids()
        task_id = task_ids[0]
        metadata = parser.get_task_metadata(task_id)

        assert "concept_group" in metadata
        assert "task_name" in metadata
        assert "file_path" in metadata
        assert "num_demonstrations" in metadata
        assert "num_test_inputs" in metadata

        # Test dataset statistics
        stats = parser.get_dataset_statistics()
        assert stats["total_tasks"] == 6
        assert stats["total_concept_groups"] == 3
        assert "concept_groups" in stats

        # Check concept group statistics
        for concept in ["AboveBelow", "Center", "Copy"]:
            assert concept in stats["concept_groups"]
            concept_stats = stats["concept_groups"][concept]
            assert concept_stats["num_tasks"] == 2
            assert "avg_demonstrations" in concept_stats
            assert "avg_test_inputs" in concept_stats


def test_concept_arc_parser_empty_directory(concept_arc_config):
    """Test handling of empty or missing corpus directory."""
    concept_arc_config.corpus.path = "/nonexistent/path"

    with patch("jaxarc.parsers.concept_arc.here") as mock_here:
        mock_here.return_value = Path("/nonexistent/path")

        # Should not raise exception during initialization, but log warning
        parser = ConceptArcParser(concept_arc_config)

        # Should have no tasks
        assert len(parser.get_available_task_ids()) == 0
        assert len(parser.get_concept_groups()) == 0

        # Should raise error when trying to get random task
        key = jax.random.PRNGKey(0)
        with pytest.raises(RuntimeError, match="No tasks available"):
            parser.get_random_task(key)


def test_concept_arc_parser_load_task_file(concept_arc_config, sample_concept_arc_task):
    """Test loading task file directly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_concept_arc_task, f)
        temp_file = f.name

    try:
        with patch("jaxarc.parsers.concept_arc.here") as mock_here:
            mock_here.return_value = Path("/mock/path")

            with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
                parser = ConceptArcParser(concept_arc_config)

            # Test loading valid file
            task_data = parser.load_task_file(temp_file)
            assert task_data == sample_concept_arc_task

            # Test loading nonexistent file
            with pytest.raises(FileNotFoundError):
                parser.load_task_file("/nonexistent/file.json")

    finally:
        Path(temp_file).unlink()


def test_concept_arc_parser_grid_validation(concept_arc_config):
    """Test grid validation functionality."""
    with patch("jaxarc.parsers.concept_arc.here") as mock_here:
        mock_here.return_value = Path("/mock/path")

        with patch.object(ConceptArcParser, "_load_and_cache_tasks"):
            parser = ConceptArcParser(concept_arc_config)

        # Test valid grid
        valid_grid = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
        parser._validate_grid_colors(valid_grid)  # Should not raise

        # Test invalid grid (color out of range)
        invalid_grid = jnp.array([[0, 1, 15]], dtype=jnp.int32)  # 15 > max_colors
        with pytest.raises(ValueError, match="Invalid color in grid"):
            parser._validate_grid_colors(invalid_grid)
