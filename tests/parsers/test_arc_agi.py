from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp  # Import jnp at the top
import pytest

from jaxarc.parsers import ArcAgiParser


@pytest.fixture
def sample_challenge_task_data():
    """Sample ARC challenge task data (test inputs, no test outputs)."""
    return {
        "test_task_id": {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                {"input": [[5, 6]], "output": [[6, 5]]},
            ],
            "test": [
                {"input": [[7, 8]]}  # No "output" here for challenge file
            ],
        },
        "another_task_id": {
            "train": [
                {"input": [[1]], "output": [[0]]},
            ],
            "test": [{"input": [[9]]}],
        },
    }


@pytest.fixture
def sample_solutions_data():
    """Sample ARC solutions data (only test outputs)."""
    return {
        "test_task_id": [  # List of outputs for the test inputs of "test_task_id"
            [[8, 7]]
        ],
        "another_task_id": [
            [[99]]  # Solution for another_task_id
        ],
    }


@pytest.fixture
def sample_challenge_file(sample_challenge_task_data):
    """Create a temporary JSON file with sample challenge task data."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(sample_challenge_task_data, f)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def sample_solutions_file(sample_solutions_data):
    """Create a temporary JSON file with sample solutions data."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(sample_solutions_data, f)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


def test_parse_valid_task(sample_challenge_file):  # Renamed fixture
    """Test parsing a valid task from a challenge file."""
    parser = ArcAgiParser()
    task = parser.parse_task_file(sample_challenge_file, "test_task_id")

    assert task.task_id == "test_task_id"
    assert len(task.train_pairs) == 2
    assert len(task.test_pairs) == 1

    # Check first training pair
    train_pair = task.train_pairs[0]
    assert train_pair.input.array.shape == (2, 2)
    assert train_pair.output.array.shape == (2, 2)

    # Check that test pair output is None as solutions are not loaded by parse_task_file
    assert task.test_pairs[0].input.array.shape == (1, 2)
    assert task.test_pairs[0].output is None

    # Clean up is handled by fixture's yield


def test_parse_all_tasks_from_challenges_only(sample_challenge_file):
    """Test parsing all tasks from a challenge file only (no solutions)."""
    parser = ArcAgiParser()
    tasks = parser.parse_all_tasks_from_file(sample_challenge_file)

    assert len(tasks) == 2
    assert "test_task_id" in tasks
    assert tasks["test_task_id"].task_id == "test_task_id"
    assert len(tasks["test_task_id"].test_pairs) == 1
    assert tasks["test_task_id"].test_pairs[0].output is None  # Output should be None

    assert "another_task_id" in tasks
    assert tasks["another_task_id"].test_pairs[0].output is None

    # Clean up is handled by fixture's yield


def test_parse_all_tasks_with_solutions(sample_challenge_file, sample_solutions_file):
    """Test parsing all tasks from a challenge file and a solutions file."""
    parser = ArcAgiParser()
    tasks = parser.parse_all_tasks_from_file(
        sample_challenge_file, sample_solutions_file
    )

    assert len(tasks) == 2
    assert "test_task_id" in tasks
    task1 = tasks["test_task_id"]
    assert task1.task_id == "test_task_id"
    assert len(task1.test_pairs) == 1
    assert task1.test_pairs[0].output is not None
    assert task1.test_pairs[0].output.array.shape == (1, 2)
    assert jnp.array_equal(task1.test_pairs[0].output.array, jnp.array([[8, 7]]))

    assert task1.test_pairs[0].input is not None  # Ensure input is still there
    assert task1.test_pairs[0].input.array.shape == (1, 2)

    assert "another_task_id" in tasks
    task2 = tasks["another_task_id"]
    assert task2.test_pairs[0].output is not None
    assert task2.test_pairs[0].output.array.shape == (1, 1)
    assert jnp.array_equal(task2.test_pairs[0].output.array, jnp.array([[99]]))

    # Clean up is handled by fixtures' yield


def test_parse_empty_file():
    """Test parsing an empty JSON file."""
    parser = ArcAgiParser()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump({}, f)
        temp_path = Path(f.name)

    tasks = parser.parse_all_tasks_from_file(temp_path)
    assert len(tasks) == 0

    # Clean up
    temp_path.unlink()


def test_parse_missing_file():
    """Test parsing a non-existent file."""
    parser = ArcAgiParser()
    with pytest.raises(FileNotFoundError):
        parser.parse_task_file("nonexistent_file.json", "any_task_id")


def test_parse_missing_task_id(sample_challenge_file):  # Use new fixture
    """Test parsing a task ID that doesn't exist in the file."""
    parser = ArcAgiParser()
    with pytest.raises(KeyError):
        parser.parse_task_file(sample_challenge_file, "nonexistent_task_id")

    # Clean up is handled by fixture's yield


def test_parse_invalid_json():
    """Test parsing a file with invalid JSON."""
    parser = ArcAgiParser()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        f.write("invalid json content")
        temp_path = Path(f.name)

    with pytest.raises(ValueError, match="Invalid JSON"):
        parser.parse_task_file(temp_path, "any_task_id")

    # Clean up
    temp_path.unlink()


def test_parse_task_with_multiple_train_pairs():
    """Test parsing a task with multiple training pairs."""
    task_data = {
        "multi_train_task": {
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[3]], "output": [[4]]},
                {"input": [[5]], "output": [[6]]},
            ],
            "test": [{"input": [[7]]}],  # No "output" in test
        }
    }

    parser = ArcAgiParser()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(task_data, f)
        temp_path = Path(f.name)

    task = parser.parse_task_file(temp_path, "multi_train_task")
    assert len(task.train_pairs) == 3
    assert len(task.test_pairs) == 1
    assert task.test_pairs[0].output is None  # Output should be None

    # Clean up
    temp_path.unlink()


def test_parse_irregular_grid_sizes():
    """Test parsing tasks with irregular grid sizes."""
    task_data = {
        "irregular_task": {
            "train": [
                {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},  # 1x3
                {"input": [[4], [5]], "output": [[5], [4]]},  # 2x1
            ],
            "test": [{"input": [[6, 7], [8, 9]]}],  # No "output" in test, 2x2
        }
    }

    parser = ArcAgiParser()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(task_data, f)
        temp_path = Path(f.name)

    task = parser.parse_task_file(temp_path, "irregular_task")

    # Check shapes
    assert task.train_pairs[0].input.array.shape == (1, 3)
    assert task.train_pairs[1].input.array.shape == (2, 1)
    assert task.test_pairs[0].input.array.shape == (2, 2)
    assert task.test_pairs[0].output is None  # Output should be None

    # Clean up
    temp_path.unlink()


def test_grid_dtype_consistency_challenge_only():
    """Test that grid arrays have consistent dtypes when parsing challenge file only."""
    task_data = {
        "dtype_task": {
            "train": [{"input": [[0, 9]], "output": [[9, 0]]}],
            "test": [{"input": [[1, 8]]}],  # No "output" in test
        }
    }

    parser = ArcAgiParser()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(task_data, f)
        temp_path = Path(f.name)

    task = parser.parse_task_file(temp_path, "dtype_task")

    # Check dtypes are int32
    assert task.train_pairs[0].input.array.dtype == jnp.int32
    assert task.train_pairs[0].output.array.dtype == jnp.int32
    assert task.test_pairs[0].input.array.dtype == jnp.int32
    assert task.test_pairs[0].output is None  # Output is None

    # Clean up
    temp_path.unlink()


def test_grid_dtype_consistency_with_solutions(
    sample_challenge_file, sample_solutions_file
):
    """Test that grid arrays have consistent dtypes when solutions are provided."""
    parser = ArcAgiParser()
    tasks = parser.parse_all_tasks_from_file(
        sample_challenge_file, sample_solutions_file
    )

    task = tasks["test_task_id"]  # Use one of the tasks from the fixture
    assert task.train_pairs[0].input.array.dtype == jnp.int32
    assert task.train_pairs[0].output.array.dtype == jnp.int32
    assert task.test_pairs[0].input.array.dtype == jnp.int32
    assert task.test_pairs[0].output.array.dtype == jnp.int32

    task_another = tasks["another_task_id"]
    assert task_another.train_pairs[0].input.array.dtype == jnp.int32
    assert task_another.train_pairs[0].output.array.dtype == jnp.int32
    assert task_another.test_pairs[0].input.array.dtype == jnp.int32
    assert task_another.test_pairs[0].output.array.dtype == jnp.int32
    # Clean up is handled by fixtures


# Add a test for when solutions file is present but a task_id is missing
def test_parse_all_tasks_missing_solution_for_a_task(sample_challenge_file):
    """Test parsing when a task_id in challenges is missing in solutions."""
    parser = ArcAgiParser()
    # Create a solutions file that is missing 'another_task_id'
    partial_solutions_data = {"test_task_id": [[[8, 7]]]}
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(partial_solutions_data, f)
        partial_solutions_file = Path(f.name)

    tasks = parser.parse_all_tasks_from_file(
        sample_challenge_file, partial_solutions_file
    )

    assert tasks["test_task_id"].test_pairs[0].output is not None
    assert jnp.array_equal(
        tasks["test_task_id"].test_pairs[0].output.array, jnp.array([[8, 7]])
    )

    # 'another_task_id' was in challenges but not solutions, so its test output should be None
    assert tasks["another_task_id"].test_pairs[0].output is None

    partial_solutions_file.unlink(missing_ok=True)


# Add a test for when solutions file has more/less solutions than test inputs
def test_parse_all_tasks_mismatched_solutions_count(sample_challenge_file):
    """Test parsing when solutions count mismatches test input count for a task."""
    parser = ArcAgiParser()
    # 'test_task_id' has 1 test input. Provide 0 solutions.
    mismatched_solutions_data_less = {"test_task_id": []}
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(mismatched_solutions_data_less, f)
        solutions_file_less = Path(f.name)

    tasks_less = parser.parse_all_tasks_from_file(
        sample_challenge_file, solutions_file_less
    )
    assert (
        tasks_less["test_task_id"].test_pairs[0].output is None
    )  # Should be None due to mismatch
    solutions_file_less.unlink(missing_ok=True)

    # Provide 2 solutions for 1 test input.
    mismatched_solutions_data_more = {"test_task_id": [[[8, 7]], [[1, 1]]]}
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(mismatched_solutions_data_more, f)
        solutions_file_more = Path(f.name)

    tasks_more = parser.parse_all_tasks_from_file(
        sample_challenge_file, solutions_file_more
    )
    # Output should be taken from the first solution, subsequent are ignored due to mismatch log
    assert tasks_more["test_task_id"].test_pairs[0].output is not None
    assert jnp.array_equal(
        tasks_more["test_task_id"].test_pairs[0].output.array, jnp.array([[8, 7]])
    )
    solutions_file_more.unlink(missing_ok=True)


def test_parse_malformed_solution_grid(sample_challenge_file):
    """Test parsing when a solution grid is malformed."""
    parser = ArcAgiParser()
    malformed_solutions_data = {
        "test_task_id": [  # Correct task_id
            "this_is_not_a_grid"  # Malformed grid
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(malformed_solutions_data, f)
        malformed_solutions_file = Path(f.name)

    tasks = parser.parse_all_tasks_from_file(
        sample_challenge_file, malformed_solutions_file
    )
    # The output should be None because the grid parsing failed
    assert tasks["test_task_id"].test_pairs[0].output is None
    malformed_solutions_file.unlink(missing_ok=True)


# Ensure old sample_task_data and sample_task_file are removed if they existed
# (Handled by replacing the fixtures at the top)
