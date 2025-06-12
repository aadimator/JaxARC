from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from jaxarc.parsers import ArcAgiParser


@pytest.fixture
def sample_task_data():
    """Sample ARC task data for testing."""
    return {
        "test_task_id": {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[2, 1], [4, 3]],
                },
                {
                    "input": [[5, 6]],
                    "output": [[6, 5]],
                },
            ],
            "test": [
                {
                    "input": [[7, 8]],
                    "output": [[8, 7]],
                }
            ],
        }
    }


@pytest.fixture
def sample_task_file(sample_task_data):
    """Create a temporary JSON file with sample task data."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(sample_task_data, f)
        return Path(f.name)


def test_parse_valid_task(sample_task_file):
    """Test parsing a valid task from a file."""
    parser = ArcAgiParser()
    task = parser.parse_task_file(sample_task_file, "test_task_id")

    assert task.task_id == "test_task_id"
    assert len(task.train_pairs) == 2
    assert len(task.test_pairs) == 1

    # Check first training pair
    train_pair = task.train_pairs[0]
    assert train_pair.input.array.shape == (2, 2)
    assert train_pair.output.array.shape == (2, 2)

    # Clean up
    sample_task_file.unlink()


def test_parse_all_tasks(sample_task_file):
    """Test parsing all tasks from a file."""
    parser = ArcAgiParser()
    tasks = parser.parse_all_tasks_from_file(sample_task_file)

    assert len(tasks) == 1
    assert "test_task_id" in tasks
    assert tasks["test_task_id"].task_id == "test_task_id"

    # Clean up
    sample_task_file.unlink()


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


def test_parse_missing_task_id(sample_task_file):
    """Test parsing a task ID that doesn't exist in the file."""
    parser = ArcAgiParser()
    with pytest.raises(KeyError):
        parser.parse_task_file(sample_task_file, "nonexistent_task_id")

    # Clean up
    sample_task_file.unlink()


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
            "test": [{"input": [[7]], "output": [[8]]}],
        }
    }

    parser = ArcAgiParser()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(task_data, f)
        temp_path = Path(f.name)

    task = parser.parse_task_file(temp_path, "multi_train_task")
    assert len(task.train_pairs) == 3
    assert len(task.test_pairs) == 1

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
            "test": [{"input": [[6, 7], [8, 9]], "output": [[9, 8], [7, 6]]}],  # 2x2
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

    # Clean up
    temp_path.unlink()


def test_grid_dtype_consistency():
    """Test that grid arrays have consistent dtypes."""
    task_data = {
        "dtype_task": {
            "train": [{"input": [[0, 9]], "output": [[9, 0]]}],
            "test": [{"input": [[1, 8]], "output": [[8, 1]]}],
        }
    }

    parser = ArcAgiParser()
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(task_data, f)
        temp_path = Path(f.name)

    task = parser.parse_task_file(temp_path, "dtype_task")

    # Check dtypes are int32
    import jax.numpy as jnp

    assert task.train_pairs[0].input.array.dtype == jnp.int32
    assert task.train_pairs[0].output.array.dtype == jnp.int32
    assert task.test_pairs[0].input.array.dtype == jnp.int32
    assert task.test_pairs[0].output.array.dtype == jnp.int32

    # Clean up
    temp_path.unlink()


def test_parse_task_json_directly():
    """Test parsing task JSON directly without file I/O."""
    task_json = {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
    }

    parser = ArcAgiParser()
    task = parser.parse_task_json(task_json, "direct_task")

    assert task.task_id == "direct_task"
    assert len(task.train_pairs) == 1
    assert len(task.test_pairs) == 1


def test_invalid_task_structure():
    """Test parsing with invalid task structure."""
    parser = ArcAgiParser()

    # Missing 'train' key
    with pytest.raises(ValueError, match="Missing 'train' or 'test' key"):
        parser.parse_task_json({"test": []}, "invalid_task")

    # Missing 'test' key
    with pytest.raises(ValueError, match="Missing 'train' or 'test' key"):
        parser.parse_task_json({"train": []}, "invalid_task")

    # Invalid pair structure
    with pytest.raises(ValueError, match="Missing 'input' or 'output' key"):
        parser.parse_task_json(
            {"train": [{"input": [[1]]}], "test": []}, "invalid_task"
        )
