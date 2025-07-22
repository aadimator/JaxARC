"""Tests for task manager system."""

from __future__ import annotations

import json
import pickle
import threading
import time
from pathlib import Path

import jax.numpy as jnp
import pytest

from jaxarc.utils.task_manager import (
    TaskIDManager,
    TemporaryTaskManager,
    create_jax_task_index,
    extract_task_id_from_index,
    get_global_task_manager,
    get_jax_task_index,
    get_task_id_globally,
    get_task_index_globally,
    is_dummy_task_index,
    register_task_globally,
    set_global_task_manager,
)


class TestTaskIDManager:
    """Test TaskIDManager class."""

    def test_init(self):
        """Test TaskIDManager initialization."""
        manager = TaskIDManager()

        assert manager.num_tasks() == 0
        assert manager.get_all_task_ids() == set()
        assert manager.get_all_indices() == set()

    def test_register_task_new(self):
        """Test registering a new task."""
        manager = TaskIDManager()

        index = manager.register_task("task_001")

        assert index == 0
        assert manager.num_tasks() == 1
        assert manager.has_task("task_001")
        assert manager.has_index(0)
        assert manager.get_index("task_001") == 0
        assert manager.get_task_id(0) == "task_001"

    def test_register_task_existing(self):
        """Test registering an existing task returns same index."""
        manager = TaskIDManager()

        index1 = manager.register_task("task_001")
        index2 = manager.register_task("task_001")  # Same task

        assert index1 == index2 == 0
        assert manager.num_tasks() == 1

    def test_register_multiple_tasks(self):
        """Test registering multiple different tasks."""
        manager = TaskIDManager()

        index1 = manager.register_task("task_001")
        index2 = manager.register_task("task_002")
        index3 = manager.register_task("task_003")

        assert index1 == 0
        assert index2 == 1
        assert index3 == 2
        assert manager.num_tasks() == 3

        # Test bidirectional mapping
        assert manager.get_task_id(0) == "task_001"
        assert manager.get_task_id(1) == "task_002"
        assert manager.get_task_id(2) == "task_003"

        assert manager.get_index("task_001") == 0
        assert manager.get_index("task_002") == 1
        assert manager.get_index("task_003") == 2

    def test_get_index_nonexistent(self):
        """Test getting index for non-existent task."""
        manager = TaskIDManager()

        assert manager.get_index("nonexistent") is None

    def test_get_task_id_nonexistent(self):
        """Test getting task ID for non-existent index."""
        manager = TaskIDManager()

        assert manager.get_task_id(999) is None

    def test_get_jax_index_success(self):
        """Test getting JAX-compatible index."""
        manager = TaskIDManager()
        manager.register_task("task_001")

        jax_index = manager.get_jax_index("task_001")

        assert jax_index.shape == ()
        assert jax_index.dtype == jnp.int32
        assert int(jax_index) == 0

    def test_get_jax_index_nonexistent(self):
        """Test getting JAX index for non-existent task raises error."""
        manager = TaskIDManager()

        with pytest.raises(ValueError, match="Task ID 'nonexistent' not registered"):
            manager.get_jax_index("nonexistent")

    def test_has_task(self):
        """Test has_task method."""
        manager = TaskIDManager()

        assert not manager.has_task("task_001")

        manager.register_task("task_001")

        assert manager.has_task("task_001")
        assert not manager.has_task("task_002")

    def test_has_index(self):
        """Test has_index method."""
        manager = TaskIDManager()

        assert not manager.has_index(0)

        manager.register_task("task_001")

        assert manager.has_index(0)
        assert not manager.has_index(1)

    def test_get_all_task_ids(self):
        """Test getting all task IDs."""
        manager = TaskIDManager()

        manager.register_task("task_001")
        manager.register_task("task_002")
        manager.register_task("task_003")

        all_ids = manager.get_all_task_ids()

        assert all_ids == {"task_001", "task_002", "task_003"}

    def test_get_all_indices(self):
        """Test getting all indices."""
        manager = TaskIDManager()

        manager.register_task("task_001")
        manager.register_task("task_002")
        manager.register_task("task_003")

        all_indices = manager.get_all_indices()

        assert all_indices == {0, 1, 2}

    def test_clear(self):
        """Test clearing all registrations."""
        manager = TaskIDManager()

        manager.register_task("task_001")
        manager.register_task("task_002")

        assert manager.num_tasks() == 2

        manager.clear()

        assert manager.num_tasks() == 0
        assert manager.get_all_task_ids() == set()
        assert manager.get_all_indices() == set()
        assert not manager.has_task("task_001")
        assert not manager.has_index(0)

    def test_clear_resets_next_index(self):
        """Test that clear resets the next index counter."""
        manager = TaskIDManager()

        manager.register_task("task_001")
        manager.register_task("task_002")

        manager.clear()

        # Next registration should start from 0 again
        index = manager.register_task("new_task")
        assert index == 0

    def test_repr(self):
        """Test string representation."""
        manager = TaskIDManager()

        repr_str = repr(manager)
        assert "TaskIDManager" in repr_str
        assert "num_tasks=0" in repr_str
        assert "next_index=0" in repr_str

        manager.register_task("task_001")

        repr_str = repr(manager)
        assert "num_tasks=1" in repr_str
        assert "next_index=1" in repr_str


class TestTaskIDManagerPersistence:
    """Test TaskIDManager persistence functionality."""

    def test_save_to_json_file(self, tmp_path):
        """Test saving to JSON file."""
        manager = TaskIDManager()
        manager.register_task("task_001")
        manager.register_task("task_002")

        json_file = tmp_path / "tasks.json"
        manager.save_to_file(json_file)

        assert json_file.exists()

        # Verify file contents
        with open(json_file) as f:
            data = json.load(f)

        assert data["id_to_index"] == {"task_001": 0, "task_002": 1}
        assert data["index_to_id"] == {"0": "task_001", "1": "task_002"}
        assert data["next_index"] == 2

    def test_save_to_pickle_file(self, tmp_path):
        """Test saving to pickle file."""
        manager = TaskIDManager()
        manager.register_task("task_001")
        manager.register_task("task_002")

        pickle_file = tmp_path / "tasks.pkl"
        manager.save_to_file(pickle_file)

        assert pickle_file.exists()

        # Verify file contents
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)

        assert data["id_to_index"] == {"task_001": 0, "task_002": 1}
        # Note: pickle saves integer keys as strings for JSON compatibility
        assert data["index_to_id"] == {"0": "task_001", "1": "task_002"}
        assert data["next_index"] == 2

    def test_load_from_json_file(self, tmp_path):
        """Test loading from JSON file."""
        # Create JSON file manually
        json_file = tmp_path / "tasks.json"
        data = {
            "id_to_index": {"task_001": 0, "task_002": 1},
            "index_to_id": {"0": "task_001", "1": "task_002"},
            "next_index": 2,
        }
        with open(json_file, "w") as f:
            json.dump(data, f)

        manager = TaskIDManager()
        manager.load_from_file(json_file)

        assert manager.num_tasks() == 2
        assert manager.get_index("task_001") == 0
        assert manager.get_index("task_002") == 1
        assert manager.get_task_id(0) == "task_001"
        assert manager.get_task_id(1) == "task_002"

        # Next registration should use index 2
        index = manager.register_task("task_003")
        assert index == 2

    def test_load_from_pickle_file(self, tmp_path):
        """Test loading from pickle file."""
        # Create pickle file manually
        pickle_file = tmp_path / "tasks.pkl"
        data = {
            "id_to_index": {"task_001": 0, "task_002": 1},
            "index_to_id": {0: "task_001", 1: "task_002"},
            "next_index": 2,
        }
        with open(pickle_file, "wb") as f:
            pickle.dump(data, f)

        manager = TaskIDManager()
        manager.load_from_file(pickle_file)

        assert manager.num_tasks() == 2
        assert manager.get_index("task_001") == 0
        assert manager.get_index("task_002") == 1
        assert manager.get_task_id(0) == "task_001"
        assert manager.get_task_id(1) == "task_002"

    def test_load_from_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file raises error."""
        manager = TaskIDManager()
        nonexistent_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            manager.load_from_file(nonexistent_file)

    def test_save_load_roundtrip_json(self, tmp_path):
        """Test save/load roundtrip with JSON."""
        original_manager = TaskIDManager()
        original_manager.register_task("task_001")
        original_manager.register_task("task_002")
        original_manager.register_task("task_003")

        json_file = tmp_path / "tasks.json"
        original_manager.save_to_file(json_file)

        loaded_manager = TaskIDManager()
        loaded_manager.load_from_file(json_file)

        # Should have same state
        assert loaded_manager.num_tasks() == original_manager.num_tasks()
        assert loaded_manager.get_all_task_ids() == original_manager.get_all_task_ids()
        assert loaded_manager.get_all_indices() == original_manager.get_all_indices()

        # Should continue indexing from same point
        new_index_original = original_manager.register_task("task_004")
        new_index_loaded = loaded_manager.register_task("task_004")
        assert new_index_original == new_index_loaded

    def test_save_load_roundtrip_pickle(self, tmp_path):
        """Test save/load roundtrip with pickle."""
        original_manager = TaskIDManager()
        original_manager.register_task("task_001")
        original_manager.register_task("task_002")

        pickle_file = tmp_path / "tasks.pkl"
        original_manager.save_to_file(pickle_file)

        loaded_manager = TaskIDManager()
        loaded_manager.load_from_file(pickle_file)

        # Should have same state
        assert loaded_manager.num_tasks() == original_manager.num_tasks()
        assert loaded_manager.get_all_task_ids() == original_manager.get_all_task_ids()
        assert loaded_manager.get_all_indices() == original_manager.get_all_indices()

    def test_save_error_handling(self, tmp_path):
        """Test error handling during save operations."""
        manager = TaskIDManager()
        manager.register_task("task_001")

        # Try to save to a directory that doesn't exist and can't be created
        invalid_path = tmp_path / "nonexistent" / "deeply" / "nested" / "tasks.json"

        with pytest.raises(Exception):  # Could be FileNotFoundError or PermissionError
            manager.save_to_file(invalid_path)

    def test_load_error_handling(self, tmp_path):
        """Test error handling during load operations."""
        manager = TaskIDManager()

        # Create invalid JSON file
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("invalid json content")

        with pytest.raises(Exception):  # JSON decode error
            manager.load_from_file(invalid_json)


class TestTaskIDManagerThreadSafety:
    """Test TaskIDManager thread safety."""

    def test_concurrent_registration(self):
        """Test concurrent task registration from multiple threads."""
        manager = TaskIDManager()
        results = {}

        def register_tasks(thread_id, num_tasks):
            thread_results = []
            for i in range(num_tasks):
                task_id = f"thread_{thread_id}_task_{i}"
                index = manager.register_task(task_id)
                thread_results.append((task_id, index))
            results[thread_id] = thread_results

        # Start multiple threads
        threads = []
        num_threads = 5
        tasks_per_thread = 10

        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=register_tasks, args=(thread_id, tasks_per_thread)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        total_expected_tasks = num_threads * tasks_per_thread
        assert manager.num_tasks() == total_expected_tasks

        # Verify all indices are unique
        all_indices = set()
        for thread_results in results.values():
            for task_id, index in thread_results:
                assert index not in all_indices, (
                    f"Duplicate index {index} for task {task_id}"
                )
                all_indices.add(index)
                assert manager.get_index(task_id) == index
                assert manager.get_task_id(index) == task_id

    def test_concurrent_access_patterns(self):
        """Test various concurrent access patterns."""
        manager = TaskIDManager()

        # Pre-register some tasks
        for i in range(10):
            manager.register_task(f"initial_task_{i}")

        results = {"registrations": [], "lookups": [], "errors": []}

        def register_worker():
            try:
                for i in range(5):
                    task_id = (
                        f"register_worker_task_{threading.current_thread().ident}_{i}"
                    )
                    index = manager.register_task(task_id)
                    results["registrations"].append((task_id, index))
            except Exception as e:
                results["errors"].append(e)

        def lookup_worker():
            try:
                for i in range(20):
                    # Look up existing tasks
                    task_id = f"initial_task_{i % 10}"
                    index = manager.get_index(task_id)
                    task_id_back = manager.get_task_id(index)
                    results["lookups"].append((task_id, index, task_id_back))
            except Exception as e:
                results["errors"].append(e)

        # Start multiple workers
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=register_worker))
        for _ in range(3):
            threads.append(threading.Thread(target=lookup_worker))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"

        # Verify lookups were consistent
        for task_id, index, task_id_back in results["lookups"]:
            assert task_id == task_id_back
            assert manager.get_index(task_id) == index

    def test_concurrent_clear_operations(self):
        """Test concurrent clear operations."""
        manager = TaskIDManager()

        # Register initial tasks
        for i in range(10):
            manager.register_task(f"task_{i}")

        results = {"clear_count": 0, "register_count": 0, "errors": []}

        def clear_worker():
            try:
                time.sleep(0.01)  # Small delay to increase chance of race conditions
                manager.clear()
                results["clear_count"] += 1
            except Exception as e:
                results["errors"].append(e)

        def register_worker():
            try:
                for i in range(5):
                    task_id = f"concurrent_task_{threading.current_thread().ident}_{i}"
                    manager.register_task(task_id)
                    results["register_count"] += 1
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                results["errors"].append(e)

        # Start workers
        threads = []
        threads.append(threading.Thread(target=clear_worker))
        for _ in range(3):
            threads.append(threading.Thread(target=register_worker))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should not have errors (thread safety should prevent corruption)
        assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"

        # Final state should be consistent
        assert manager.num_tasks() >= 0  # Could be 0 if clear happened last


class TestGlobalTaskManager:
    """Test global task manager functionality."""

    def test_get_global_task_manager_singleton(self):
        """Test that global task manager is a singleton."""
        manager1 = get_global_task_manager()
        manager2 = get_global_task_manager()

        assert manager1 is manager2

    def test_set_global_task_manager(self):
        """Test setting custom global task manager."""
        original_manager = get_global_task_manager()
        custom_manager = TaskIDManager()
        custom_manager.register_task("custom_task")

        set_global_task_manager(custom_manager)

        current_manager = get_global_task_manager()
        assert current_manager is custom_manager
        assert current_manager.has_task("custom_task")

        # Restore original for other tests
        set_global_task_manager(original_manager)

    def test_register_task_globally(self):
        """Test global task registration."""
        # Clear global manager for clean test
        global_manager = get_global_task_manager()
        global_manager.clear()

        index = register_task_globally("global_task_001")

        assert index == 0
        assert get_task_index_globally("global_task_001") == 0
        assert get_task_id_globally(0) == "global_task_001"

    def test_get_task_index_globally_nonexistent(self):
        """Test getting non-existent task index globally."""
        result = get_task_index_globally("nonexistent_global_task")
        assert result is None

    def test_get_task_id_globally_nonexistent(self):
        """Test getting non-existent task ID globally."""
        result = get_task_id_globally(999)
        assert result is None

    def test_get_jax_task_index_global(self):
        """Test getting JAX task index globally."""
        global_manager = get_global_task_manager()
        global_manager.clear()

        register_task_globally("jax_task_001")

        jax_index = get_jax_task_index("jax_task_001")

        assert jax_index.shape == ()
        assert jax_index.dtype == jnp.int32
        assert int(jax_index) == 0

    def test_get_jax_task_index_nonexistent(self):
        """Test getting JAX index for non-existent global task."""
        with pytest.raises(
            ValueError, match="Task ID 'nonexistent_jax_task' not registered"
        ):
            get_jax_task_index("nonexistent_jax_task")


class TestUtilityFunctions:
    """Test utility functions for JAX-compatible task data."""

    def test_create_jax_task_index_with_task_id(self):
        """Test creating JAX task index with task ID."""
        global_manager = get_global_task_manager()
        global_manager.clear()

        jax_index = create_jax_task_index("utility_task_001")

        assert jax_index.shape == ()
        assert jax_index.dtype == jnp.int32
        assert int(jax_index) == 0

        # Should be registered globally
        assert get_task_index_globally("utility_task_001") == 0

    def test_create_jax_task_index_with_none(self):
        """Test creating JAX task index with None (dummy task)."""
        jax_index = create_jax_task_index(None)

        assert jax_index.shape == ()
        assert jax_index.dtype == jnp.int32
        assert int(jax_index) == -1

    def test_extract_task_id_from_index_valid(self):
        """Test extracting task ID from valid index."""
        global_manager = get_global_task_manager()
        global_manager.clear()

        register_task_globally("extract_task_001")
        jax_index = jnp.array(0, dtype=jnp.int32)

        task_id = extract_task_id_from_index(jax_index)

        assert task_id == "extract_task_001"

    def test_extract_task_id_from_index_dummy(self):
        """Test extracting task ID from dummy index."""
        dummy_index = jnp.array(-1, dtype=jnp.int32)

        task_id = extract_task_id_from_index(dummy_index)

        assert task_id is None

    def test_extract_task_id_from_index_nonexistent(self):
        """Test extracting task ID from non-existent index."""
        nonexistent_index = jnp.array(999, dtype=jnp.int32)

        task_id = extract_task_id_from_index(nonexistent_index)

        assert task_id is None

    def test_is_dummy_task_index_true(self):
        """Test is_dummy_task_index with dummy index."""
        dummy_index = jnp.array(-1, dtype=jnp.int32)

        assert is_dummy_task_index(dummy_index) is True

    def test_is_dummy_task_index_false(self):
        """Test is_dummy_task_index with valid index."""
        valid_index = jnp.array(0, dtype=jnp.int32)

        assert is_dummy_task_index(valid_index) is False

    def test_is_dummy_task_index_other_negative(self):
        """Test is_dummy_task_index with other negative values."""
        other_negative = jnp.array(-5, dtype=jnp.int32)

        assert is_dummy_task_index(other_negative) is False


class TestTemporaryTaskManager:
    """Test TemporaryTaskManager context manager."""

    def test_temporary_task_manager_with_new_manager(self):
        """Test temporary task manager with new manager."""
        global_manager = get_global_task_manager()
        global_manager.clear()
        global_manager.register_task("global_task")

        with TemporaryTaskManager() as temp_manager:
            # Inside context, should use temporary manager
            current_manager = get_global_task_manager()
            assert current_manager is temp_manager
            assert not current_manager.has_task("global_task")

            # Register task in temporary manager
            temp_manager.register_task("temp_task")
            assert register_task_globally("temp_task_2") == 1  # Should use temp manager

        # Outside context, should restore original manager
        restored_manager = get_global_task_manager()
        assert restored_manager is global_manager
        assert restored_manager.has_task("global_task")
        assert not restored_manager.has_task("temp_task")

    def test_temporary_task_manager_with_custom_manager(self):
        """Test temporary task manager with custom manager."""
        custom_manager = TaskIDManager()
        custom_manager.register_task("custom_task")

        with TemporaryTaskManager(custom_manager) as temp_manager:
            assert temp_manager is custom_manager
            current_manager = get_global_task_manager()
            assert current_manager is custom_manager
            assert current_manager.has_task("custom_task")

        # Should restore original after context
        restored_manager = get_global_task_manager()
        assert restored_manager is not custom_manager

    def test_temporary_task_manager_exception_handling(self):
        """Test that temporary task manager restores original even on exception."""
        original_manager = get_global_task_manager()

        try:
            with TemporaryTaskManager() as temp_manager:
                assert get_global_task_manager() is temp_manager
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still restore original manager
        assert get_global_task_manager() is original_manager

    def test_temporary_task_manager_nested(self):
        """Test nested temporary task managers."""
        original_manager = get_global_task_manager()
        original_manager.clear()
        original_manager.register_task("original_task")

        with TemporaryTaskManager() as temp1:
            temp1.register_task("temp1_task")
            assert get_global_task_manager() is temp1

            with TemporaryTaskManager() as temp2:
                temp2.register_task("temp2_task")
                assert get_global_task_manager() is temp2
                assert not temp2.has_task("temp1_task")
                assert not temp2.has_task("original_task")

            # Should restore temp1
            assert get_global_task_manager() is temp1
            assert temp1.has_task("temp1_task")

        # Should restore original
        assert get_global_task_manager() is original_manager
        assert original_manager.has_task("original_task")


class TestIntegrationScenarios:
    """Test integration scenarios and real-world usage patterns."""

    def test_task_lifecycle_management(self):
        """Test complete task lifecycle management."""
        # Use a temporary manager to avoid global state interference
        with TemporaryTaskManager() as manager:
            # Register tasks from different sources
            task_ids = [
                "dataset_task_001",
                "dataset_task_002",
                "generated_task_001",
                "user_task_001",
            ]

        indices = []
        for task_id in task_ids:
            index = manager.register_task(task_id)
            indices.append(index)

        # Verify all tasks are registered
        assert manager.num_tasks() == len(task_ids)

        # Test JAX compatibility
        jax_indices = []
        for task_id in task_ids:
            jax_index = manager.get_jax_index(task_id)
            jax_indices.append(jax_index)

            # Verify roundtrip conversion using the manager's method
            index_value = int(jax_index.item())
            extracted_id = manager.get_task_id(index_value)
            assert extracted_id == task_id

            # Test persistence
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                temp_path = Path(f.name)

            try:
                manager.save_to_file(temp_path)

                # Create new manager and load
                new_manager = TaskIDManager()
                new_manager.load_from_file(temp_path)

                # Verify state is preserved
                assert new_manager.num_tasks() == manager.num_tasks()
                for task_id, expected_index in zip(task_ids, indices):
                    assert new_manager.get_index(task_id) == expected_index
                    assert new_manager.get_task_id(expected_index) == task_id
            finally:
                temp_path.unlink()

    def test_batch_processing_scenario(self):
        """Test scenario with batch processing of tasks."""
        # Use a temporary manager to avoid global state interference
        with TemporaryTaskManager() as manager:
            # Simulate batch of tasks
            batch_task_ids = [f"batch_task_{i:03d}" for i in range(100)]

            # Register all tasks
            batch_indices = []
            for task_id in batch_task_ids:
                index = manager.register_task(task_id)
                batch_indices.append(index)

            # Create JAX arrays for batch processing
            jax_batch_indices = jnp.array(batch_indices, dtype=jnp.int32)

            # Simulate JAX operations on indices
            processed_indices = jax_batch_indices + 0  # Identity operation

            # Verify all indices can be converted back to task IDs
            for i, jax_index in enumerate(processed_indices):
                task_id = extract_task_id_from_index(jax_index)
                assert task_id == batch_task_ids[i]

    def test_concurrent_global_access(self):
        """Test concurrent access to global task manager."""
        global_manager = get_global_task_manager()
        global_manager.clear()

        results = {"success": 0, "errors": []}

        def worker(worker_id):
            try:
                for i in range(10):
                    task_id = f"worker_{worker_id}_task_{i}"
                    index = register_task_globally(task_id)

                    # Verify registration
                    retrieved_index = get_task_index_globally(task_id)
                    assert retrieved_index == index

                    retrieved_task_id = get_task_id_globally(index)
                    assert retrieved_task_id == task_id

                    results["success"] += 1
            except Exception as e:
                results["errors"].append(e)

        # Start multiple workers
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify results
        assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
        assert results["success"] == 50  # 5 workers * 10 tasks each
        assert global_manager.num_tasks() == 50

    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        manager = TaskIDManager()

        # Test recovery from invalid operations
        manager.register_task("valid_task")

        # These should not crash the manager
        assert manager.get_index("nonexistent") is None
        assert manager.get_task_id(999) is None

        # Manager should still work normally
        index = manager.register_task("another_task")
        assert index == 1
        assert manager.get_index("another_task") == 1

        # Test recovery from clear operation
        manager.clear()
        assert manager.num_tasks() == 0

        # Should be able to register new tasks
        index = manager.register_task("post_clear_task")
        assert index == 0

    def test_memory_efficiency_large_scale(self):
        """Test memory efficiency with large number of tasks."""
        manager = TaskIDManager()

        # Register many tasks
        num_tasks = 1000
        task_ids = [f"large_scale_task_{i:04d}" for i in range(num_tasks)]

        for task_id in task_ids:
            manager.register_task(task_id)

        assert manager.num_tasks() == num_tasks

        # Verify all mappings work correctly
        for i, task_id in enumerate(task_ids):
            assert manager.get_index(task_id) == i
            assert manager.get_task_id(i) == task_id

        # Test that clear frees memory (at least doesn't crash)
        manager.clear()
        assert manager.num_tasks() == 0
