"""Tests for task manager utilities."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import chex
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
    """Test the TaskIDManager class."""

    def test_register_task_basic(self):
        """Test basic task registration."""
        manager = TaskIDManager()

        index1 = manager.register_task("task_001")
        index2 = manager.register_task("task_002")

        assert index1 == 0
        assert index2 == 1
        assert index1 != index2

    def test_register_task_duplicate(self):
        """Test registering the same task returns same index."""
        manager = TaskIDManager()

        index1 = manager.register_task("task_001")
        index2 = manager.register_task("task_001")  # Same task

        assert index1 == index2
        assert index1 == 0

    def test_get_index_existing(self):
        """Test getting index for existing task."""
        manager = TaskIDManager()

        registered_index = manager.register_task("task_001")
        retrieved_index = manager.get_index("task_001")

        assert retrieved_index == registered_index
        assert retrieved_index == 0

    def test_get_index_nonexistent(self):
        """Test getting index for nonexistent task returns None."""
        manager = TaskIDManager()

        index = manager.get_index("nonexistent_task")
        assert index is None

    def test_get_task_id_existing(self):
        """Test getting task ID for existing index."""
        manager = TaskIDManager()

        index = manager.register_task("task_001")
        task_id = manager.get_task_id(index)

        assert task_id == "task_001"

    def test_get_task_id_nonexistent(self):
        """Test getting task ID for nonexistent index returns None."""
        manager = TaskIDManager()

        task_id = manager.get_task_id(999)
        assert task_id is None

    def test_get_jax_index_existing(self):
        """Test getting JAX index for existing task."""
        manager = TaskIDManager()

        manager.register_task("task_001")
        jax_index = manager.get_jax_index("task_001")

        chex.assert_shape(jax_index, ())
        assert jax_index.dtype == jnp.int32
        assert int(jax_index) == 0

    def test_get_jax_index_nonexistent(self):
        """Test getting JAX index for nonexistent task raises error."""
        manager = TaskIDManager()

        with pytest.raises(ValueError, match="Task ID 'nonexistent' not registered"):
            manager.get_jax_index("nonexistent")

    def test_has_task(self):
        """Test checking if task exists."""
        manager = TaskIDManager()

        assert not manager.has_task("task_001")

        manager.register_task("task_001")
        assert manager.has_task("task_001")
        assert not manager.has_task("task_002")

    def test_has_index(self):
        """Test checking if index exists."""
        manager = TaskIDManager()

        assert not manager.has_index(0)

        manager.register_task("task_001")
        assert manager.has_index(0)
        assert not manager.has_index(1)

    def test_get_all_task_ids(self):
        """Test getting all registered task IDs."""
        manager = TaskIDManager()

        assert manager.get_all_task_ids() == set()

        manager.register_task("task_001")
        manager.register_task("task_002")

        all_ids = manager.get_all_task_ids()
        assert all_ids == {"task_001", "task_002"}

    def test_get_all_indices(self):
        """Test getting all assigned indices."""
        manager = TaskIDManager()

        assert manager.get_all_indices() == set()

        manager.register_task("task_001")
        manager.register_task("task_002")

        all_indices = manager.get_all_indices()
        assert all_indices == {0, 1}

    def test_num_tasks(self):
        """Test getting number of registered tasks."""
        manager = TaskIDManager()

        assert manager.num_tasks() == 0

        manager.register_task("task_001")
        assert manager.num_tasks() == 1

        manager.register_task("task_002")
        assert manager.num_tasks() == 2

        # Registering same task doesn't increase count
        manager.register_task("task_001")
        assert manager.num_tasks() == 2

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

        # Next registration should start from 0 again
        index = manager.register_task("new_task")
        assert index == 0

    def test_sequential_index_assignment(self):
        """Test that indices are assigned sequentially."""
        manager = TaskIDManager()

        indices = []
        for i in range(5):
            index = manager.register_task(f"task_{i:03d}")
            indices.append(index)

        assert indices == [0, 1, 2, 3, 4]

    def test_repr(self):
        """Test string representation."""
        manager = TaskIDManager()

        repr_empty = repr(manager)
        assert "num_tasks=0" in repr_empty
        assert "next_index=0" in repr_empty

        manager.register_task("task_001")
        manager.register_task("task_002")

        repr_filled = repr(manager)
        assert "num_tasks=2" in repr_filled
        assert "next_index=2" in repr_filled


class TestTaskIDManagerPersistence:
    """Test task manager persistence functionality."""

    def test_save_load_json(self):
        """Test saving and loading with JSON format."""
        manager = TaskIDManager()

        # Register some tasks
        manager.register_task("task_001")
        manager.register_task("task_002")
        manager.register_task("task_003")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Save to file
            manager.save_to_file(filepath)
            assert filepath.exists()

            # Create new manager and load
            new_manager = TaskIDManager()
            new_manager.load_from_file(filepath)

            # Verify loaded data
            assert new_manager.num_tasks() == 3
            assert new_manager.get_index("task_001") == 0
            assert new_manager.get_index("task_002") == 1
            assert new_manager.get_index("task_003") == 2
            assert new_manager.get_task_id(0) == "task_001"
            assert new_manager.get_task_id(1) == "task_002"
            assert new_manager.get_task_id(2) == "task_003"

            # Next registration should continue from 3
            next_index = new_manager.register_task("task_004")
            assert next_index == 3

        finally:
            if filepath.exists():
                filepath.unlink()

    def test_save_load_pickle(self):
        """Test saving and loading with pickle format."""
        manager = TaskIDManager()

        # Register some tasks
        manager.register_task("task_001")
        manager.register_task("task_002")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = Path(f.name)

        try:
            # Save to file
            manager.save_to_file(filepath)
            assert filepath.exists()

            # Create new manager and load
            new_manager = TaskIDManager()
            new_manager.load_from_file(filepath)

            # Verify loaded data
            assert new_manager.num_tasks() == 2
            assert new_manager.get_index("task_001") == 0
            assert new_manager.get_index("task_002") == 1

        finally:
            if filepath.exists():
                filepath.unlink()

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        manager = TaskIDManager()

        with pytest.raises(FileNotFoundError):
            manager.load_from_file("nonexistent_file.json")

    def test_save_load_empty_manager(self):
        """Test saving and loading empty manager."""
        manager = TaskIDManager()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            manager.save_to_file(filepath)

            new_manager = TaskIDManager()
            new_manager.load_from_file(filepath)

            assert new_manager.num_tasks() == 0

        finally:
            if filepath.exists():
                filepath.unlink()


class TestTaskIDManagerThreadSafety:
    """Test thread safety of task manager."""

    def test_concurrent_registration(self):
        """Test concurrent task registration from multiple threads."""
        manager = TaskIDManager()
        results = {}
        num_threads = 10
        tasks_per_thread = 20

        def register_tasks(thread_id):
            thread_results = []
            for i in range(tasks_per_thread):
                task_id = f"thread_{thread_id}_task_{i:03d}"
                index = manager.register_task(task_id)
                thread_results.append((task_id, index))
            results[thread_id] = thread_results

        # Start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=register_tasks, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        total_expected = num_threads * tasks_per_thread
        assert manager.num_tasks() == total_expected

        # Check that all indices are unique
        all_indices = set()
        for thread_results in results.values():
            for task_id, index in thread_results:
                assert index not in all_indices, f"Duplicate index {index}"
                all_indices.add(index)
                # Verify bidirectional mapping
                assert manager.get_task_id(index) == task_id
                assert manager.get_index(task_id) == index

        assert len(all_indices) == total_expected

    def test_concurrent_access_patterns(self):
        """Test various concurrent access patterns."""
        manager = TaskIDManager()

        # Pre-register some tasks
        for i in range(10):
            manager.register_task(f"initial_task_{i:03d}")

        results = {"registrations": [], "lookups": [], "errors": []}

        def register_worker():
            try:
                for i in range(50):
                    task_id = f"reg_task_{threading.current_thread().ident}_{i:03d}"
                    index = manager.register_task(task_id)
                    results["registrations"].append((task_id, index))
            except Exception as e:
                results["errors"].append(e)

        def lookup_worker():
            try:
                for i in range(100):
                    # Look up existing tasks
                    task_id = f"initial_task_{i % 10:03d}"
                    index = manager.get_index(task_id)
                    results["lookups"].append((task_id, index))
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                results["errors"].append(e)

        # Start mixed workload
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=register_worker))
        for _ in range(2):
            threads.append(threading.Thread(target=lookup_worker))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(results["errors"]) == 0

        # Verify lookups were successful
        assert len(results["lookups"]) == 200  # 2 threads * 100 lookups
        for task_id, index in results["lookups"]:
            assert index is not None
            assert 0 <= index <= 9  # Initial tasks had indices 0-9


class TestGlobalTaskManager:
    """Test global task manager functionality."""

    def test_global_manager_singleton(self):
        """Test that global manager is a singleton."""
        manager1 = get_global_task_manager()
        manager2 = get_global_task_manager()

        assert manager1 is manager2

    def test_set_global_manager(self):
        """Test setting custom global manager."""
        original_manager = get_global_task_manager()
        custom_manager = TaskIDManager()

        try:
            set_global_task_manager(custom_manager)
            current_manager = get_global_task_manager()

            assert current_manager is custom_manager
            assert current_manager is not original_manager

        finally:
            # Restore original manager
            set_global_task_manager(original_manager)

    def test_global_convenience_functions(self):
        """Test global convenience functions."""
        # Clear global manager for clean test
        manager = get_global_task_manager()
        manager.clear()

        # Test registration
        index1 = register_task_globally("global_task_001")
        index2 = register_task_globally("global_task_002")

        assert index1 == 0
        assert index2 == 1

        # Test lookups
        assert get_task_index_globally("global_task_001") == 0
        assert get_task_index_globally("global_task_002") == 1
        assert get_task_index_globally("nonexistent") is None

        assert get_task_id_globally(0) == "global_task_001"
        assert get_task_id_globally(1) == "global_task_002"
        assert get_task_id_globally(999) is None

        # Test JAX index creation
        jax_index = get_jax_task_index("global_task_001")
        chex.assert_shape(jax_index, ())
        assert jax_index.dtype == jnp.int32
        assert int(jax_index) == 0


class TestUtilityFunctions:
    """Test utility functions for JAX compatibility."""

    def test_create_jax_task_index_with_id(self):
        """Test creating JAX task index with task ID."""
        # Clear global manager
        get_global_task_manager().clear()

        jax_index = create_jax_task_index("test_task")

        chex.assert_shape(jax_index, ())
        assert jax_index.dtype == jnp.int32
        assert int(jax_index) == 0

        # Second call should return same index
        jax_index2 = create_jax_task_index("test_task")
        assert int(jax_index2) == 0

    def test_create_jax_task_index_none(self):
        """Test creating JAX task index with None (dummy task)."""
        jax_index = create_jax_task_index(None)

        chex.assert_shape(jax_index, ())
        assert jax_index.dtype == jnp.int32
        assert int(jax_index) == -1

    def test_extract_task_id_from_index(self):
        """Test extracting task ID from JAX index."""
        # Clear global manager
        get_global_task_manager().clear()

        # Register a task
        jax_index = create_jax_task_index("extract_test")

        # Extract the ID back
        extracted_id = extract_task_id_from_index(jax_index)
        assert extracted_id == "extract_test"

    def test_extract_task_id_from_dummy_index(self):
        """Test extracting task ID from dummy index returns None."""
        dummy_index = create_jax_task_index(None)
        extracted_id = extract_task_id_from_index(dummy_index)
        assert extracted_id is None

    def test_is_dummy_task_index(self):
        """Test checking if task index is dummy."""
        dummy_index = create_jax_task_index(None)
        real_index = create_jax_task_index("real_task")

        assert is_dummy_task_index(dummy_index) == True
        assert is_dummy_task_index(real_index) == False


class TestTemporaryTaskManager:
    """Test temporary task manager context manager."""

    def test_temporary_manager_context(self):
        """Test using temporary task manager context."""
        # Get original global manager and register a task
        original_manager = get_global_task_manager()
        original_manager.clear()
        original_index = register_task_globally("original_task")

        # Use temporary manager
        with TemporaryTaskManager() as temp_manager:
            # Should be using temporary manager now
            current_manager = get_global_task_manager()
            assert current_manager is temp_manager
            assert current_manager is not original_manager

            # Register task in temporary manager
            temp_index = register_task_globally("temp_task")
            assert temp_index == 0  # Fresh manager starts from 0

            # Original task should not be visible
            assert get_task_index_globally("original_task") is None

        # Should be back to original manager
        current_manager = get_global_task_manager()
        assert current_manager is original_manager

        # Original task should be visible again
        assert get_task_index_globally("original_task") == original_index

        # Temporary task should not be visible
        assert get_task_index_globally("temp_task") is None

    def test_temporary_manager_with_custom_manager(self):
        """Test temporary manager with custom manager instance."""
        custom_manager = TaskIDManager()
        custom_manager.register_task("custom_task")

        with TemporaryTaskManager(custom_manager) as temp_manager:
            assert temp_manager is custom_manager

            # Custom task should be visible
            assert get_task_index_globally("custom_task") == 0

    def test_temporary_manager_exception_handling(self):
        """Test that temporary manager restores original even on exception."""
        original_manager = get_global_task_manager()

        try:
            with TemporaryTaskManager():
                # Verify we're using temporary manager
                temp_manager = get_global_task_manager()
                assert temp_manager is not original_manager

                # Raise an exception
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Should be back to original manager despite exception
        current_manager = get_global_task_manager()
        assert current_manager is original_manager


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_large_number_of_tasks(self):
        """Test handling large number of tasks."""
        manager = TaskIDManager()

        num_tasks = 1000
        for i in range(num_tasks):
            task_id = f"task_{i:06d}"
            index = manager.register_task(task_id)
            assert index == i

        assert manager.num_tasks() == num_tasks

        # Verify all mappings are correct
        for i in range(num_tasks):
            task_id = f"task_{i:06d}"
            assert manager.get_index(task_id) == i
            assert manager.get_task_id(i) == task_id

    def test_special_characters_in_task_ids(self):
        """Test task IDs with special characters."""
        manager = TaskIDManager()

        special_ids = [
            "task-with-dashes",
            "task_with_underscores",
            "task.with.dots",
            "task with spaces",
            "task/with/slashes",
            "task:with:colons",
            "task@with@symbols",
            "task#with#hash",
            "task$with$dollar",
            "task%with%percent",
        ]

        for i, task_id in enumerate(special_ids):
            index = manager.register_task(task_id)
            assert index == i
            assert manager.get_task_id(index) == task_id
            assert manager.get_index(task_id) == index

    def test_unicode_task_ids(self):
        """Test task IDs with unicode characters."""
        manager = TaskIDManager()

        unicode_ids = [
            "task_中文",
            "task_русский",
            "task_العربية",
            "task_🚀",
            "task_émojis_🎯",
        ]

        for i, task_id in enumerate(unicode_ids):
            index = manager.register_task(task_id)
            assert index == i
            assert manager.get_task_id(index) == task_id
            assert manager.get_index(task_id) == index

    def test_empty_string_task_id(self):
        """Test empty string as task ID."""
        manager = TaskIDManager()

        index = manager.register_task("")
        assert index == 0
        assert manager.get_task_id(0) == ""
        assert manager.get_index("") == 0

    def test_very_long_task_id(self):
        """Test very long task ID."""
        manager = TaskIDManager()

        long_id = "task_" + "x" * 1000
        index = manager.register_task(long_id)
        assert index == 0
        assert manager.get_task_id(0) == long_id
        assert manager.get_index(long_id) == 0

    def test_jax_array_dtypes(self):
        """Test that JAX arrays have correct dtypes."""
        manager = TaskIDManager()

        manager.register_task("test_task")
        jax_index = manager.get_jax_index("test_task")

        # Should be int32 scalar
        assert jax_index.dtype == jnp.int32
        assert jax_index.ndim == 0

        # Test utility functions too
        util_index = create_jax_task_index("util_task")
        assert util_index.dtype == jnp.int32
        assert util_index.ndim == 0

        dummy_index = create_jax_task_index(None)
        assert dummy_index.dtype == jnp.int32
        assert dummy_index.ndim == 0
