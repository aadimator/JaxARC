"""Tests for memory management system."""

from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from jaxarc.utils.visualization.memory_manager import (
    CompressedStorage,
    GarbageCollectionOptimizer,
    LazyLoader,
    MemoryManager,
    MemoryUsageMonitor,
    VisualizationCache,
    create_lazy_visualization_loader,
    get_memory_manager,
    optimize_array_memory,
)


class TestMemoryUsageMonitor:
    """Test memory usage monitoring."""

    def test_record_memory_usage(self):
        """Test recording memory usage."""
        monitor = MemoryUsageMonitor()

        monitor.record_memory_usage(100.0)
        monitor.record_memory_usage(150.0)
        monitor.record_memory_usage(120.0)

        stats = monitor.get_memory_stats()
        assert stats["current_mb"] == 120.0
        assert stats["peak_mb"] == 150.0
        assert stats["samples_count"] == 3
        assert stats["avg_mb"] == pytest.approx(123.33, rel=1e-2)

    def test_should_trigger_cleanup(self):
        """Test cleanup trigger logic."""
        monitor = MemoryUsageMonitor()

        monitor.record_memory_usage(400.0)
        assert not monitor.should_trigger_cleanup(500.0)

        monitor.record_memory_usage(600.0)
        assert monitor.should_trigger_cleanup(500.0)

    def test_sample_limit(self):
        """Test that samples are limited to prevent unbounded growth."""
        monitor = MemoryUsageMonitor()

        # Add more than 1000 samples
        for i in range(1200):
            monitor.record_memory_usage(float(i))

        stats = monitor.get_memory_stats()
        assert stats["samples_count"] == 1000  # Should be capped at 1000


class TestLazyLoader:
    """Test lazy loading functionality."""

    def test_basic_loading(self):
        """Test basic lazy loading."""
        load_count = 0

        def loader():
            nonlocal load_count
            load_count += 1
            return {"data": "test_value", "count": load_count}

        lazy_loader = LazyLoader(loader)

        # First access should load
        data1 = lazy_loader.get()
        assert data1["data"] == "test_value"
        assert data1["count"] == 1
        assert load_count == 1

        # Second access should return cached data
        data2 = lazy_loader.get()
        assert data2["count"] == 1  # Same data
        assert load_count == 1  # No additional load

    def test_timeout_cleanup(self):
        """Test automatic cleanup after timeout."""
        load_count = 0

        def loader():
            nonlocal load_count
            load_count += 1
            return {"count": load_count}

        # Very short timeout for testing
        lazy_loader = LazyLoader(loader, max_idle_time=0.1)

        # Load data
        data1 = lazy_loader.get()
        assert data1["count"] == 1
        assert lazy_loader.is_loaded()

        # Wait for timeout
        time.sleep(0.2)

        # Next access should reload
        data2 = lazy_loader.get()
        assert data2["count"] == 2
        assert load_count == 2

    def test_cleanup_function(self):
        """Test custom cleanup function."""
        cleanup_called = False

        def cleanup_func(data):
            nonlocal cleanup_called
            cleanup_called = True

        def loader():
            return {"data": "test"}

        lazy_loader = LazyLoader(loader, cleanup_func)

        # Load and force cleanup
        lazy_loader.get()
        lazy_loader.force_cleanup()

        assert cleanup_called
        assert not lazy_loader.is_loaded()

    def test_loader_error_handling(self):
        """Test error handling in loader function."""

        def failing_loader():
            raise ValueError("Load failed")

        lazy_loader = LazyLoader(failing_loader)

        with pytest.raises(ValueError, match="Load failed"):
            lazy_loader.get()


class TestCompressedStorage:
    """Test compressed storage functionality."""

    def test_save_and_load(self):
        """Test saving and loading compressed data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = CompressedStorage(Path(temp_dir))

            test_data = {
                "arrays": [np.array([1, 2, 3]), np.array([4, 5, 6])],
                "metadata": "test",
            }

            # Save data
            filepath = storage.save(test_data, "test_file")
            assert filepath.exists()
            assert filepath.suffix == ".gz"

            # Load data
            loaded_data = storage.load("test_file")
            assert loaded_data["metadata"] == "test"
            np.testing.assert_array_equal(loaded_data["arrays"][0], [1, 2, 3])
            np.testing.assert_array_equal(loaded_data["arrays"][1], [4, 5, 6])

    def test_file_operations(self):
        """Test file operation utilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = CompressedStorage(Path(temp_dir))

            # Test non-existent file
            assert not storage.exists("nonexistent")
            assert storage.get_size_mb("nonexistent") == 0.0

            # Save and test existence
            storage.save({"data": "test"}, "test_file")
            assert storage.exists("test_file")
            assert storage.get_size_mb("test_file") > 0.0

            # Test listing files
            files = storage.list_files()
            assert "test_file" in files

            # Test deletion
            assert storage.delete("test_file")
            assert not storage.exists("test_file")
            assert not storage.delete("test_file")  # Already deleted

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = CompressedStorage(Path(temp_dir))

            with pytest.raises(FileNotFoundError):
                storage.load("nonexistent")


class TestVisualizationCache:
    """Test visualization cache functionality."""

    def test_basic_caching(self):
        """Test basic cache operations."""
        cache = VisualizationCache(max_memory_mb=10.0, max_items=5)

        # Test put and get
        cache.put("key1", {"data": "value1"})
        result = cache.get("key1")
        assert result["data"] == "value1"

        # Test get with loader
        def loader():
            return {"data": "loaded_value"}

        result = cache.get("key2", loader)
        assert result["data"] == "loaded_value"

        # Verify it's cached
        result2 = cache.get("key2")
        assert result2["data"] == "loaded_value"

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = VisualizationCache(max_memory_mb=1.0, max_items=2)

        # Fill cache to capacity
        cache.put("key1", {"data": "value1"})
        cache.put("key2", {"data": "value2"})

        # Access key1 to make it more recent
        cache.get("key1")

        # Add key3, should evict key2 (LRU)
        cache.put("key3", {"data": "value3"})

        # key1 and key3 should exist, key2 should be evicted
        assert cache.get("key1")["data"] == "value1"
        assert cache.get("key3")["data"] == "value3"

        with pytest.raises(KeyError):
            cache.get("key2")

    def test_storage_integration(self):
        """Test integration with compressed storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = VisualizationCache(
                max_memory_mb=1.0, max_items=1, storage_path=Path(temp_dir)
            )

            # Add item that will be evicted to storage
            cache.put("key1", {"data": "value1"})
            cache.put("key2", {"data": "value2"})  # Should evict key1 to storage

            # key1 should be loadable from storage
            result = cache.get("key1")
            assert result["data"] == "value1"

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = VisualizationCache(max_memory_mb=10.0, max_items=5)

        cache.put("key1", {"data": "value1"})
        cache.put("key2", {"data": "value2"})

        stats = cache.get_stats()
        assert stats["items_in_memory"] == 2
        assert stats["memory_usage_mb"] > 0
        assert stats["memory_utilization"] < 1.0

    def test_clear_cache(self):
        """Test cache clearing."""
        cache = VisualizationCache(max_memory_mb=10.0, max_items=5)

        cache.put("key1", {"data": "value1"})
        cache.put("key2", {"data": "value2"})

        cache.clear()

        stats = cache.get_stats()
        assert stats["items_in_memory"] == 0
        assert stats["memory_usage_mb"] == 0


class TestGarbageCollectionOptimizer:
    """Test garbage collection optimization."""

    def test_gc_optimization(self):
        """Test GC threshold optimization."""
        optimizer = GarbageCollectionOptimizer()
        original_thresholds = gc.get_threshold()

        # Optimize for visualization
        optimizer.optimize_for_visualization()
        new_thresholds = gc.get_threshold()
        assert new_thresholds != original_thresholds

        # Restore original settings
        optimizer.restore_original_settings()
        restored_thresholds = gc.get_threshold()
        assert restored_thresholds == original_thresholds

    def test_force_cleanup(self):
        """Test forced garbage collection."""
        optimizer = GarbageCollectionOptimizer()

        # Create some garbage
        garbage_list = []
        for i in range(1000):
            garbage_list.append({"data": f"item_{i}", "refs": garbage_list})

        # Force cleanup
        collected = optimizer.force_cleanup()

        assert isinstance(collected, dict)
        assert "generation_0" in collected
        assert "generation_1" in collected
        assert "generation_2" in collected
        assert sum(collected.values()) >= 0

    def test_gc_stats(self):
        """Test GC statistics collection."""
        optimizer = GarbageCollectionOptimizer()

        # Force a collection to generate stats
        optimizer.force_cleanup()

        stats = optimizer.get_gc_stats()
        assert "total_gc_runs" in stats
        assert "recent_total_collected" in stats
        assert "current_counts" in stats
        assert "current_thresholds" in stats

    def test_should_force_gc(self):
        """Test GC trigger logic."""
        optimizer = GarbageCollectionOptimizer()

        # Should trigger initially (no previous GC)
        assert optimizer.should_force_gc(interval_seconds=60.0)

        # Force GC
        optimizer.force_cleanup()

        # Should not trigger immediately after
        assert not optimizer.should_force_gc(interval_seconds=60.0)

        # Should trigger with very short interval
        assert optimizer.should_force_gc(interval_seconds=0.0)


class TestMemoryManager:
    """Test comprehensive memory manager."""

    def test_memory_manager_creation(self):
        """Test memory manager creation and configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(
                max_cache_memory_mb=50.0,
                max_total_memory_mb=200.0,
                storage_path=Path(temp_dir),
            )

            assert manager.max_total_memory_mb == 200.0
            assert isinstance(manager.monitor, MemoryUsageMonitor)
            assert isinstance(manager.cache, VisualizationCache)
            assert isinstance(manager.gc_optimizer, GarbageCollectionOptimizer)

    def test_lazy_loader_registration(self):
        """Test lazy loader registration and cleanup."""
        manager = MemoryManager()

        def loader():
            return {"data": "test"}

        lazy_loader = LazyLoader(loader)
        manager.register_lazy_loader(lazy_loader)

        # Load data
        lazy_loader.get()
        assert lazy_loader.is_loaded()

        # Cleanup should clean the loader
        manager.cleanup_memory()
        assert not lazy_loader.is_loaded()

    def test_memory_monitoring(self):
        """Test memory usage monitoring."""
        # Skip this test if psutil is not available
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available")

        with patch("psutil.Process") as mock_process_class:
            # Mock psutil
            mock_process = Mock()
            mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
            mock_process_class.return_value = mock_process

            manager = MemoryManager(max_total_memory_mb=50.0)  # Low threshold

            # Should trigger cleanup due to high memory usage
            with patch.object(manager, "cleanup_memory") as mock_cleanup:
                manager.check_memory_usage()
                mock_cleanup.assert_called_once()

    def test_memory_report(self):
        """Test memory usage report generation."""
        manager = MemoryManager()

        report = manager.get_memory_report()

        assert "monitor_stats" in report
        assert "cache_stats" in report
        assert "gc_stats" in report
        assert "active_lazy_loaders" in report
        assert "max_total_memory_mb" in report

    def test_context_manager(self):
        """Test memory manager as context manager."""
        with MemoryManager() as manager:
            assert isinstance(manager, MemoryManager)

        # Should have cleaned up after exiting context


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_memory_manager(self):
        """Test global memory manager singleton."""
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()

        assert manager1 is manager2  # Should be the same instance

    def test_create_lazy_visualization_loader(self):
        """Test lazy loader creation utility."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")

            def loader_func(path):
                return {"content": path.read_text(), "path": str(path)}

            lazy_loader = create_lazy_visualization_loader(test_file, loader_func)

            data = lazy_loader.get()
            assert data["content"] == "test content"
            assert str(test_file) in data["path"]

    def test_optimize_array_memory(self):
        """Test array memory optimization."""
        # Test int64 to int8 optimization
        arr_int64 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        optimized = optimize_array_memory(arr_int64)
        assert optimized.dtype == np.int8
        np.testing.assert_array_equal(optimized, arr_int64)

        # Test int64 to int16 optimization
        arr_large = np.array([1000, 2000, 3000], dtype=np.int64)
        optimized = optimize_array_memory(arr_large)
        assert optimized.dtype == np.int16

        # Test int64 to int32 optimization
        arr_very_large = np.array([100000, 200000, 300000], dtype=np.int64)
        optimized = optimize_array_memory(arr_very_large)
        assert optimized.dtype == np.int32

        # Test float64 to float32 optimization
        arr_float64 = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        optimized = optimize_array_memory(arr_float64)
        assert optimized.dtype == np.float32
        np.testing.assert_allclose(optimized, arr_float64, rtol=1e-6)

        # Test no optimization needed
        arr_int8 = np.array([1, 2, 3], dtype=np.int8)
        optimized = optimize_array_memory(arr_int8)
        assert optimized.dtype == np.int8
        assert optimized is arr_int8  # Should return same array


class TestIntegration:
    """Test integration between components."""

    def test_full_workflow(self):
        """Test complete memory management workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create memory manager with storage
            manager = MemoryManager(
                max_cache_memory_mb=1.0,  # Small cache
                max_total_memory_mb=100.0,
                storage_path=Path(temp_dir),
            )

            # Create and register lazy loader
            def expensive_loader():
                return {"large_data": np.random.rand(1000, 1000)}

            lazy_loader = LazyLoader(expensive_loader)
            manager.register_lazy_loader(lazy_loader)

            # Load data
            data = lazy_loader.get()
            assert "large_data" in data

            # Add data to cache
            manager.cache.put("test_key", {"cached_data": "test"})

            # Get memory report
            report = manager.get_memory_report()
            assert report["cache_stats"]["items_in_memory"] > 0

            # Force cleanup
            cleanup_stats = manager.cleanup_memory()
            assert cleanup_stats["lazy_loaders_cleaned"] > 0

            # Verify cleanup worked
            assert not lazy_loader.is_loaded()
            assert manager.cache.get_stats()["items_in_memory"] == 0
