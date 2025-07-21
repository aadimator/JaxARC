"""Memory management and optimization for visualization system.

This module provides memory-efficient handling of visualization data including:
- Lazy loading for large visualization datasets
- Memory usage monitoring and cleanup
- Efficient image compression and storage
- Garbage collection optimization for visualization data
"""

from __future__ import annotations

import gc
import gzip
import pickle
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import numpy as np
from loguru import logger

T = TypeVar("T")


class MemoryUsageMonitor:
    """Monitor memory usage of visualization components."""

    def __init__(self) -> None:
        self.peak_memory_mb: float = 0.0
        self.current_memory_mb: float = 0.0
        self.memory_samples: list[tuple[float, float]] = []  # (timestamp, memory_mb)
        self.lock = threading.Lock()

    def record_memory_usage(self, memory_mb: float) -> None:
        """Record current memory usage."""
        with self.lock:
            self.current_memory_mb = memory_mb
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            self.memory_samples.append((time.time(), memory_mb))

            # Keep only last 1000 samples to prevent unbounded growth
            if len(self.memory_samples) > 1000:
                self.memory_samples = self.memory_samples[-1000:]

    def get_memory_stats(self) -> dict[str, float]:
        """Get memory usage statistics."""
        with self.lock:
            if not self.memory_samples:
                return {"current_mb": 0.0, "peak_mb": 0.0, "avg_mb": 0.0}

            recent_samples = [mem for _, mem in self.memory_samples[-100:]]
            return {
                "current_mb": self.current_memory_mb,
                "peak_mb": self.peak_memory_mb,
                "avg_mb": np.mean(recent_samples),
                "samples_count": len(self.memory_samples),
            }

    def should_trigger_cleanup(self, threshold_mb: float = 500.0) -> bool:
        """Check if memory cleanup should be triggered."""
        return self.current_memory_mb > threshold_mb


class LazyLoader(Generic[T]):
    """Lazy loader for visualization data with automatic cleanup."""

    def __init__(
        self,
        loader_func: Callable[[], T],
        cleanup_func: Callable[[T], None] | None = None,
        max_idle_time: float = 300.0,  # 5 minutes
    ) -> None:
        self.loader_func = loader_func
        self.cleanup_func = cleanup_func
        self.max_idle_time = max_idle_time
        self._data: T | None = None
        self._last_access_time: float = 0.0
        self._lock = threading.Lock()

    def get(self) -> T:
        """Get the data, loading it if necessary."""
        with self._lock:
            current_time = time.time()

            # Check if data needs to be reloaded due to timeout
            if (
                self._data is not None
                and current_time - self._last_access_time > self.max_idle_time
            ):
                self._cleanup_data()

            # Load data if not available
            if self._data is None:
                try:
                    self._data = self.loader_func()
                except Exception as e:
                    logger.error(f"Failed to load lazy data: {e}")
                    raise

            self._last_access_time = current_time
            return self._data

    def _cleanup_data(self) -> None:
        """Clean up loaded data."""
        if self._data is not None:
            if self.cleanup_func:
                try:
                    self.cleanup_func(self._data)
                except Exception as e:
                    logger.warning(f"Error during lazy data cleanup: {e}")
            self._data = None

    def force_cleanup(self) -> None:
        """Force cleanup of loaded data."""
        with self._lock:
            self._cleanup_data()

    def is_loaded(self) -> bool:
        """Check if data is currently loaded."""
        return self._data is not None


class CompressedStorage:
    """Efficient compressed storage for visualization data."""

    def __init__(self, base_path: Path, compression_level: int = 6) -> None:
        self.base_path = Path(base_path)
        self.compression_level = compression_level
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, data: Any, filename: str) -> Path:
        """Save data with compression."""
        filepath = self.base_path / f"{filename}.pkl.gz"

        try:
            with gzip.open(filepath, "wb", compresslevel=self.compression_level) as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return filepath
        except Exception as e:
            logger.error(f"Failed to save compressed data to {filepath}: {e}")
            raise

    def load(self, filename: str) -> Any:
        """Load compressed data."""
        filepath = self.base_path / f"{filename}.pkl.gz"

        if not filepath.exists():
            raise FileNotFoundError(f"Compressed file not found: {filepath}")

        try:
            with gzip.open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load compressed data from {filepath}: {e}")
            raise

    def exists(self, filename: str) -> bool:
        """Check if compressed file exists."""
        filepath = self.base_path / f"{filename}.pkl.gz"
        return filepath.exists()

    def delete(self, filename: str) -> bool:
        """Delete compressed file."""
        filepath = self.base_path / f"{filename}.pkl.gz"
        try:
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to delete compressed file {filepath}: {e}")
            return False

    def get_size_mb(self, filename: str) -> float:
        """Get file size in MB."""
        filepath = self.base_path / f"{filename}.pkl.gz"
        if filepath.exists():
            return filepath.stat().st_size / (1024 * 1024)
        return 0.0

    def list_files(self) -> list[str]:
        """List all compressed files."""
        try:
            return [f.stem.replace(".pkl", "") for f in self.base_path.glob("*.pkl.gz")]
        except Exception as e:
            logger.warning(f"Failed to list compressed files: {e}")
            return []


class VisualizationCache:
    """Memory-efficient cache for visualization data."""

    def __init__(
        self,
        max_memory_mb: float = 100.0,
        max_items: int = 1000,
        storage_path: Path | None = None,
    ) -> None:
        self.max_memory_mb = max_memory_mb
        self.max_items = max_items
        self.cache: dict[str, Any] = {}
        self.access_times: dict[str, float] = {}
        self.memory_usage: dict[str, float] = {}
        self.total_memory_mb: float = 0.0
        self.lock = threading.Lock()

        # Optional compressed storage for overflow
        self.storage: CompressedStorage | None = None
        if storage_path:
            self.storage = CompressedStorage(storage_path)

    def get(self, key: str, loader_func: Callable[[], Any] | None = None) -> Any:
        """Get item from cache, loading if necessary."""
        with self.lock:
            current_time = time.time()

            # Check if item is in memory cache
            if key in self.cache:
                self.access_times[key] = current_time
                return self.cache[key]

            # Try to load from compressed storage
            if self.storage and self.storage.exists(key):
                try:
                    data = self.storage.load(key)
                    self._add_to_memory_cache(key, data, current_time)
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load from storage: {e}")

            # Load using provided function
            if loader_func:
                try:
                    data = loader_func()
                    self._add_to_memory_cache(key, data, current_time)
                    return data
                except Exception as e:
                    logger.error(f"Failed to load data for key {key}: {e}")
                    raise

            raise KeyError(f"Key not found and no loader provided: {key}")

    def put(self, key: str, data: Any) -> None:
        """Put item in cache."""
        with self.lock:
            self._add_to_memory_cache(key, data, time.time())

    def _add_to_memory_cache(self, key: str, data: Any, access_time: float) -> None:
        """Add item to memory cache with cleanup if necessary."""
        # Estimate memory usage (rough approximation)
        try:
            memory_mb = len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)) / (
                1024 * 1024
            )
        except Exception:
            memory_mb = 1.0  # Default estimate

        # Remove existing entry if present
        if key in self.cache:
            self.total_memory_mb -= self.memory_usage.get(key, 0)
            del self.cache[key]

        # Check if we need to make space
        while (
            len(self.cache) >= self.max_items
            or self.total_memory_mb + memory_mb > self.max_memory_mb
        ):
            self._evict_lru_item()

        # Add new item
        self.cache[key] = data
        self.access_times[key] = access_time
        self.memory_usage[key] = memory_mb
        self.total_memory_mb += memory_mb

    def _evict_lru_item(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return

        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])

        # Move to storage if available
        if self.storage:
            try:
                self.storage.save(self.cache[lru_key], lru_key)
            except Exception as e:
                logger.warning(f"Failed to save evicted item to storage: {e}")

        # Remove from memory
        self.total_memory_mb -= self.memory_usage.get(lru_key, 0)
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.memory_usage[lru_key]

    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.memory_usage.clear()
            self.total_memory_mb = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "items_in_memory": len(self.cache),
                "memory_usage_mb": self.total_memory_mb,
                "max_memory_mb": self.max_memory_mb,
                "memory_utilization": self.total_memory_mb / self.max_memory_mb,
                "storage_items": len(self.storage.list_files()) if self.storage else 0,
            }


class GarbageCollectionOptimizer:
    """Optimize garbage collection for visualization workloads."""

    def __init__(self) -> None:
        self.original_thresholds = gc.get_threshold()
        self.gc_stats: list[dict[str, Any]] = []
        self.last_gc_time: float = 0.0

    def optimize_for_visualization(self) -> None:
        """Optimize GC settings for visualization workloads."""
        # Increase thresholds to reduce GC frequency during visualization
        # This trades some memory for better performance
        gc.set_threshold(1000, 15, 15)  # More lenient than default (700, 10, 10)
        logger.debug("Optimized GC thresholds for visualization workload")

    def restore_original_settings(self) -> None:
        """Restore original GC settings."""
        gc.set_threshold(*self.original_thresholds)
        logger.debug("Restored original GC thresholds")

    def force_cleanup(self) -> dict[str, int]:
        """Force garbage collection and return statistics."""
        current_time = time.time()

        # Collect statistics before GC
        before_counts = gc.get_count()

        # Force collection of all generations
        collected = {
            "generation_0": gc.collect(0),
            "generation_1": gc.collect(1),
            "generation_2": gc.collect(2),
        }

        # Collect statistics after GC
        after_counts = gc.get_count()

        stats = {
            "timestamp": current_time,
            "before_counts": before_counts,
            "after_counts": after_counts,
            "collected": collected,
            "total_collected": sum(collected.values()),
        }

        self.gc_stats.append(stats)
        self.last_gc_time = current_time

        # Keep only last 100 GC stats
        if len(self.gc_stats) > 100:
            self.gc_stats = self.gc_stats[-100:]

        logger.debug(f"Forced GC collected {stats['total_collected']} objects")
        return collected

    def should_force_gc(self, interval_seconds: float = 60.0) -> bool:
        """Check if forced GC should be triggered."""
        return time.time() - self.last_gc_time > interval_seconds

    def get_gc_stats(self) -> dict[str, Any]:
        """Get garbage collection statistics."""
        if not self.gc_stats:
            return {}

        recent_stats = self.gc_stats[-10:]  # Last 10 GC runs
        total_collected = sum(stat["total_collected"] for stat in recent_stats)

        return {
            "total_gc_runs": len(self.gc_stats),
            "recent_total_collected": total_collected,
            "avg_collected_per_run": total_collected / len(recent_stats)
            if recent_stats
            else 0,
            "last_gc_time": self.last_gc_time,
            "current_counts": gc.get_count(),
            "current_thresholds": gc.get_threshold(),
        }


class MemoryManager:
    """Comprehensive memory management for visualization system."""

    def __init__(
        self,
        max_cache_memory_mb: float = 100.0,
        max_total_memory_mb: float = 500.0,
        storage_path: Path | None = None,
    ) -> None:
        self.max_total_memory_mb = max_total_memory_mb
        self.monitor = MemoryUsageMonitor()
        self.cache = VisualizationCache(max_cache_memory_mb, storage_path=storage_path)
        self.gc_optimizer = GarbageCollectionOptimizer()
        self.lazy_loaders: list[weakref.ref[LazyLoader[Any]]] = []

        # Start with optimized GC settings
        self.gc_optimizer.optimize_for_visualization()

    def register_lazy_loader(self, loader: LazyLoader[Any]) -> None:
        """Register a lazy loader for cleanup management."""
        self.lazy_loaders.append(weakref.ref(loader))

    def check_memory_usage(self) -> None:
        """Check current memory usage and trigger cleanup if needed."""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.monitor.record_memory_usage(memory_mb)

            if self.monitor.should_trigger_cleanup(self.max_total_memory_mb):
                self.cleanup_memory()

        except ImportError:
            logger.warning("psutil not available for memory monitoring")
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")

    def cleanup_memory(self) -> dict[str, Any]:
        """Perform comprehensive memory cleanup."""
        logger.info("Performing memory cleanup...")

        cleanup_stats = {
            "lazy_loaders_cleaned": 0,
            "cache_items_before": len(self.cache.cache),
            "gc_collected": {},
        }

        # Clean up lazy loaders
        active_loaders = []
        for loader_ref in self.lazy_loaders:
            loader = loader_ref()
            if loader is not None:
                if loader.is_loaded():
                    loader.force_cleanup()
                    cleanup_stats["lazy_loaders_cleaned"] += 1
                active_loaders.append(loader_ref)

        self.lazy_loaders = active_loaders

        # Clear cache
        self.cache.clear()
        cleanup_stats["cache_items_after"] = len(self.cache.cache)

        # Force garbage collection
        cleanup_stats["gc_collected"] = self.gc_optimizer.force_cleanup()

        logger.info(f"Memory cleanup completed: {cleanup_stats}")
        return cleanup_stats

    def get_memory_report(self) -> dict[str, Any]:
        """Get comprehensive memory usage report."""
        return {
            "monitor_stats": self.monitor.get_memory_stats(),
            "cache_stats": self.cache.get_stats(),
            "gc_stats": self.gc_optimizer.get_gc_stats(),
            "active_lazy_loaders": len(
                [ref for ref in self.lazy_loaders if ref() is not None]
            ),
            "max_total_memory_mb": self.max_total_memory_mb,
        }

    def __enter__(self) -> MemoryManager:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.cleanup_memory()
        self.gc_optimizer.restore_original_settings()


# Global memory manager instance
_global_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def create_lazy_visualization_loader(
    data_path: Path,
    loader_func: Callable[[Path], Any],
    max_idle_time: float = 300.0,
) -> LazyLoader[Any]:
    """Create a lazy loader for visualization data.

    Args:
        data_path: Path to the data file
        loader_func: Function to load the data
        max_idle_time: Maximum idle time before cleanup

    Returns:
        LazyLoader instance
    """

    def load_data() -> Any:
        return loader_func(data_path)

    loader = LazyLoader(load_data, max_idle_time=max_idle_time)
    get_memory_manager().register_lazy_loader(loader)
    return loader


def optimize_array_memory(arr: np.ndarray) -> np.ndarray:
    """Optimize numpy array memory usage.

    Args:
        arr: Input array

    Returns:
        Memory-optimized array
    """
    # Use smallest possible dtype
    if arr.dtype == np.int64:
        if np.all((arr >= -128) & (arr <= 127)):
            return arr.astype(np.int8)
        if np.all((arr >= -32768) & (arr <= 32767)):
            return arr.astype(np.int16)
        if np.all((arr >= -2147483648) & (arr <= 2147483647)):
            return arr.astype(np.int32)

    elif arr.dtype == np.float64:
        # Check if we can use float32 without significant precision loss
        arr_f32 = arr.astype(np.float32)
        if np.allclose(arr, arr_f32, rtol=1e-6):
            return arr_f32

    return arr
