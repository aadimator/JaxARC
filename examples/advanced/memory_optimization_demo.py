#!/usr/bin/env python3
"""
Demonstration of memory management and optimization for visualization system.

This example shows how to use the memory management features including:
- Lazy loading for large datasets
- Memory usage monitoring and cleanup
- Efficient image compression and storage
- Garbage collection optimization
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from loguru import logger

from jaxarc.utils.visualization import (
    CompressedStorage,
    GarbageCollectionOptimizer,
    LazyLoader,
    MemoryManager,
    VisualizationCache,
    create_lazy_visualization_loader,
    optimize_array_memory,
)


def demo_lazy_loading():
    """Demonstrate lazy loading functionality."""
    logger.info("=== Lazy Loading Demo ===")

    # Create a function that loads expensive data
    load_count = 0

    def expensive_data_loader():
        nonlocal load_count
        load_count += 1
        logger.info(f"Loading expensive data (call #{load_count})")
        # Simulate expensive computation
        time.sleep(0.1)
        return {
            "large_array": np.random.default_rng().random((1000, 1000)),
            "metadata": f"Loaded at call #{load_count}",
            "timestamp": time.time(),
        }

    # Create lazy loader
    lazy_loader = LazyLoader(expensive_data_loader, max_idle_time=2.0)

    # First access - should load
    logger.info("First access:")
    data1 = lazy_loader.get()
    logger.info(f"Data metadata: {data1['metadata']}")
    logger.info(f"Array shape: {data1['large_array'].shape}")

    # Second access - should use cached data
    logger.info("Second access (should be cached):")
    data2 = lazy_loader.get()
    logger.info(f"Data metadata: {data2['metadata']}")
    logger.info(f"Same data? {data1['timestamp'] == data2['timestamp']}")

    # Wait for timeout and access again
    logger.info("Waiting for timeout...")
    time.sleep(2.5)

    logger.info("Third access (should reload after timeout):")
    data3 = lazy_loader.get()
    logger.info(f"Data metadata: {data3['metadata']}")
    logger.info(f"Same data? {data1['timestamp'] == data3['timestamp']}")

    # Force cleanup
    lazy_loader.force_cleanup()
    logger.info("Forced cleanup completed")

    return load_count


def demo_compressed_storage():
    """Demonstrate compressed storage functionality."""
    logger.info("=== Compressed Storage Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        storage = CompressedStorage(Path(temp_dir))

        # Create test data with various types
        test_data = {
            "arrays": [
                np.random.default_rng().random((100, 100)),
                np.random.default_rng().integers(0, 10, (50, 50)),
            ],
            "metadata": {
                "description": "Test visualization data",
                "created_at": time.time(),
                "version": "1.0",
            },
            "config": {"param1": 42, "param2": "test_value"},
        }

        # Save data
        logger.info("Saving test data...")
        filepath = storage.save(test_data, "test_visualization")
        logger.info(f"Saved to: {filepath}")
        logger.info(f"File size: {storage.get_size_mb('test_visualization'):.2f} MB")

        # Load data back
        logger.info("Loading test data...")
        loaded_data = storage.load("test_visualization")

        # Verify data integrity
        logger.info("Verifying data integrity...")
        assert (
            loaded_data["metadata"]["description"]
            == test_data["metadata"]["description"]
        )
        assert loaded_data["config"]["param1"] == test_data["config"]["param1"]
        np.testing.assert_array_equal(loaded_data["arrays"][0], test_data["arrays"][0])
        np.testing.assert_array_equal(loaded_data["arrays"][1], test_data["arrays"][1])

        logger.info("Data integrity verified!")

        # List files
        files = storage.list_files()
        logger.info(f"Files in storage: {files}")

        # Delete file
        deleted = storage.delete("test_visualization")
        logger.info(f"File deleted: {deleted}")

        return storage.get_size_mb("test_visualization")


def demo_visualization_cache():
    """Demonstrate visualization cache functionality."""
    logger.info("=== Visualization Cache Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create cache with small memory limit to trigger eviction
        cache = VisualizationCache(
            max_memory_mb=1.0,  # Small limit
            max_items=3,
            storage_path=Path(temp_dir),
        )

        # Add items to cache
        for i in range(5):
            data = {
                "visualization": np.random.default_rng().random((100, 100)),
                "metadata": f"Visualization {i}",
                "step": i,
            }
            cache.put(f"viz_{i}", data)
            logger.info(f"Added visualization {i}")

            # Show cache stats
            stats = cache.get_stats()
            logger.info(
                f"Cache: {stats['items_in_memory']} items, "
                f"{stats['memory_usage_mb']:.2f} MB, "
                f"{stats['storage_items']} in storage"
            )

        # Try to access evicted items (should load from storage)
        logger.info("Accessing potentially evicted items...")
        for i in range(5):
            try:
                data = cache.get(f"viz_{i}")
                logger.info(f"Retrieved visualization {i}: {data['metadata']}")
            except KeyError:
                logger.warning(f"Visualization {i} not found")

        # Clear cache
        cache.clear()
        final_stats = cache.get_stats()
        logger.info(f"After clear: {final_stats['items_in_memory']} items in memory")

        return final_stats


def demo_gc_optimization():
    """Demonstrate garbage collection optimization."""
    logger.info("=== Garbage Collection Optimization Demo ===")

    optimizer = GarbageCollectionOptimizer()

    # Get original settings
    original_thresholds = optimizer.original_thresholds
    logger.info(f"Original GC thresholds: {original_thresholds}")

    # Optimize for visualization
    optimizer.optimize_for_visualization()
    optimized_thresholds = (
        optimizer.gc_optimizer.get_threshold()
        if hasattr(optimizer, "gc_optimizer")
        else None
    )
    logger.info(f"Optimized GC thresholds: {optimized_thresholds}")

    # Create some garbage to collect
    garbage_list = []
    for i in range(1000):
        # Create circular references to generate garbage
        item = {"id": i, "data": list(range(100)), "refs": garbage_list}
        garbage_list.append(item)

    logger.info(f"Created {len(garbage_list)} objects with circular references")

    # Force garbage collection
    logger.info("Forcing garbage collection...")
    collected = optimizer.force_cleanup()
    logger.info(f"Collected objects: {collected}")

    # Get GC stats
    stats = optimizer.get_gc_stats()
    logger.info(f"GC stats: {stats}")

    # Restore original settings
    optimizer.restore_original_settings()
    restored_thresholds = (
        optimizer.gc_optimizer.get_threshold()
        if hasattr(optimizer, "gc_optimizer")
        else None
    )
    logger.info(f"Restored GC thresholds: {restored_thresholds}")

    return collected


def demo_memory_manager():
    """Demonstrate comprehensive memory manager."""
    logger.info("=== Memory Manager Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create memory manager with custom settings
        manager = MemoryManager(
            max_cache_memory_mb=2.0,
            max_total_memory_mb=50.0,
            storage_path=Path(temp_dir),
        )

        # Create and register lazy loaders
        loaders = []
        for i in range(3):

            def create_loader(index):
                def loader():
                    logger.info(f"Loading dataset {index}")
                    return {
                        "data": np.random.default_rng().random((200, 200)),
                        "index": index,
                        "size": "large",
                    }

                return loader

            lazy_loader = LazyLoader(create_loader(i))
            manager.register_lazy_loader(lazy_loader)
            loaders.append(lazy_loader)

        # Load data from lazy loaders
        logger.info("Loading data from lazy loaders...")
        for i, loader in enumerate(loaders):
            data = loader.get()
            logger.info(f"Loaded dataset {i}: shape {data['data'].shape}")

        # Add data to cache
        logger.info("Adding data to cache...")
        for i in range(5):
            cache_data = {
                "cached_viz": np.random.default_rng().random((50, 50)),
                "id": i,
            }
            manager.cache.put(f"cache_item_{i}", cache_data)

        # Get memory report
        logger.info("Getting memory report...")
        report = manager.get_memory_report()
        logger.info(f"Memory report: {report}")

        # Simulate high memory usage and trigger cleanup
        logger.info("Simulating memory cleanup...")
        cleanup_stats = manager.cleanup_memory()
        logger.info(f"Cleanup stats: {cleanup_stats}")

        # Verify cleanup worked
        post_cleanup_report = manager.get_memory_report()
        logger.info(f"Post-cleanup report: {post_cleanup_report}")

        return cleanup_stats


def demo_array_optimization():
    """Demonstrate array memory optimization."""
    logger.info("=== Array Optimization Demo ===")

    # Test different array types
    test_arrays = [
        ("int64_small", np.array([1, 2, 3, 4, 5], dtype=np.int64)),
        ("int64_medium", np.array([1000, 2000, 3000], dtype=np.int64)),
        ("int64_large", np.array([100000, 200000, 300000], dtype=np.int64)),
        ("float64", np.array([1.1, 2.2, 3.3], dtype=np.float64)),
        ("already_int8", np.array([1, 2, 3], dtype=np.int8)),
    ]

    total_savings = 0

    for name, arr in test_arrays:
        original_size = arr.nbytes
        optimized = optimize_array_memory(arr)
        optimized_size = optimized.nbytes
        savings = original_size - optimized_size

        logger.info(
            f"{name}: {arr.dtype} -> {optimized.dtype}, "
            f"{original_size} -> {optimized_size} bytes "
            f"(saved {savings} bytes)"
        )

        total_savings += savings

        # Verify data integrity
        if arr.dtype.kind == "f":  # float
            np.testing.assert_allclose(arr, optimized, rtol=1e-6)
        else:  # integer
            np.testing.assert_array_equal(arr, optimized)

    logger.info(f"Total memory savings: {total_savings} bytes")
    return total_savings


def demo_integration_workflow():
    """Demonstrate complete memory management workflow."""
    logger.info("=== Integration Workflow Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Use context manager for automatic cleanup
        with MemoryManager(
            max_cache_memory_mb=1.0,
            max_total_memory_mb=100.0,
            storage_path=Path(temp_dir),
        ) as manager:
            # Create lazy loader for large visualization dataset
            def load_visualization_data():
                logger.info("Loading large visualization dataset...")
                return {
                    "episode_data": [
                        {
                            "step": i,
                            "grid": np.random.default_rng().integers(0, 10, (30, 30)),
                            "action": np.random.default_rng().random((30, 30)),
                            "reward": np.random.default_rng().random(),
                        }
                        for i in range(100)
                    ],
                    "metadata": {"total_steps": 100, "episode_id": "test_episode"},
                }

            viz_loader = create_lazy_visualization_loader(
                Path(temp_dir) / "viz_data.pkl", lambda _: load_visualization_data()
            )

            # Load and process data
            logger.info("Processing visualization data...")
            viz_data = viz_loader.get()
            logger.info(f"Loaded episode with {len(viz_data['episode_data'])} steps")

            # Optimize arrays in the data
            optimized_data = []
            total_savings = 0

            for step_data in viz_data["episode_data"][:5]:  # Process first 5 steps
                optimized_step = {
                    "step": step_data["step"],
                    "grid": optimize_array_memory(step_data["grid"]),
                    "action": optimize_array_memory(step_data["action"]),
                    "reward": step_data["reward"],
                }
                optimized_data.append(optimized_step)

                # Calculate savings
                original_size = step_data["grid"].nbytes + step_data["action"].nbytes
                optimized_size = (
                    optimized_step["grid"].nbytes + optimized_step["action"].nbytes
                )
                total_savings += original_size - optimized_size

            logger.info(f"Memory optimization saved {total_savings} bytes")

            # Cache processed data
            manager.cache.put("processed_episode", optimized_data)

            # Get final report
            final_report = manager.get_memory_report()
            logger.info(f"Final memory report: {final_report}")

        # Context manager automatically cleans up
        logger.info("Memory manager context exited - automatic cleanup performed")

        return total_savings


def main():
    """Run all memory optimization demonstrations."""
    logger.info("Starting Memory Optimization Demonstration")
    logger.info("=" * 60)

    try:
        # Run all demos
        load_count = demo_lazy_loading()
        storage_size = demo_compressed_storage()
        cache_stats = demo_visualization_cache()
        gc_collected = demo_gc_optimization()
        demo_memory_manager()
        array_savings = demo_array_optimization()
        integration_savings = demo_integration_workflow()

        logger.info("=" * 60)
        logger.info("All memory optimization demonstrations completed successfully!")

        # Summary
        logger.info("\n=== Summary ===")
        logger.info(f"Lazy loading calls: {load_count}")
        logger.info(f"Compressed storage final size: {storage_size:.2f} MB")
        logger.info(f"Cache items after clear: {cache_stats['items_in_memory']}")
        logger.info(f"GC collected objects: {sum(gc_collected.values())}")
        logger.info(f"Array optimization savings: {array_savings} bytes")
        logger.info(f"Integration workflow savings: {integration_savings} bytes")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
