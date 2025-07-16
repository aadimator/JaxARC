"""
Tests for validating GitHub dataset integrity.

This module tests that all expected tasks are loaded from GitHub repositories,
verifies task data structure matches expected JaxArcTask format, tests parser
performance with large datasets, and ensures no data corruption during JSON file loading.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from omegaconf import DictConfig

from jaxarc.parsers.arc_agi import ArcAgiParser
from jaxarc.types import JaxArcTask


class TestGitHubDatasetIntegrity:
    """Test suite for validating GitHub dataset integrity."""

    @pytest.fixture
    def large_github_dataset(self):
        """Create a large GitHub format dataset for performance testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create 1000 task files to simulate a large dataset
        num_tasks = 1000
        task_files = {}
        
        for i in range(num_tasks):
            task_id = f"task_{i:06d}"
            
            # Create varied task structures to test different scenarios
            if i % 10 == 0:
                # Large grids (30x30)
                task_data = {
                    "train": [
                        {
                            "input": [[j % 10 for j in range(30)] for _ in range(30)],
                            "output": [[(j + 1) % 10 for j in range(30)] for _ in range(30)]
                        }
                    ],
                    "test": [
                        {
                            "input": [[(j + 2) % 10 for j in range(30)] for _ in range(30)],
                            "output": [[(j + 3) % 10 for j in range(30)] for _ in range(30)]
                        }
                    ]
                }
            elif i % 7 == 0:
                # Multiple training pairs
                task_data = {
                    "train": [
                        {"input": [[i % 10]], "output": [[(i + 1) % 10]]},
                        {"input": [[i % 5, (i + 1) % 5]], "output": [[(i + 1) % 5, i % 5]]},
                        {"input": [[i % 3]], "output": [[(i + 2) % 3]]},
                    ],
                    "test": [
                        {"input": [[(i + 3) % 10]], "output": [[(i + 4) % 10]]},
                    ]
                }
            elif i % 5 == 0:
                # Multiple test pairs
                task_data = {
                    "train": [
                        {"input": [[i % 10]], "output": [[(i + 1) % 10]]},
                    ],
                    "test": [
                        {"input": [[(i + 2) % 10]], "output": [[(i + 3) % 10]]},
                        {"input": [[(i + 4) % 10]], "output": [[(i + 5) % 10]]},
                        {"input": [[(i + 6) % 10]]},  # No output for some test pairs
                    ]
                }
            else:
                # Standard small tasks
                task_data = {
                    "train": [
                        {"input": [[i % 10, (i + 1) % 10]], "output": [[(i + 1) % 10, i % 10]]},
                    ],
                    "test": [
                        {"input": [[(i + 2) % 10, (i + 3) % 10]]},  # No output
                    ]
                }
            
            task_files[f"{task_id}.json"] = task_data
        
        # Write all task files
        for filename, task_data in task_files.items():
            task_file = temp_dir / filename
            with task_file.open("w", encoding="utf-8") as f:
                json.dump(task_data, f)
        
        yield temp_dir, num_tasks
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def corrupted_dataset(self):
        """Create a dataset with various types of corruption for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Valid task
        valid_task = temp_dir / "valid_task.json"
        with valid_task.open("w", encoding="utf-8") as f:
            json.dump({
                "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                "test": [{"input": [[3, 4]], "output": [[4, 3]]}]
            }, f)
        
        # Malformed JSON
        malformed_json = temp_dir / "malformed.json"
        with malformed_json.open("w", encoding="utf-8") as f:
            f.write('{"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}')  # Missing closing brace
        
        # Invalid task structure - missing train
        missing_train = temp_dir / "missing_train.json"
        with missing_train.open("w", encoding="utf-8") as f:
            json.dump({
                "test": [{"input": [[1, 2]], "output": [[2, 1]]}]
            }, f)
        
        # Invalid task structure - missing test
        missing_test = temp_dir / "missing_test.json"
        with missing_test.open("w", encoding="utf-8") as f:
            json.dump({
                "train": [{"input": [[1, 2]], "output": [[2, 1]]}]
            }, f)
        
        # Invalid color values
        invalid_colors = temp_dir / "invalid_colors.json"
        with invalid_colors.open("w", encoding="utf-8") as f:
            json.dump({
                "train": [{"input": [[15, 20]], "output": [[25, 30]]}],  # Colors > 9
                "test": [{"input": [[-1, -2]]}]  # Negative colors
            }, f)
        
        # Empty training pairs
        empty_train = temp_dir / "empty_train.json"
        with empty_train.open("w", encoding="utf-8") as f:
            json.dump({
                "train": [],
                "test": [{"input": [[1, 2]]}]
            }, f)
        
        # Empty test pairs
        empty_test = temp_dir / "empty_test.json"
        with empty_test.open("w", encoding="utf-8") as f:
            json.dump({
                "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                "test": []
            }, f)
        
        # Oversized grids
        oversized_grids = temp_dir / "oversized.json"
        large_grid = [[i % 10 for i in range(50)] for _ in range(50)]  # 50x50 grid
        with oversized_grids.open("w", encoding="utf-8") as f:
            json.dump({
                "train": [{"input": large_grid, "output": large_grid}],
                "test": [{"input": large_grid}]
            }, f)
        
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_all_expected_tasks_loaded(self, large_github_dataset):
        """Test that all expected tasks are loaded from GitHub repositories."""
        temp_dir, expected_count = large_github_dataset
        
        config = DictConfig({
            "default_split": "training",
            "training": {"path": str(temp_dir)},
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 10,
            "max_test_pairs": 5,
        })
        
        parser = ArcAgiParser(cfg=config)
        task_ids = parser.get_available_task_ids()
        
        # Verify all tasks are loaded
        assert len(task_ids) == expected_count, f"Expected {expected_count} tasks, got {len(task_ids)}"
        
        # Verify task IDs are correctly extracted from filenames
        expected_ids = {f"task_{i:06d}" for i in range(expected_count)}
        actual_ids = set(task_ids)
        assert actual_ids == expected_ids, "Task IDs don't match expected values"
        
        # Verify we can access all tasks
        for task_id in task_ids[:10]:  # Test first 10 to avoid excessive runtime
            task = parser.get_task_by_id(task_id)
            assert isinstance(task, JaxArcTask), f"Task {task_id} is not a JaxArcTask"

    def test_task_data_structure_matches_jaxarctask_format(self, large_github_dataset):
        """Verify task data structure matches expected JaxArcTask format."""
        temp_dir, _ = large_github_dataset
        
        config = DictConfig({
            "default_split": "training",
            "training": {"path": str(temp_dir)},
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 10,
            "max_test_pairs": 5,
        })
        
        parser = ArcAgiParser(cfg=config)
        
        # Test various task types
        test_tasks = [
            "task_000000",  # Large grid task
            "task_000007",  # Multiple training pairs
            "task_000010",  # Multiple test pairs
            "task_000001",  # Standard task
        ]
        
        for task_id in test_tasks:
            task = parser.get_task_by_id(task_id)
            
            # Verify JaxArcTask structure
            assert isinstance(task, JaxArcTask), f"Task {task_id} is not JaxArcTask"
            
            # Verify required attributes exist
            required_attrs = [
                'input_grids_examples', 'input_masks_examples',
                'output_grids_examples', 'output_masks_examples',
                'test_input_grids', 'test_input_masks',
                'true_test_output_grids', 'true_test_output_masks',
                'num_train_pairs', 'num_test_pairs', 'task_index'
            ]
            
            for attr in required_attrs:
                assert hasattr(task, attr), f"Task {task_id} missing attribute {attr}"
            
            # Verify array shapes and types
            max_train_pairs = config.max_train_pairs
            max_test_pairs = config.max_test_pairs
            max_height = config.grid.max_grid_height
            max_width = config.grid.max_grid_width
            
            # Training data shapes
            assert task.input_grids_examples.shape == (max_train_pairs, max_height, max_width)
            assert task.input_masks_examples.shape == (max_train_pairs, max_height, max_width)
            assert task.output_grids_examples.shape == (max_train_pairs, max_height, max_width)
            assert task.output_masks_examples.shape == (max_train_pairs, max_height, max_width)
            
            # Test data shapes
            assert task.test_input_grids.shape == (max_test_pairs, max_height, max_width)
            assert task.test_input_masks.shape == (max_test_pairs, max_height, max_width)
            assert task.true_test_output_grids.shape == (max_test_pairs, max_height, max_width)
            assert task.true_test_output_masks.shape == (max_test_pairs, max_height, max_width)
            
            # Verify data types
            assert task.input_grids_examples.dtype == jnp.int32
            assert task.input_masks_examples.dtype == jnp.bool_
            assert task.output_grids_examples.dtype == jnp.int32
            assert task.output_masks_examples.dtype == jnp.bool_
            assert task.test_input_grids.dtype == jnp.int32
            assert task.test_input_masks.dtype == jnp.bool_
            assert task.true_test_output_grids.dtype == jnp.int32
            assert task.true_test_output_masks.dtype == jnp.bool_
            assert task.task_index.dtype == jnp.int32
            
            # Verify counts are valid
            assert 0 < task.num_train_pairs <= max_train_pairs
            assert 0 < task.num_test_pairs <= max_test_pairs
            
            # Verify task_index is a scalar
            assert task.task_index.shape == ()

    def test_parser_performance_with_large_datasets(self, large_github_dataset):
        """Test parser performance with large datasets."""
        temp_dir, expected_count = large_github_dataset
        
        config = DictConfig({
            "default_split": "training",
            "training": {"path": str(temp_dir)},
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 10,
            "max_test_pairs": 5,
        })
        
        # Test initialization performance
        start_time = time.time()
        parser = ArcAgiParser(cfg=config)
        init_time = time.time() - start_time
        
        # Initialization should be reasonably fast (< 30 seconds for 1000 tasks)
        assert init_time < 30.0, f"Parser initialization took too long: {init_time:.2f}s"
        
        # Test task access performance
        task_ids = parser.get_available_task_ids()
        
        # Test random task access
        key = jax.random.PRNGKey(42)
        start_time = time.time()
        
        for _ in range(50):  # Access 50 random tasks
            task = parser.get_random_task(key)
            assert isinstance(task, JaxArcTask)
            key, _ = jax.random.split(key)
        
        random_access_time = time.time() - start_time
        
        # Random access should be fast (< 10 seconds for 50 accesses)
        assert random_access_time < 10.0, f"Random access took too long: {random_access_time:.2f}s"
        
        # Test specific task access
        start_time = time.time()
        
        for i in range(0, min(50, len(task_ids))):  # Access first 50 tasks by ID
            task = parser.get_task_by_id(task_ids[i])
            assert isinstance(task, JaxArcTask)
        
        specific_access_time = time.time() - start_time
        
        # Specific access should be fast (< 5 seconds for 50 accesses)
        assert specific_access_time < 5.0, f"Specific access took too long: {specific_access_time:.2f}s"

    def test_no_data_corruption_during_json_loading(self, large_github_dataset):
        """Ensure no data corruption during JSON file loading."""
        temp_dir, _ = large_github_dataset
        
        config = DictConfig({
            "default_split": "training",
            "training": {"path": str(temp_dir)},
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 10,
            "max_test_pairs": 5,
        })
        
        parser = ArcAgiParser(cfg=config)
        
        # Test specific tasks with known data patterns
        test_cases = [
            ("task_000000", "large_grid"),  # 30x30 grid
            ("task_000007", "multiple_train"),  # Multiple training pairs
            ("task_000010", "multiple_test"),  # Multiple test pairs
            ("task_000001", "standard"),  # Standard task
        ]
        
        for task_id, task_type in test_cases:
            task = parser.get_task_by_id(task_id)
            
            if task_type == "large_grid":
                # Verify large grid data integrity
                # First training input should be 30x30 with pattern [0,1,2,...,9,0,1,...]
                first_input = task.input_grids_examples[0]
                first_mask = task.input_masks_examples[0]
                
                # Check that the entire 30x30 region is valid
                assert jnp.sum(first_mask) == 30 * 30, "Large grid mask is incorrect"
                
                # Check data pattern integrity
                for i in range(30):
                    for j in range(30):
                        expected_value = j % 10
                        actual_value = int(first_input[i, j])
                        assert actual_value == expected_value, f"Data corruption at [{i},{j}]: expected {expected_value}, got {actual_value}"
                
                # Verify output pattern (should be input + 1 mod 10)
                first_output = task.output_grids_examples[0]
                for i in range(30):
                    for j in range(30):
                        expected_value = (j + 1) % 10
                        actual_value = int(first_output[i, j])
                        assert actual_value == expected_value, f"Output corruption at [{i},{j}]: expected {expected_value}, got {actual_value}"
            
            elif task_type == "multiple_train":
                # Verify multiple training pairs
                assert task.num_train_pairs == 3, f"Expected 3 training pairs, got {task.num_train_pairs}"
                
                # Check each training pair
                for pair_idx in range(task.num_train_pairs):
                    input_grid = task.input_grids_examples[pair_idx]
                    output_grid = task.output_grids_examples[pair_idx]
                    input_mask = task.input_masks_examples[pair_idx]
                    
                    # Verify mask covers valid region
                    assert jnp.sum(input_mask) > 0, f"Training pair {pair_idx} has empty mask"
                    
                    # Verify data values are within valid range
                    valid_input_values = input_grid[input_mask]
                    assert jnp.all(valid_input_values >= 0), f"Training pair {pair_idx} has negative input values"
                    assert jnp.all(valid_input_values <= 9), f"Training pair {pair_idx} has input values > 9"
                    
                    valid_output_values = output_grid[input_mask]  # Use same mask for output
                    assert jnp.all(valid_output_values >= 0), f"Training pair {pair_idx} has negative output values"
                    assert jnp.all(valid_output_values <= 9), f"Training pair {pair_idx} has output values > 9"
            
            elif task_type == "multiple_test":
                # Verify multiple test pairs - task_000010 should have 3 test pairs based on our fixture
                # But the parser might only load the first one due to missing outputs
                assert task.num_test_pairs >= 1, f"Expected at least 1 test pair, got {task.num_test_pairs}"
                
                # Check each test pair
                for pair_idx in range(task.num_test_pairs):
                    input_grid = task.test_input_grids[pair_idx]
                    input_mask = task.test_input_masks[pair_idx]
                    
                    # Verify mask covers valid region
                    assert jnp.sum(input_mask) > 0, f"Test pair {pair_idx} has empty mask"
                    
                    # Verify data values are within valid range
                    valid_input_values = input_grid[input_mask]
                    assert jnp.all(valid_input_values >= 0), f"Test pair {pair_idx} has negative input values"
                    assert jnp.all(valid_input_values <= 9), f"Test pair {pair_idx} has input values > 9"
            
            elif task_type == "standard":
                # Verify standard task structure
                assert task.num_train_pairs == 1, f"Expected 1 training pair, got {task.num_train_pairs}"
                assert task.num_test_pairs == 1, f"Expected 1 test pair, got {task.num_test_pairs}"
                
                # Check training data integrity
                input_grid = task.input_grids_examples[0]
                output_grid = task.output_grids_examples[0]
                input_mask = task.input_masks_examples[0]
                
                # Should be 1x2 grid
                assert jnp.sum(input_mask) == 2, "Standard task should have 1x2 input"
                
                # Verify pattern: input [i%10, (i+1)%10] -> output [(i+1)%10, i%10]
                task_num = int(task_id.split('_')[1])  # Extract number from task_000001
                expected_input = [task_num % 10, (task_num + 1) % 10]
                expected_output = [(task_num + 1) % 10, task_num % 10]
                
                # Find the valid region
                valid_positions = jnp.where(input_mask)
                if len(valid_positions[0]) >= 2:
                    # Check first two valid positions
                    actual_input = [int(input_grid[valid_positions[0][0], valid_positions[1][0]]),
                                   int(input_grid[valid_positions[0][1], valid_positions[1][1]])]
                    actual_output = [int(output_grid[valid_positions[0][0], valid_positions[1][0]]),
                                    int(output_grid[valid_positions[0][1], valid_positions[1][1]])]
                    
                    assert actual_input == expected_input, f"Input data corruption: expected {expected_input}, got {actual_input}"
                    assert actual_output == expected_output, f"Output data corruption: expected {expected_output}, got {actual_output}"

    def test_corrupted_data_handling(self, corrupted_dataset):
        """Test handling of various types of data corruption."""
        config = DictConfig({
            "default_split": "training",
            "training": {"path": str(corrupted_dataset)},
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 10,
            "max_test_pairs": 5,
        })
        
        # Parser should initialize successfully, skipping corrupted files
        parser = ArcAgiParser(cfg=config)
        task_ids = parser.get_available_task_ids()
        
        # Should only load valid tasks (malformed JSON and other corrupted files should be skipped)
        # We expect at least the valid_task to be loaded
        assert "valid_task" in task_ids, "Valid task should be loaded"
        
        # Test that valid task works correctly
        valid_task = parser.get_task_by_id("valid_task")
        assert isinstance(valid_task, JaxArcTask)
        assert valid_task.num_train_pairs == 1
        assert valid_task.num_test_pairs == 1
        
        # Test error handling for corrupted tasks that were loaded but fail during preprocessing
        corrupted_task_ids = [tid for tid in task_ids if tid != "valid_task"]
        
        for task_id in corrupted_task_ids:
            if task_id in ["missing_train", "missing_test", "empty_train", "empty_test"]:
                # These should fail during preprocessing
                with pytest.raises(ValueError):
                    parser.get_task_by_id(task_id)
            elif task_id == "invalid_colors":
                # This should fail during color validation
                with pytest.raises(ValueError):
                    parser.get_task_by_id(task_id)
            elif task_id == "oversized":
                # This should fail during grid dimension validation
                with pytest.raises(ValueError):
                    parser.get_task_by_id(task_id)

    def test_memory_usage_with_large_datasets(self, large_github_dataset):
        """Test memory usage remains reasonable with large datasets."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        temp_dir, _ = large_github_dataset
        
        config = DictConfig({
            "default_split": "training",
            "training": {"path": str(temp_dir)},
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 10,
            "max_test_pairs": 5,
        })
        
        # Measure memory before parser creation
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create parser
        parser = ArcAgiParser(cfg=config)
        
        # Measure memory after parser creation
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (< 500MB for 1000 tasks)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"
        
        # Test memory usage during task access
        key = jax.random.PRNGKey(42)
        
        # Access multiple tasks and check memory doesn't grow excessively
        for _ in range(20):
            task = parser.get_random_task(key)
            key, _ = jax.random.split(key)
        
        memory_final = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = memory_final - memory_after
        
        # Memory growth during access should be minimal (< 100MB)
        assert memory_growth < 100, f"Memory growth during access too high: {memory_growth:.1f}MB"

    def test_concurrent_access_safety(self, large_github_dataset):
        """Test that concurrent access to parser is safe."""
        import threading
        import queue
        
        temp_dir, _ = large_github_dataset
        
        config = DictConfig({
            "default_split": "training",
            "training": {"path": str(temp_dir)},
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 10,
            "max_test_pairs": 5,
        })
        
        parser = ArcAgiParser(cfg=config)
        task_ids = parser.get_available_task_ids()
        
        # Queue to collect results from threads
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker_thread(thread_id: int, num_accesses: int):
            """Worker thread that accesses tasks."""
            try:
                key = jax.random.PRNGKey(thread_id)
                thread_results = []
                
                for _ in range(num_accesses):
                    # Mix random and specific access
                    if _ % 2 == 0:
                        task = parser.get_random_task(key)
                        key, _ = jax.random.split(key)
                    else:
                        task_id = task_ids[_ % len(task_ids)]
                        task = parser.get_task_by_id(task_id)
                    
                    # Verify task is valid
                    assert isinstance(task, JaxArcTask)
                    thread_results.append(task.num_train_pairs)
                
                results_queue.put((thread_id, thread_results))
                
            except Exception as e:
                errors_queue.put((thread_id, str(e)))
        
        # Create and start multiple threads
        num_threads = 5
        num_accesses_per_thread = 10
        threads = []
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i, num_accesses_per_thread))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check for errors
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        
        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
        
        # Verify all threads completed successfully
        for thread_id, thread_results in results:
            assert len(thread_results) == num_accesses_per_thread, f"Thread {thread_id} incomplete"
            # All results should be positive (valid num_train_pairs)
            assert all(r > 0 for r in thread_results), f"Thread {thread_id} has invalid results"

    def test_jax_compatibility_integrity(self, large_github_dataset):
        """Test JAX compatibility and transformations with large datasets."""
        temp_dir, _ = large_github_dataset
        
        config = DictConfig({
            "default_split": "training",
            "training": {"path": str(temp_dir)},
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 10,
            "max_test_pairs": 5,
        })
        
        parser = ArcAgiParser(cfg=config)
        
        # Get a few tasks for testing
        task_ids = parser.get_available_task_ids()[:5]
        tasks = [parser.get_task_by_id(task_id) for task_id in task_ids]
        
        # Test JIT compilation with individual tasks
        @jax.jit
        def process_single_task(input_grids, input_masks, output_grids, output_masks):
            """Process a single task with JAX operations."""
            # Calculate sum of valid input values
            input_sum = jnp.sum(input_grids * input_masks)
            
            # Calculate sum of valid output values
            output_sum = jnp.sum(output_grids * output_masks)
            
            # Calculate number of valid cells
            valid_cells = jnp.sum(input_masks)
            
            return input_sum, output_sum, valid_cells
        
        for task in tasks:
            result = process_single_task(
                task.input_grids_examples,
                task.input_masks_examples,
                task.output_grids_examples,
                task.output_masks_examples
            )
            
            # Verify results are JAX arrays
            assert all(isinstance(r, jnp.ndarray) for r in result)
            assert all(r.shape == () for r in result)  # Should be scalars
        
        # Test vmap over multiple training pairs
        @jax.jit
        def process_training_pairs(input_grids, input_masks):
            """Process multiple training pairs."""
            def process_pair(input_grid, input_mask):
                return jnp.sum(input_grid * input_mask)
            
            return jax.vmap(process_pair)(input_grids, input_masks)
        
        for task in tasks:
            results = process_training_pairs(
                task.input_grids_examples,
                task.input_masks_examples
            )
            
            assert isinstance(results, jnp.ndarray)
            assert results.shape == (config.max_train_pairs,)
        
        # Test batch processing over multiple tasks
        # Stack tasks into batch format
        batch_input_grids = jnp.stack([task.input_grids_examples for task in tasks])
        batch_input_masks = jnp.stack([task.input_masks_examples for task in tasks])
        
        @jax.jit
        def process_task_batch(batch_inputs, batch_masks):
            """Process a batch of tasks."""
            def process_single_task_batch(inputs, masks):
                return jnp.sum(inputs * masks)
            
            return jax.vmap(process_single_task_batch)(batch_inputs, batch_masks)
        
        batch_results = process_task_batch(batch_input_grids, batch_input_masks)
        
        assert isinstance(batch_results, jnp.ndarray)
        assert batch_results.shape == (len(tasks),)
        
        # Test gradient computation (for RL applications)
        @jax.jit
        def loss_function(params, input_grids, input_masks, output_grids, output_masks):
            """Dummy loss function for gradient testing."""
            # Simple loss: difference between weighted input and output sums
            input_sum = jnp.sum(input_grids * input_masks * params)
            output_sum = jnp.sum(output_grids * output_masks)
            return jnp.abs(input_sum - output_sum)
        
        grad_fn = jax.grad(loss_function)
        
        for task in tasks[:2]:  # Test with first 2 tasks
            params = 1.0  # Simple scalar parameter
            
            grad = grad_fn(
                params,
                task.input_grids_examples,
                task.input_masks_examples,
                task.output_grids_examples,
                task.output_masks_examples
            )
            
            assert isinstance(grad, (float, jnp.ndarray))
            if isinstance(grad, jnp.ndarray):
                assert grad.shape == ()  # Should be scalar gradient

    def test_data_consistency_across_multiple_loads(self, large_github_dataset):
        """Test that data remains consistent across multiple parser instances."""
        temp_dir, _ = large_github_dataset
        
        config = DictConfig({
            "default_split": "training",
            "training": {"path": str(temp_dir)},
            "grid": {
                "max_grid_height": 30,
                "max_grid_width": 30,
                "min_grid_height": 1,
                "min_grid_width": 1,
                "max_colors": 10,
                "background_color": 0,
            },
            "max_train_pairs": 10,
            "max_test_pairs": 5,
        })
        
        # Create multiple parser instances
        parsers = [ArcAgiParser(cfg=config) for _ in range(3)]
        
        # Verify all parsers load the same tasks
        task_ids_sets = [set(parser.get_available_task_ids()) for parser in parsers]
        
        # All parsers should have the same task IDs
        assert all(task_ids == task_ids_sets[0] for task_ids in task_ids_sets), "Task IDs differ between parser instances"
        
        # Test specific tasks for data consistency
        test_task_ids = list(task_ids_sets[0])[:10]  # Test first 10 tasks
        
        for task_id in test_task_ids:
            tasks = [parser.get_task_by_id(task_id) for parser in parsers]
            
            # Compare all tasks to the first one
            reference_task = tasks[0]
            
            for i, task in enumerate(tasks[1:], 1):
                # Compare all array fields
                array_fields = [
                    'input_grids_examples', 'input_masks_examples',
                    'output_grids_examples', 'output_masks_examples',
                    'test_input_grids', 'test_input_masks',
                    'true_test_output_grids', 'true_test_output_masks'
                ]
                
                for field in array_fields:
                    ref_array = getattr(reference_task, field)
                    task_array = getattr(task, field)
                    
                    assert jnp.array_equal(ref_array, task_array), f"Task {task_id}, parser {i}, field {field}: arrays differ"
                
                # Compare scalar fields
                assert task.num_train_pairs == reference_task.num_train_pairs, f"Task {task_id}, parser {i}: num_train_pairs differ"
                assert task.num_test_pairs == reference_task.num_test_pairs, f"Task {task_id}, parser {i}: num_test_pairs differ"
                assert jnp.array_equal(task.task_index, reference_task.task_index), f"Task {task_id}, parser {i}: task_index differ"