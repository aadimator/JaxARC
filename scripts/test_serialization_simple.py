#!/usr/bin/env python3
"""
Simple test script for efficient serialization functionality.

This script tests the core serialization logic without requiring
the full ARC dataset, focusing on the key functionality.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
from loguru import logger

# Set up logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")


def create_mock_task():
    """Create a mock JaxArcTask for testing."""
    from jaxarc.types import JaxArcTask
    from jaxarc.utils.task_manager import create_jax_task_index
    
    # Create mock task data with realistic shapes
    return JaxArcTask(
        input_grids_examples=jnp.zeros((10, 30, 30), dtype=jnp.int32),
        input_masks_examples=jnp.ones((10, 30, 30), dtype=jnp.bool_),
        output_grids_examples=jnp.zeros((10, 30, 30), dtype=jnp.int32),
        output_masks_examples=jnp.ones((10, 30, 30), dtype=jnp.bool_),
        num_train_pairs=3,
        test_input_grids=jnp.zeros((3, 30, 30), dtype=jnp.int32),
        test_input_masks=jnp.ones((3, 30, 30), dtype=jnp.bool_),
        true_test_output_grids=jnp.zeros((3, 30, 30), dtype=jnp.int32),
        true_test_output_masks=jnp.ones((3, 30, 30), dtype=jnp.bool_),
        num_test_pairs=1,
        task_index=create_jax_task_index("test_task_001"),
    )


def create_mock_state():
    """Create a mock ArcEnvState for testing."""
    from jaxarc.state import ArcEnvState
    from jaxarc.utils.jax_types import (
        DEFAULT_MAX_TEST_PAIRS,
        DEFAULT_MAX_TRAIN_PAIRS,
        MAX_HISTORY_LENGTH,
        ACTION_RECORD_FIELDS,
        NUM_OPERATIONS,
    )
    
    task_data = create_mock_task()
    
    return ArcEnvState(
        task_data=task_data,
        working_grid=jnp.zeros((30, 30), dtype=jnp.int32),
        working_grid_mask=jnp.ones((30, 30), dtype=jnp.bool_),
        target_grid=jnp.zeros((30, 30), dtype=jnp.int32),
        target_grid_mask=jnp.ones((30, 30), dtype=jnp.bool_),
        step_count=jnp.array(5, dtype=jnp.int32),
        episode_done=jnp.array(False, dtype=jnp.bool_),
        current_example_idx=jnp.array(0, dtype=jnp.int32),
        selected=jnp.zeros((30, 30), dtype=jnp.bool_),
        clipboard=jnp.zeros((30, 30), dtype=jnp.int32),
        similarity_score=jnp.array(0.5, dtype=jnp.float32),
        episode_mode=jnp.array(0, dtype=jnp.int32),
        available_demo_pairs=jnp.ones(DEFAULT_MAX_TRAIN_PAIRS, dtype=jnp.bool_),
        available_test_pairs=jnp.ones(DEFAULT_MAX_TEST_PAIRS, dtype=jnp.bool_),
        demo_completion_status=jnp.zeros(DEFAULT_MAX_TRAIN_PAIRS, dtype=jnp.bool_),
        test_completion_status=jnp.zeros(DEFAULT_MAX_TEST_PAIRS, dtype=jnp.bool_),
        action_history=jnp.zeros((MAX_HISTORY_LENGTH, ACTION_RECORD_FIELDS), dtype=jnp.float32),
        action_history_length=jnp.array(0, dtype=jnp.int32),
        action_history_write_pos=jnp.array(0, dtype=jnp.int32),
        allowed_operations_mask=jnp.ones(NUM_OPERATIONS, dtype=jnp.bool_),
    )


class MockParser:
    """Mock parser for testing serialization without real data."""
    
    def __init__(self):
        self.tasks = {"test_task_001": create_mock_task()}
    
    def get_task_by_id(self, task_id: str):
        if task_id in self.tasks:
            return self.tasks[task_id]
        raise ValueError(f"Task ID '{task_id}' not found")
    
    def get_available_task_ids(self):
        return list(self.tasks.keys())


def test_basic_serialization():
    """Test basic serialization and deserialization."""
    logger.info("Testing basic serialization...")
    
    try:
        from jaxarc.state import ArcEnvState
        
        # Create mock state
        state = create_mock_state()
        parser = MockParser()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "test_state.eqx"
            
            # Save state
            state.save(str(state_path))
            
            # Verify file was created
            assert state_path.exists(), "Serialized state file was not created"
            
            # Load state back
            loaded_state = ArcEnvState.load(str(state_path), parser)
            
            # Verify key fields
            assert loaded_state.step_count == state.step_count, "Step count mismatch"
            assert loaded_state.episode_done == state.episode_done, "Episode done mismatch"
            assert loaded_state.similarity_score == state.similarity_score, "Similarity score mismatch"
            assert loaded_state.task_data is not None, "Task data was not reconstructed"
            
            logger.info("✓ Basic serialization test successful")
            return True
            
    except Exception as e:
        logger.error(f"✗ Basic serialization test failed: {e}")
        return False


def test_file_size_comparison():
    """Test that efficient serialization produces smaller files."""
    logger.info("Testing file size comparison...")
    
    try:
        import equinox as eqx
        from jaxarc.utils.serialization_utils import calculate_serialization_savings, format_file_size
        
        # Create mock state
        state = create_mock_state()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            efficient_path = Path(temp_dir) / "efficient_state.eqx"
            full_path = Path(temp_dir) / "full_state.eqx"
            
            # Save state efficiently (excluding task_data)
            state.save(str(efficient_path))
            
            # Save full state for comparison
            eqx.tree_serialise_leaves(str(full_path), state)
            
            # Compare file sizes
            efficient_size = efficient_path.stat().st_size
            full_size = full_path.stat().st_size
            
            # Calculate savings
            savings = calculate_serialization_savings(full_size, efficient_size)
            
            logger.info(f"Full serialization size: {format_file_size(full_size)}")
            logger.info(f"Efficient serialization size: {format_file_size(efficient_size)}")
            logger.info(f"Space savings: {savings['percentage']:.1f}% ({format_file_size(savings['savings_bytes'])})")
            
            # Verify we got some savings (should be significant for mock data)
            assert efficient_size < full_size, "Efficient serialization should be smaller"
            assert savings['percentage'] > 0, "Should have some space savings"
            
            logger.info("✓ File size comparison test successful")
            return True, savings
            
    except Exception as e:
        logger.error(f"✗ File size comparison test failed: {e}")
        return False, None


def test_task_id_extraction():
    """Test task ID extraction utilities."""
    logger.info("Testing task ID extraction...")
    
    try:
        from jaxarc.utils.serialization_utils import extract_task_id_from_index
        from jaxarc.utils.task_manager import register_task_globally
        
        # Register a test task
        task_id = "test_extraction_task"
        task_index = register_task_globally(task_id)
        
        # Create JAX array with the index
        jax_index = jnp.array(task_index, dtype=jnp.int32)
        
        # Extract task ID back
        extracted_id = extract_task_id_from_index(jax_index)
        
        assert extracted_id == task_id, f"Expected '{task_id}', got '{extracted_id}'"
        
        # Test unknown task (-1)
        unknown_index = jnp.array(-1, dtype=jnp.int32)
        unknown_result = extract_task_id_from_index(unknown_index)
        assert unknown_result is None, "Unknown task should return None"
        
        logger.info("✓ Task ID extraction test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Task ID extraction test failed: {e}")
        return False


def test_config_serialization():
    """Test configuration serialization."""
    logger.info("Testing configuration serialization...")
    
    try:
        from jaxarc.envs.config import JaxArcConfig
        
        # Create test configuration
        config = JaxArcConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.eqx"
            
            # Save and load configuration
            config.save(str(config_path))
            loaded_config = JaxArcConfig.load(str(config_path))
            
            # Verify configuration fields
            assert config.environment.max_episode_steps == loaded_config.environment.max_episode_steps
            assert config.dataset.max_grid_height == loaded_config.dataset.max_grid_height
            assert config.action.selection_format == loaded_config.action.selection_format
            
            logger.info("✓ Configuration serialization test successful")
            return True
            
    except Exception as e:
        logger.error(f"✗ Configuration serialization test failed: {e}")
        return False


def test_error_handling():
    """Test error handling scenarios."""
    logger.info("Testing error handling...")
    
    try:
        from jaxarc.state import ArcEnvState
        from jaxarc.utils.serialization_utils import extract_task_id_from_index
        
        # Test 1: Invalid task_index extraction
        logger.info("  Testing invalid task_index extraction...")
        try:
            extract_task_id_from_index("not_an_array")
            assert False, "Should have raised ValueError"
        except ValueError:
            logger.info("    ✓ Invalid input properly rejected")
        
        # Test 2: Loading non-existent file
        logger.info("  Testing non-existent file loading...")
        try:
            parser = MockParser()
            ArcEnvState.load("non_existent_file.eqx", parser)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            logger.info("    ✓ Non-existent file properly rejected")
        
        logger.info("✓ Error handling test successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error handling test failed: {e}")
        return False


def main():
    """Run all serialization tests."""
    logger.info("Starting simple serialization functionality tests...\n")
    
    # Track test results
    results = {}
    
    # Run all tests
    tests = [
        ("Basic Serialization", test_basic_serialization),
        ("File Size Comparison", test_file_size_comparison),
        ("Task ID Extraction", test_task_id_extraction),
        ("Configuration Serialization", test_config_serialization),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info('='*60)
        
        try:
            result = test_func()
            if isinstance(result, tuple):
                success, data = result
                results[test_name] = {"success": success, "data": data}
            else:
                success = result
                results[test_name] = {"success": success}
                
            if success:
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results[test_name] = {"success": False, "error": str(e)}
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info('='*60)
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All tests passed! Serialization functionality is working correctly.")
        
        # Print key metrics if available
        if "File Size Comparison" in results and results["File Size Comparison"]["success"]:
            savings = results["File Size Comparison"]["data"]
            logger.info(f"📊 File size reduction: {savings['percentage']:.1f}%")
            
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)