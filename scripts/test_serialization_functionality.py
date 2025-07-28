#!/usr/bin/env python3
"""
Comprehensive test script for efficient serialization functionality.

This script tests all aspects of the efficient serialization system including:
- Task data exclusion during serialization
- File size reduction validation
- Round-trip accuracy with task_data reconstruction
- Error handling for invalid task indices
- Performance benchmarking
"""

import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
from loguru import logger

# Set up logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")


def test_task_data_exclusion():
    """Test that task_data is properly excluded during serialization."""
    logger.info("Testing task_data exclusion during serialization...")

    try:
        from jaxarc.envs.config import DatasetConfig
        from jaxarc.parsers import ArcAgiParser
        from jaxarc.state import ArcEnvState
        from jaxarc.envs.functional import arc_reset
        from jaxarc.envs.config import JaxArcConfig

        # Create test configuration
        config = JaxArcConfig()

        # Create parser with test data
        parser = ArcAgiParser(config.dataset)

        # Get a test task
        key = jax.random.PRNGKey(42)
        task = parser.get_random_task(key)

        # Create initial state
        state, _ = arc_reset(key, config, task)

        # Test serialization with task_data exclusion
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "test_state.eqx"

            # Save state efficiently
            state.save(str(state_path))

            # Verify file was created
            assert state_path.exists(), "Serialized state file was not created"

            # Load state back
            loaded_state = ArcEnvState.load(str(state_path), parser)

            # Verify task_data was reconstructed correctly
            assert loaded_state.task_data is not None, "Task data was not reconstructed"
            assert loaded_state.task_data.task_index == state.task_data.task_index, (
                "Task index mismatch"
            )

            logger.info("✓ Task data exclusion and reconstruction successful")
            return True

    except Exception as e:
        logger.error(f"✗ Task data exclusion test failed: {e}")
        return False


def test_file_size_reduction():
    """Test that serialized files are significantly smaller without task_data."""
    logger.info("Testing file size reduction...")

    try:
        from jaxarc.envs.config import JaxArcConfig
        from jaxarc.parsers import ArcAgiParser
        from jaxarc.state import ArcEnvState
        from jaxarc.envs.functional import arc_reset
        from jaxarc.utils.serialization_utils import (
            calculate_serialization_savings,
            format_file_size,
        )
        import equinox as eqx

        # Create test configuration
        config = JaxArcConfig()

        # Create parser with test data
        parser = ArcAgiParser(config.dataset)

        # Get a test task
        key = jax.random.PRNGKey(42)
        task = parser.get_random_task(key)

        # Create initial state
        state, _ = arc_reset(key, config, task)

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
            logger.info(
                f"Efficient serialization size: {format_file_size(efficient_size)}"
            )
            logger.info(
                f"Space savings: {savings['percentage']:.1f}% ({format_file_size(savings['savings_bytes'])})"
            )

            # Verify significant savings (should be > 50% for typical ARC tasks)
            assert savings["percentage"] > 50, (
                f"Expected >50% savings, got {savings['percentage']:.1f}%"
            )

            logger.info("✓ File size reduction test successful")
            return True, savings

    except Exception as e:
        logger.error(f"✗ File size reduction test failed: {e}")
        return False, None


def test_round_trip_accuracy():
    """Test that serialization/deserialization maintains accuracy."""
    logger.info("Testing round-trip accuracy...")

    try:
        from jaxarc.envs.config import JaxArcConfig
        from jaxarc.parsers import ArcAgiParser
        from jaxarc.state import ArcEnvState
        from jaxarc.envs.functional import arc_reset, arc_step
        from jaxarc.envs.actions import PointAction
        from jaxarc.utils.serialization_utils import validate_task_data_reconstruction
        import equinox as eqx

        # Create test configuration
        config = JaxArcConfig()

        # Create parser with test data
        parser = ArcAgiParser(config.dataset)

        # Get a test task
        key = jax.random.PRNGKey(42)
        task = parser.get_random_task(key)

        # Create initial state and take some steps
        state, _ = arc_reset(key, config, task)

        # Take a few steps to create a more complex state
        for i in range(3):
            action = PointAction(
                operation=jnp.array(0, dtype=jnp.int32),
                row=jnp.array(5 + i, dtype=jnp.int32),
                col=jnp.array(5 + i, dtype=jnp.int32),
            )
            state, _, _, _, _ = arc_step(state, action, config)

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "test_state.eqx"

            # Save and load state
            state.save(str(state_path))
            loaded_state = ArcEnvState.load(str(state_path), parser)

            # Validate task_data reconstruction
            task_data_valid = validate_task_data_reconstruction(
                state.task_data, loaded_state.task_data
            )
            assert task_data_valid, "Task data reconstruction validation failed"

            # Compare key state fields (excluding task_data which we know is reconstructed)
            assert jnp.array_equal(state.working_grid, loaded_state.working_grid), (
                "Working grid mismatch"
            )
            assert jnp.array_equal(state.target_grid, loaded_state.target_grid), (
                "Target grid mismatch"
            )
            assert state.step_count == loaded_state.step_count, "Step count mismatch"
            assert state.episode_done == loaded_state.episode_done, (
                "Episode done mismatch"
            )
            assert state.current_example_idx == loaded_state.current_example_idx, (
                "Example index mismatch"
            )
            assert jnp.array_equal(state.selected, loaded_state.selected), (
                "Selection mismatch"
            )
            assert state.similarity_score == loaded_state.similarity_score, (
                "Similarity score mismatch"
            )

            # Test action history preservation
            assert state.action_history_length == loaded_state.action_history_length, (
                "Action history length mismatch"
            )
            if state.action_history_length > 0:
                assert jnp.array_equal(
                    state.action_history, loaded_state.action_history
                ), "Action history mismatch"

            logger.info("✓ Round-trip accuracy test successful")
            return True

    except Exception as e:
        logger.error(f"✗ Round-trip accuracy test failed: {e}")
        return False


def test_different_action_formats():
    """Test serialization with different action formats."""
    logger.info("Testing serialization with different action formats...")

    try:
        from jaxarc.envs.config import JaxArcConfig, ActionConfig
        from jaxarc.parsers import ArcAgiParser
        from jaxarc.state import ArcEnvState
        from jaxarc.envs.functional import arc_reset

        formats = ["point", "bbox", "mask"]
        results = {}

        for format_name in formats:
            logger.info(f"  Testing {format_name} format...")

            # Create config with specific action format
            action_config = ActionConfig(selection_format=format_name)
            config = JaxArcConfig(action=action_config)

            # Create parser
            parser = ArcAgiParser(config.dataset)

            # Get a test task
            key = jax.random.PRNGKey(42)
            task = parser.get_random_task(key)

            # Create initial state
            state, _ = arc_reset(key, config, task)

            with tempfile.TemporaryDirectory() as temp_dir:
                state_path = Path(temp_dir) / f"test_state_{format_name}.eqx"

                # Save and load state
                state.save(str(state_path))
                loaded_state = ArcEnvState.load(str(state_path), parser)

                # Verify basic consistency
                assert loaded_state.task_data is not None, (
                    f"Task data not reconstructed for {format_name}"
                )
                assert state.step_count == loaded_state.step_count, (
                    f"Step count mismatch for {format_name}"
                )

                # Record file size
                file_size = state_path.stat().st_size
                results[format_name] = file_size

                logger.info(
                    f"    ✓ {format_name} format successful (size: {file_size} bytes)"
                )

        logger.info("✓ Different action formats test successful")
        return True, results

    except Exception as e:
        logger.error(f"✗ Different action formats test failed: {e}")
        return False, None


def test_error_handling():
    """Test error handling for invalid scenarios."""
    logger.info("Testing error handling...")

    try:
        from jaxarc.envs.config import JaxArcConfig
        from jaxarc.parsers import ArcAgiParser
        from jaxarc.state import ArcEnvState
        from jaxarc.utils.serialization_utils import extract_task_id_from_index

        # Test 1: Invalid task_index extraction
        logger.info("  Testing invalid task_index extraction...")
        try:
            # Test with invalid input
            result = extract_task_id_from_index("not_an_array")
            assert False, "Should have raised ValueError"
        except ValueError:
            logger.info("    ✓ Invalid input properly rejected")

        # Test 2: Unknown task_index (-1)
        logger.info("  Testing unknown task_index...")
        unknown_index = jnp.array(-1, dtype=jnp.int32)
        result = extract_task_id_from_index(unknown_index)
        assert result is None, "Unknown task should return None"
        logger.info("    ✓ Unknown task_index handled correctly")

        # Test 3: Loading non-existent file
        logger.info("  Testing non-existent file loading...")
        try:
            config = JaxArcConfig()
            parser = ArcAgiParser(config.dataset)
            ArcEnvState.load("non_existent_file.eqx", parser)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            logger.info("    ✓ Non-existent file properly rejected")

        logger.info("✓ Error handling test successful")
        return True

    except Exception as e:
        logger.error(f"✗ Error handling test failed: {e}")
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
            assert (
                config.environment.max_episode_steps
                == loaded_config.environment.max_episode_steps
            )
            assert (
                config.dataset.max_grid_height == loaded_config.dataset.max_grid_height
            )
            assert (
                config.action.selection_format == loaded_config.action.selection_format
            )

            logger.info("✓ Configuration serialization successful")
            return True

    except Exception as e:
        logger.error(f"✗ Configuration serialization test failed: {e}")
        return False


def benchmark_serialization_performance():
    """Benchmark serialization performance."""
    logger.info("Benchmarking serialization performance...")

    try:
        import time
        from jaxarc.envs.config import JaxArcConfig
        from jaxarc.parsers import ArcAgiParser
        from jaxarc.state import ArcEnvState
        from jaxarc.envs.functional import arc_reset

        # Create test setup
        config = JaxArcConfig()
        parser = ArcAgiParser(config.dataset)
        key = jax.random.PRNGKey(42)
        task = parser.get_random_task(key)
        state, _ = arc_reset(key, config, task)

        num_iterations = 10

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "benchmark_state.eqx"

            # Benchmark saving
            save_times = []
            for i in range(num_iterations):
                start_time = time.perf_counter()
                state.save(str(state_path))
                end_time = time.perf_counter()
                save_times.append(end_time - start_time)

                # Clean up for next iteration
                if state_path.exists():
                    state_path.unlink()

            # Benchmark loading
            state.save(str(state_path))  # Create file for loading tests
            load_times = []
            for i in range(num_iterations):
                start_time = time.perf_counter()
                loaded_state = ArcEnvState.load(str(state_path), parser)
                end_time = time.perf_counter()
                load_times.append(end_time - start_time)

            avg_save_time = sum(save_times) / len(save_times)
            avg_load_time = sum(load_times) / len(load_times)

            logger.info(f"Average save time: {avg_save_time * 1000:.2f} ms")
            logger.info(f"Average load time: {avg_load_time * 1000:.2f} ms")

            # Verify performance is reasonable (< 100ms for typical operations)
            assert avg_save_time < 0.1, f"Save time too slow: {avg_save_time:.3f}s"
            assert avg_load_time < 0.1, f"Load time too slow: {avg_load_time:.3f}s"

            logger.info("✓ Performance benchmark successful")
            return True, {"save_time": avg_save_time, "load_time": avg_load_time}

    except Exception as e:
        logger.error(f"✗ Performance benchmark failed: {e}")
        return False, None


def main():
    """Run all serialization tests."""
    logger.info("Starting comprehensive serialization functionality tests...\n")

    # Track test results
    results = {}

    # Run all tests
    tests = [
        ("Task Data Exclusion", test_task_data_exclusion),
        ("File Size Reduction", test_file_size_reduction),
        ("Round-trip Accuracy", test_round_trip_accuracy),
        ("Different Action Formats", test_different_action_formats),
        ("Error Handling", test_error_handling),
        ("Configuration Serialization", test_config_serialization),
        ("Performance Benchmark", benchmark_serialization_performance),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {test_name}")
        logger.info("=" * 60)

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
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed / total * 100:.1f}%")

    if passed == total:
        logger.info(
            "🎉 All tests passed! Serialization functionality is working correctly."
        )

        # Print key metrics if available
        if (
            "File Size Reduction" in results
            and results["File Size Reduction"]["success"]
        ):
            savings = results["File Size Reduction"]["data"]
            logger.info(f"📊 File size reduction: {savings['percentage']:.1f}%")

        if (
            "Performance Benchmark" in results
            and results["Performance Benchmark"]["success"]
        ):
            perf = results["Performance Benchmark"]["data"]
            logger.info(f"⚡ Average save time: {perf['save_time'] * 1000:.1f}ms")
            logger.info(f"⚡ Average load time: {perf['load_time'] * 1000:.1f}ms")

        return True
    else:
        logger.error(
            f"❌ {total - passed} tests failed. Please check the implementation."
        )
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
