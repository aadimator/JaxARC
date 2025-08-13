#!/usr/bin/env python3
"""
Baseline Performance Benchmark for JaxARC

This script establishes baseline performance metrics for the current JaxARC implementation
before applying optimizations. It measures:

1. Single environment performance (arc_reset, arc_step)
2. Batch environment performance (batch_reset, batch_step)
3. Scaling behavior with different batch sizes
4. System information and configuration

Results are saved to JSON for comparison after optimization.

Usage Examples:
    # Basic benchmark with default settings
    pixi run python benchmarks/baseline_benchmark.py
    
    # With description and tag
    pixi run python benchmarks/baseline_benchmark.py -d "Phase 1: Removed callbacks" -t "phase1"
    
    # Custom batch sizes
    pixi run python benchmarks/baseline_benchmark.py -b "1,10,100,1000,5000"
    
    # Automatic batch size range
    pixi run python benchmarks/baseline_benchmark.py --min-batch 1 --max-batch 1000
    
    # Custom step counts
    pixi run python benchmarks/baseline_benchmark.py --single-steps 2000 --batch-steps 100
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import typer
from loguru import logger

# JaxARC imports
from jaxarc.configs import JaxArcConfig
from jaxarc.configs.action_config import ActionConfig
from jaxarc.configs.dataset_config import DatasetConfig
from jaxarc.configs.environment_config import EnvironmentConfig
from jaxarc.configs.grid_initialization_config import GridInitializationConfig
from jaxarc.configs.history_config import HistoryConfig
from jaxarc.configs.logging_config import LoggingConfig
from jaxarc.configs.reward_config import RewardConfig
from jaxarc.configs.storage_config import StorageConfig
from jaxarc.configs.visualization_config import VisualizationConfig
from jaxarc.configs.wandb_config import WandbConfig
from jaxarc.envs.actions import (
    create_bbox_action,
    create_mask_action,
    create_point_action,
)
from jaxarc.envs.functional import arc_reset, arc_step, batch_reset, batch_step
from jaxarc.types import JaxArcTask
from jaxarc.utils.jax_types import EPISODE_MODE_TRAIN


def get_system_info() -> Dict[str, Any]:
    """Collect system information for benchmark context."""
    try:
        # JAX device info
        devices = jax.devices()
        device_info = {
            "count": len(devices),
            "types": [str(d.device_kind) for d in devices],
            "platform": str(devices[0].platform) if devices else "unknown",
        }
    except Exception:
        device_info = {"error": "Could not get JAX device info"}

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_devices": device_info,
        "cpu_count": platform.processor() or "unknown",
        "architecture": platform.architecture()[0],
    }


def create_minimal_config() -> JaxArcConfig:
    """Create a minimal configuration for benchmarking."""
    return JaxArcConfig(
        environment=EnvironmentConfig(
            max_episode_steps=100,
            auto_reset=True,
        ),
        dataset=DatasetConfig(
            max_grid_height=30,
            max_grid_width=30,
            max_train_pairs=3,
            max_test_pairs=1,
            background_color=0,
            dataset_path="data/arc-prize-2024",  # Will create dummy data
            task_split="train",
        ),
        action=ActionConfig(
            selection_format="point",
            dynamic_action_filtering=False,
        ),
        reward=RewardConfig(
            similarity_weight=1.0,
            step_penalty=-0.01,
            success_bonus=10.0,
        ),
        grid_initialization=GridInitializationConfig(
            mode="demo",
            demo_pair_selection="random",
        ),
        visualization=VisualizationConfig.from_hydra({}),
        storage=StorageConfig(),
        logging=LoggingConfig(
            log_operations=False,  # Disable logging for performance
            log_level="ERROR",
        ),
        wandb=WandbConfig.from_hydra({}),
        history=HistoryConfig(
            enabled=False,  # Disable history for performance
        ),
    )


def create_dummy_task_data(config: JaxArcConfig) -> JaxArcTask:
    """Create dummy task data for benchmarking."""
    max_height = config.dataset.max_grid_height
    max_width = config.dataset.max_grid_width
    max_train_pairs = config.dataset.max_train_pairs
    max_test_pairs = config.dataset.max_test_pairs

    # Create simple dummy grids (checkerboard pattern)
    def create_checkerboard(height: int, width: int, offset: int = 0) -> jnp.ndarray:
        """Create a simple checkerboard pattern for testing."""
        grid = jnp.zeros((height, width), dtype=jnp.int32)
        for i in range(height):
            for j in range(width):
                grid = grid.at[i, j].set((i + j + offset) % 2)
        return grid

    # Training data
    input_grids_examples = jnp.zeros(
        (max_train_pairs, max_height, max_width), dtype=jnp.int32
    )
    output_grids_examples = jnp.zeros(
        (max_train_pairs, max_height, max_width), dtype=jnp.int32
    )
    input_masks_examples = jnp.zeros(
        (max_train_pairs, max_height, max_width), dtype=jnp.bool_
    )
    output_masks_examples = jnp.zeros(
        (max_train_pairs, max_height, max_width), dtype=jnp.bool_
    )

    # Fill with dummy data
    for i in range(max_train_pairs):
        # Create different patterns for each pair
        input_grid = create_checkerboard(max_height, max_width, i)
        output_grid = create_checkerboard(max_height, max_width, i + 1)

        input_grids_examples = input_grids_examples.at[i].set(input_grid)
        output_grids_examples = output_grids_examples.at[i].set(output_grid)
        input_masks_examples = input_masks_examples.at[i].set(
            jnp.ones((max_height, max_width), dtype=jnp.bool_)
        )
        output_masks_examples = output_masks_examples.at[i].set(
            jnp.ones((max_height, max_width), dtype=jnp.bool_)
        )

    # Test data
    test_input_grids = jnp.zeros(
        (max_test_pairs, max_height, max_width), dtype=jnp.int32
    )
    test_input_masks = jnp.zeros(
        (max_test_pairs, max_height, max_width), dtype=jnp.bool_
    )
    true_test_output_grids = jnp.zeros(
        (max_test_pairs, max_height, max_width), dtype=jnp.int32
    )
    true_test_output_masks = jnp.zeros(
        (max_test_pairs, max_height, max_width), dtype=jnp.bool_
    )

    for i in range(max_test_pairs):
        test_grid = create_checkerboard(max_height, max_width, max_train_pairs + i)
        test_output = create_checkerboard(
            max_height, max_width, max_train_pairs + i + 1
        )

        test_input_grids = test_input_grids.at[i].set(test_grid)
        test_input_masks = test_input_masks.at[i].set(
            jnp.ones((max_height, max_width), dtype=jnp.bool_)
        )
        true_test_output_grids = true_test_output_grids.at[i].set(test_output)
        true_test_output_masks = true_test_output_masks.at[i].set(
            jnp.ones((max_height, max_width), dtype=jnp.bool_)
        )

    return JaxArcTask(
        input_grids_examples=input_grids_examples,
        input_masks_examples=input_masks_examples,
        output_grids_examples=output_grids_examples,
        output_masks_examples=output_masks_examples,
        num_train_pairs=max_train_pairs,
        test_input_grids=test_input_grids,
        test_input_masks=test_input_masks,
        true_test_output_grids=true_test_output_grids,
        true_test_output_masks=true_test_output_masks,
        num_test_pairs=max_test_pairs,
        task_index=jnp.array(0, dtype=jnp.int32),
    )


def create_test_actions(grid_shape: tuple[int, int], num_actions: int = 100) -> list:
    """Create a variety of test actions for benchmarking."""
    actions = []
    height, width = grid_shape

    for i in range(num_actions):
        action_type = i % 3
        operation = i % 35  # Operations 0-34

        if action_type == 0:  # Point action
            row = i % height
            col = i % width
            actions.append(create_point_action(operation, row, col))
        elif action_type == 1:  # Bbox action
            r1 = i % (height // 2)
            c1 = i % (width // 2)
            r2 = min(r1 + 5, height - 1)
            c2 = min(c1 + 5, width - 1)
            actions.append(create_bbox_action(operation, r1, c1, r2, c2))
        else:  # Mask action
            mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
            # Create a small square mask
            start_r = i % (height - 5)
            start_c = i % (width - 5)
            mask = mask.at[start_r : start_r + 3, start_c : start_c + 3].set(True)
            actions.append(create_mask_action(operation, mask))

    return actions


def benchmark_single_environment(
    config: JaxArcConfig, task_data: JaxArcTask, num_steps: int = 1000
) -> Dict[str, float]:
    """Benchmark single environment performance."""
    logger.info(f"Benchmarking single environment with {num_steps} steps...")

    # Create test actions
    grid_shape = (config.dataset.max_grid_height, config.dataset.max_grid_width)
    actions = create_test_actions(grid_shape, num_steps)

    # Warmup - JIT compilation for ALL action types
    key = jax.random.PRNGKey(42)
    state, obs = arc_reset(key, config, task_data, EPISODE_MODE_TRAIN)

    # Create one of each action type for warmup
    point_action = create_point_action(0, 0, 0)
    bbox_action = create_bbox_action(0, 0, 0, 1, 1)
    mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
    mask_action = create_mask_action(0, mask)

    # Run one step for each action type to ensure all paths are compiled
    logger.info("Warming up JIT compiler for all action types...")
    arc_step(state, point_action, config)[2].block_until_ready()
    arc_step(state, bbox_action, config)[2].block_until_ready()
    arc_step(state, mask_action, config)[2].block_until_ready()
    logger.info("Warmup complete.")

    # Benchmark reset
    reset_times = []
    for i in range(10):  # Multiple resets for averaging
        key = jax.random.PRNGKey(i)
        start_time = time.perf_counter()
        state, obs = arc_reset(key, config, task_data, EPISODE_MODE_TRAIN)
        obs.block_until_ready()  # Ensure computation completes
        end_time = time.perf_counter()
        reset_times.append(end_time - start_time)

    # Benchmark steps
    key = jax.random.PRNGKey(42)
    state, obs = arc_reset(key, config, task_data, EPISODE_MODE_TRAIN)

    start_time = time.perf_counter()
    for i in range(num_steps):
        action = actions[i % len(actions)]
        state, obs, reward, done, info = arc_step(state, action, config)
        reward.block_until_ready()  # Ensure computation completes

        # Reset if episode is done
        if done:
            key = jax.random.PRNGKey(i + 1000)
            state, obs = arc_reset(key, config, task_data, EPISODE_MODE_TRAIN)

    end_time = time.perf_counter()

    total_time = end_time - start_time
    steps_per_second = num_steps / total_time
    avg_reset_time = np.mean(reset_times)

    return {
        "steps_per_second": steps_per_second,
        "total_time": total_time,
        "avg_reset_time": avg_reset_time,
        "num_steps": num_steps,
    }


def benchmark_batch_environment(
    config: JaxArcConfig, task_data: JaxArcTask, batch_size: int, num_steps: int = 100
) -> Dict[str, float]:
    """Benchmark batch environment performance."""
    logger.info(
        f"Benchmarking batch environment with batch_size={batch_size}, {num_steps} steps..."
    )

    # Create batch keys and actions
    grid_shape = (config.dataset.max_grid_height, config.dataset.max_grid_width)
    base_key = jax.random.PRNGKey(42)
    keys = jax.random.split(base_key, batch_size)

    # Create batch actions (same action for all environments for simplicity)
    actions = create_test_actions(grid_shape, num_steps)

    # Warmup - JIT compilation for ALL action types in batch mode
    states, obs = batch_reset(keys, config, task_data)

    # Create one of each action type for batch warmup
    point_action = create_point_action(0, 0, 0)
    bbox_action = create_bbox_action(0, 0, 0, 1, 1)
    mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
    mask_action = create_mask_action(0, mask)

    # Warm up batch_step with all action types
    logger.info("Warming up batch JIT compiler for all action types...")
    for action in [point_action, bbox_action, mask_action]:
        batched_action = jax.tree.map(
            lambda x: jnp.tile(x, (batch_size,) + (1,) * x.ndim), action
        )
        batch_step(states, batched_action, config)[2].block_until_ready()
    logger.info("Batch warmup complete.")

    # Benchmark batch reset
    start_time = time.perf_counter()
    states, obs = batch_reset(keys, config, task_data)
    obs.block_until_ready()
    end_time = time.perf_counter()
    batch_reset_time = end_time - start_time

    # Benchmark batch steps
    start_time = time.perf_counter()
    for i in range(num_steps):
        action = actions[i % len(actions)]
        # Replicate action across batch
        batched_action = jax.tree.map(
            lambda x: jnp.tile(x, (batch_size,) + (1,) * x.ndim), action
        )
        states, obs, rewards, dones, infos = batch_step(states, batched_action, config)
        rewards.block_until_ready()  # Ensure computation completes

    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_env_steps = batch_size * num_steps
    steps_per_second = total_env_steps / total_time

    return {
        "batch_size": batch_size,
        "steps_per_second": steps_per_second,
        "total_time": total_time,
        "batch_reset_time": batch_reset_time,
        "total_env_steps": total_env_steps,
        "num_steps": num_steps,
    }


def run_scaling_benchmark(
    config: JaxArcConfig, task_data: JaxArcTask, batch_sizes: Optional[List[int]] = None, num_steps: int = 50
) -> Dict[str, Any]:
    """Run scaling benchmark with different batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500, 1000, 2000, 5000, 10000]
    
    logger.info(f"Running scaling benchmark with batch sizes: {batch_sizes}")
    scaling_results = {}

    for batch_size in batch_sizes:
        try:
            result = benchmark_batch_environment(
                config, task_data, batch_size, num_steps=num_steps
            )
            scaling_results[str(batch_size)] = result
            logger.info(
                f"Batch size {batch_size}: {result['steps_per_second']:.0f} SPS"
            )
        except Exception as e:
            logger.error(f"Failed to benchmark batch size {batch_size}: {e}")
            scaling_results[str(batch_size)] = {"error": str(e)}

    return scaling_results


def main(
    description: str = typer.Option("", "--description", "-d", help="Description of the benchmark run (e.g., 'Phase 1: Removed callbacks')"),
    tag: str = typer.Option("", "--tag", "-t", help="Short tag for the benchmark (e.g., 'phase1', 'baseline')"),
    batch_sizes: Optional[str] = typer.Option(None, "--batch-sizes", "-b", help="Comma-separated list of batch sizes to test (e.g., '1,10,100,1000')"),
    min_batch: Optional[int] = typer.Option(None, "--min-batch", help="Minimum batch size for automatic range"),
    max_batch: Optional[int] = typer.Option(None, "--max-batch", help="Maximum batch size for automatic range"),
    single_steps: int = typer.Option(1000, "--single-steps", help="Number of steps for single environment benchmark"),
    batch_steps: int = typer.Option(50, "--batch-steps", help="Number of steps for batch environment benchmark"),
):
    """
    JaxARC Performance Benchmark
    
    This script establishes baseline performance metrics for the current JaxARC implementation.
    
    Examples:
    
        # Basic benchmark with default settings
        python benchmarks/baseline_benchmark.py
        
        # With description and tag
        python benchmarks/baseline_benchmark.py -d "Phase 1: Removed callbacks" -t "phase1"
        
        # Custom batch sizes
        python benchmarks/baseline_benchmark.py -b "1,10,100,1000,5000"
        
        # Automatic batch size range
        python benchmarks/baseline_benchmark.py --min-batch 1 --max-batch 1000
        
        # Custom step counts
        python benchmarks/baseline_benchmark.py --single-steps 2000 --batch-steps 100
    """
    # Set loguru to only show ERROR and above to reduce noise
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    benchmark_title = "JaxARC Performance Benchmark"
    if description:
        benchmark_title += f": {description}"
    
    logger.info(f"Starting {benchmark_title}")

    # Create output directory
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    config = create_minimal_config()
    task_data = create_dummy_task_data(config)

    # Collect system info
    system_info = get_system_info()
    logger.info(f"System: {system_info['platform']}")
    logger.info(f"JAX version: {system_info['jax_version']}")
    logger.info(f"JAX devices: {system_info['jax_devices']}")

    # Parse batch sizes
    parsed_batch_sizes = None
    if batch_sizes:
        try:
            parsed_batch_sizes = [int(x.strip()) for x in batch_sizes.split(",")]
        except ValueError:
            typer.echo(f"Error: Invalid batch sizes format: {batch_sizes}", err=True)
            raise typer.Exit(1)
    elif min_batch is not None and max_batch is not None:
        # Generate automatic range
        if min_batch >= max_batch:
            typer.echo("Error: min-batch must be less than max-batch", err=True)
            raise typer.Exit(1)
        
        # Generate linear scale between min and max
        num_points = 10  # Number of points in the range
        
        parsed_batch_sizes = []
        for i in range(num_points):
            batch_size = int(min_batch + (max_batch - min_batch) * i / (num_points - 1))
            if batch_size not in parsed_batch_sizes:
                parsed_batch_sizes.append(batch_size)
        
        # Ensure min and max are included
        if min_batch not in parsed_batch_sizes:
            parsed_batch_sizes.insert(0, min_batch)
        if max_batch not in parsed_batch_sizes:
            parsed_batch_sizes.append(max_batch)
        
        parsed_batch_sizes.sort()

    # Run benchmarks
    results = {
        "timestamp": time.time(),
        "description": description,
        "tag": tag,
        "system_info": system_info,
        "config_summary": {
            "max_grid_height": config.dataset.max_grid_height,
            "max_grid_width": config.dataset.max_grid_width,
            "max_train_pairs": config.dataset.max_train_pairs,
            "selection_format": config.action.selection_format,
            "logging_enabled": config.logging.log_operations,
            "history_enabled": config.history.enabled,
        },
        "benchmark_params": {
            "single_steps": single_steps,
            "batch_steps": batch_steps,
            "custom_batch_sizes": parsed_batch_sizes,
        },
    }

    # Single environment benchmark
    try:
        single_results = benchmark_single_environment(config, task_data, num_steps=single_steps)
        results["single_environment"] = single_results
        logger.info(f"Single environment: {single_results['steps_per_second']:.0f} SPS")
    except Exception as e:
        logger.error(f"Single environment benchmark failed: {e}")
        results["single_environment"] = {"error": str(e)}

    # Batch environment benchmarks
    try:
        scaling_results = run_scaling_benchmark(config, task_data, parsed_batch_sizes, batch_steps)
        results["batch_scaling"] = scaling_results
    except Exception as e:
        logger.error(f"Batch scaling benchmark failed: {e}")
        results["batch_scaling"] = {"error": str(e)}

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create filename with optional tag
    filename_parts = ["benchmark", timestamp]
    if tag:
        filename_parts.insert(1, tag)
    filename = "_".join(filename_parts) + ".json"
    
    output_file = output_dir / filename

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Benchmark results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    summary_title = "BENCHMARK SUMMARY"
    if description:
        summary_title += f": {description}"
    print(summary_title)
    print("=" * 60)

    if (
        "single_environment" in results
        and "steps_per_second" in results["single_environment"]
    ):
        single_sps = results["single_environment"]["steps_per_second"]
        print(f"Single Environment: {single_sps:,.0f} SPS")

    if "batch_scaling" in results:
        print("\nBatch Environment Scaling:")
        for batch_size, result in results["batch_scaling"].items():
            if "steps_per_second" in result:
                sps = result["steps_per_second"]
                print(f"  Batch {batch_size:>4}: {sps:,.0f} SPS")

    print(f"\nResults saved to: {output_file}")
    if description:
        print(f"Description: {description}")
    if tag:
        print(f"Tag: {tag}")
    if parsed_batch_sizes:
        print(f"Batch sizes tested: {parsed_batch_sizes}")
    print("=" * 60)


if __name__ == "__main__":
    typer.run(main)
