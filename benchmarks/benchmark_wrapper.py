#!/usr/bin/env python3
"""Scan-based Wrapper vs Functional Benchmark for JaxARC

Fast, fused benchmark comparing functional arc_step vs a wrapper-equivalent
rollout using JAX lax.scan for high utilization and low host overhead.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import functools

import jax
import jax.numpy as jnp
import numpy as np
import typer
from loguru import logger

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
    StructuredAction
)
from jaxarc.envs.functional import (
    arc_reset,
    arc_step,
    batch_reset,
)
from jaxarc.envs.wrapper import ArcEnv
from jaxarc.types import JaxArcTask
from jaxarc.state import ArcEnvState
from jaxarc.utils.jax_types import EPISODE_MODE_TRAIN

# ---------------------------------------------------------------------------
# Config & Task Creation (mirrors baseline benchmark minimal config)
# ---------------------------------------------------------------------------

def create_minimal_config() -> JaxArcConfig:
    return JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=100, auto_reset=True),
        dataset=DatasetConfig(
            max_grid_height=30,
            max_grid_width=30,
            max_train_pairs=3,
            max_test_pairs=1,
            background_color=0,
            dataset_path="data/arc-prize-2024",
            task_split="train",
        ),
        action=ActionConfig(selection_format="point", dynamic_action_filtering=False),
        reward=RewardConfig(similarity_weight=1.0, step_penalty=-0.01, success_bonus=10.0),
        grid_initialization=GridInitializationConfig(mode="demo", demo_pair_selection="random"),
        visualization=VisualizationConfig.from_hydra({}),
        storage=StorageConfig(),
        logging=LoggingConfig(log_operations=False, log_level="ERROR"),
        wandb=WandbConfig.from_hydra({}),
        history=HistoryConfig(enabled=False),
    )


def create_dummy_task_data(config: JaxArcConfig) -> JaxArcTask:
    h = config.dataset.max_grid_height
    w = config.dataset.max_grid_width
    tp = config.dataset.max_train_pairs
    tt = config.dataset.max_test_pairs

    def checker(height: int, width: int, offset: int = 0) -> jnp.ndarray:
        # Simple pattern; stay in JAX for JIT friendliness
        rows = jnp.arange(height).reshape(height, 1)
        cols = jnp.arange(width).reshape(1, width)
        return ((rows + cols + offset) % 10).astype(jnp.int32)

    input_grids_examples = jnp.stack([checker(h, w, i) for i in range(tp)], 0)
    output_grids_examples = jnp.stack([checker(h, w, i + 1) for i in range(tp)], 0)
    input_masks_examples = jnp.zeros_like(input_grids_examples, dtype=jnp.bool_)
    output_masks_examples = jnp.zeros_like(output_grids_examples, dtype=jnp.bool_)

    test_input_grids = jnp.stack([checker(h, w, i + 5) for i in range(tt)], 0)
    test_input_masks = jnp.zeros_like(test_input_grids, dtype=jnp.bool_)
    true_test_output_grids = jnp.stack([checker(h, w, i + 6) for i in range(tt)], 0)
    true_test_output_masks = jnp.zeros_like(true_test_output_grids, dtype=jnp.bool_)

    return JaxArcTask(
        input_grids_examples=input_grids_examples,
        input_masks_examples=input_masks_examples,
        output_grids_examples=output_grids_examples,
        output_masks_examples=output_masks_examples,
        num_train_pairs=tp,
        test_input_grids=test_input_grids,
        test_input_masks=test_input_masks,
        true_test_output_grids=true_test_output_grids,
        true_test_output_masks=true_test_output_masks,
        num_test_pairs=tt,
        task_index=jnp.array(0, dtype=jnp.int32),
    )

# ---------------------------------------------------------------------------
# Action tensor generation
# ---------------------------------------------------------------------------

# Field layout: [action_type, op, r1, c1, r2, c2]
# point: action_type=0, r1=row, c1=col
# bbox:  action_type=1, r1, c1, r2, c2
ACTION_FIELDS = 6


def generate_action_tensor(config: JaxArcConfig, steps: int, pattern: str, seed: int = 0) -> jnp.ndarray:
    h = config.dataset.max_grid_height
    w = config.dataset.max_grid_width
    key = jax.random.PRNGKey(seed)
    
    key, op_key, type_key, p1_key, p2_key, p3_key, p4_key = jax.random.split(key, 7)
    
    ops = jax.random.randint(op_key, (steps,), 0, 35)
    
    if pattern == "point_only":
        action_types = jnp.zeros((steps,), dtype=jnp.int32)
        p1 = jax.random.randint(p1_key, (steps,), 0, h) # row
        p2 = jax.random.randint(p2_key, (steps,), 0, w) # col
        p3 = jnp.zeros_like(p1) # r2 (unused)
        p4 = jnp.zeros_like(p2) # c2 (unused)
    else:  # mixed
        action_types = jax.random.randint(type_key, (steps,), 0, 2) # 0=point, 1=bbox
        p1 = jax.random.randint(p1_key, (steps,), 0, h)
        p2 = jax.random.randint(p2_key, (steps,), 0, w)
        p3 = jax.random.randint(p3_key, (steps,), 0, h)
        p4 = jax.random.randint(p4_key, (steps,), 0, w)

    return jnp.stack([action_types, ops, p1, p2, p3, p4], axis=1)

# ---------------------------------------------------------------------------
# Rollout logic using scan
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    steps_per_second: float
    total_time: float
    jit_compile_time: float
    num_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _time_first_call(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    # Synchronize on the final array to ensure compilation is finished
    out.block_until_ready()
    return out, time.perf_counter() - start


def make_rollout_fn(config: JaxArcConfig):
    """Creates a JIT-compiled function that runs an entire episode using lax.scan."""
    
    def _step_from_encoded(state: ArcEnvState, encoded_action: jax.Array) -> ArcEnvState:
        action_type, op, p1, p2, p3, p4 = encoded_action
        
        # Decode action and step
        def point_step() -> ArcEnvState:
            action = create_point_action(op, p1, p2)
            return arc_step(state, action, config)[0]
            
        def bbox_step() -> ArcEnvState:
            r1 = jnp.minimum(p1, p3)
            c1 = jnp.minimum(p2, p4)
            r2 = jnp.maximum(p1, p3)
            c2 = jnp.maximum(p2, p4)
            action = create_bbox_action(op, r1, c1, r2, c2)
            return arc_step(state, action, config)[0]
            
        # Use lax.switch for efficient conditional execution on the device
        return jax.lax.switch(action_type, [point_step, bbox_step])

    def rollout_fn(initial_state: ArcEnvState, action_tensor: jax.Array) -> jax.Array:
        def scan_body(state: ArcEnvState, encoded_action: jax.Array) -> tuple[ArcEnvState, None]:
            next_state = _step_from_encoded(state, encoded_action)
            return next_state, None # No per-step output needed for timing
            
        final_state, _ = jax.lax.scan(scan_body, initial_state, action_tensor)
        return final_state.working_grid # Return a single array to sync on

    # Use functools.partial to "bake in" the config as a static argument
    return jax.jit(rollout_fn)


def benchmark_scan(
    config: JaxArcConfig,
    task_data: JaxArcTask,
    batch_size: int,
    steps: int,
    pattern: str,
) -> TimingResult:
    """Measures SPS for a scan-based rollout."""
    key = jax.random.PRNGKey(42)
    
    # 1. Create initial states
    if batch_size == 1:
        initial_state, _ = arc_reset(key, config, task_data)
    else:
        keys = jax.random.split(key, batch_size)
        initial_state, _ = batch_reset(keys, config, task_data)
        
    # 2. Create action tensor
    action_tensor = generate_action_tensor(config, steps, pattern)
    
    # 3. Create the JIT-compiled rollout function
    rollout_fn = make_rollout_fn(config)
    if batch_size > 1:
        # Vmap the single-env rollout function over the batch of states
        rollout_fn = jax.vmap(rollout_fn, in_axes=(0, None))
        
    # 4. Time compilation (first call)
    logger.info(f"Compiling for batch_size={batch_size}, pattern='{pattern}'...")
    _, compile_time = _time_first_call(rollout_fn, initial_state, action_tensor)
    logger.info(f"Compilation finished in {compile_time:.2f}s")
    
    # 5. Time execution (second call)
    start_time = time.perf_counter()
    final_grid = rollout_fn(initial_state, action_tensor)
    final_grid.block_until_ready()
    run_time = time.perf_counter() - start_time
    
    # 6. Calculate SPS
    total_env_steps = steps * batch_size
    sps = total_env_steps / run_time if run_time > 0 else float('inf')
    
    return TimingResult(
        steps_per_second=sps,
        total_time=run_time,
        jit_compile_time=compile_time,
        num_steps=steps,
    )

# ---------------------------------------------------------------------------
# CLI and Orchestration
# ---------------------------------------------------------------------------

def main(
    batch_sizes: str = typer.Option("1,10,100,1000,5000,10000,20000", "--batch-sizes"),
    num_steps: int = typer.Option(500, "--num-steps"),
    patterns: str = typer.Option("point_only,mixed", "--patterns"),
):
    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO")
    
    config = create_minimal_config()
    task_data = create_dummy_task_data(config)
    batch_sizes_list = [int(x.strip()) for x in batch_sizes.split(",")]
    patterns_list = [p.strip() for p in patterns.split(",")]

    print("\n" + "=" * 60)
    print("ArcEnv High-Performance Benchmark (Scan-Based)")
    print("=" * 60)

    results = {}
    for pattern in patterns_list:
        results[pattern] = {}
        print(f"\n--- Pattern: {pattern} ---")
        for bs in batch_sizes_list:
            sps_result = benchmark_scan(config, task_data, bs, num_steps, pattern)
            results[pattern][bs] = sps_result.to_dict()
            print(f"Batch Size: {bs:<6} | Steps Per Second: {sps_result.steps_per_second:,.2f}")

    print("=" * 60)
    print("Benchmark complete.")
    
    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"scan_benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    typer.run(main)
