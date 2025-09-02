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

import jax
import jax.numpy as jnp
import typer
from chex import PRNGKey
from loguru import logger

from jaxarc.configs import JaxArcConfig
from jaxarc.configs.action_config import ActionConfig
from jaxarc.configs.dataset_config import DatasetConfig
from jaxarc.configs.environment_config import EnvironmentConfig
from jaxarc.configs.grid_initialization_config import GridInitializationConfig
from jaxarc.configs.logging_config import LoggingConfig
from jaxarc.configs.reward_config import RewardConfig
from jaxarc.configs.storage_config import StorageConfig
from jaxarc.configs.visualization_config import VisualizationConfig
from jaxarc.configs.wandb_config import WandbConfig
from jaxarc.envs.actions import (
    MaskAction,
    create_mask_action,
)
from jaxarc.envs.functional import (
    arc_reset,
    arc_step,
    batch_reset,
)
from jaxarc.envs.wrapper import ArcEnv
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask

# ---------------------------------------------------------------------------
# Config & Task Creation
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
        reward=RewardConfig(
            similarity_weight=1.0, step_penalty=-0.01, success_bonus=10.0
        ),
        grid_initialization=GridInitializationConfig(
            mode="demo", demo_pair_selection="random"
        ),
        visualization=VisualizationConfig.from_hydra({}),
        storage=StorageConfig(),
        logging=LoggingConfig(log_operations=False, log_level="ERROR"),
        wandb=WandbConfig.from_hydra({}),
    )


def create_dummy_task_data(config: JaxArcConfig) -> JaxArcTask:
    h = config.dataset.max_grid_height
    w = config.dataset.max_grid_width
    tp = config.dataset.max_train_pairs
    tt = config.dataset.max_test_pairs

    def checker(height: int, width: int, offset: int = 0) -> jnp.ndarray:
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
# Shared Benchmark Utilities
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
    out.block_until_ready()
    return out, time.perf_counter() - start


def random_agent_policy(
    state: ArcEnvState, key: jax.Array, config: JaxArcConfig
) -> MaskAction:
    """Shared random policy for both benchmarks."""
    del state  # Unused for random policy
    h, w = config.dataset.max_grid_height, config.dataset.max_grid_width
    k1, k2, k3 = jax.random.split(key, 3)
    op = jax.random.randint(k1, (), 0, 35)
    r = jax.random.randint(k2, (), 0, h)
    c = jax.random.randint(k3, (), 0, w)

    # Create point mask action
    mask = jnp.zeros((h, w), dtype=jnp.bool_)
    mask = mask.at[r, c].set(True)
    return create_mask_action(op, mask)


# ---------------------------------------------------------------------------
# Functional API Benchmark (Apples-to-Apples)
# ---------------------------------------------------------------------------


def make_rollout_fn(config: JaxArcConfig, steps: int):
    """Creates a pure rollout function for the functional API."""

    def rollout_fn(initial_state: ArcEnvState, key: PRNGKey) -> ArcEnvState:
        def scan_body(carry, _):
            state, key = carry
            key, policy_key = jax.random.split(key)
            action = random_agent_policy(state, policy_key, config)
            next_state, _, _, _, _ = arc_step(state, action, config)
            return (next_state, key), None

        (final_state, _), _ = jax.lax.scan(
            scan_body, (initial_state, key), None, length=steps
        )
        return final_state

    return rollout_fn


def benchmark_scan(
    config: JaxArcConfig, task_data: JaxArcTask, batch_size: int, steps: int
) -> TimingResult:
    """Measures SPS for a scan-based rollout (functional API) with on-the-fly actions."""
    key = jax.random.PRNGKey(42)

    # 1. Create initial states and keys
    if batch_size == 1:
        initial_state, _ = arc_reset(key, config, task_data)
        keys = key
    else:
        keys = jax.random.split(key, batch_size)
        initial_state, _ = batch_reset(keys, config, task_data)

    # 2. Create the rollout function
    rollout_fn = make_rollout_fn(config, steps)
    if batch_size > 1:
        rollout_fn = jax.vmap(rollout_fn, in_axes=(0, 0))

    # 3. JIT the final function
    jitted_rollout = jax.jit(lambda state, k: rollout_fn(state, k).working_grid)

    # 4. Time compilation
    logger.info(f"Compiling functional for batch_size={batch_size}...")
    _, compile_time = _time_first_call(jitted_rollout, initial_state, keys)
    logger.info(f"Compilation finished in {compile_time:.2f}s")

    # 5. Time execution
    start_time = time.perf_counter()
    final_grid = jitted_rollout(initial_state, keys)
    final_grid.block_until_ready()
    run_time = time.perf_counter() - start_time

    # 6. Calculate SPS
    total_env_steps = steps * batch_size
    sps = total_env_steps / run_time if run_time > 0 else float("inf")
    return TimingResult(sps, run_time, compile_time, steps)


# ---------------------------------------------------------------------------
# Wrapper-based benchmark
# ---------------------------------------------------------------------------


def benchmark_wrapper_scan(
    config: JaxArcConfig,
    task_data: JaxArcTask,
    batch_size: int,
    steps: int,
) -> TimingResult:
    """Measures SPS for the ArcEnv wrapper using a proper vmap(scan(...)) rollout."""
    env = ArcEnv(
        config, num_envs=batch_size, task_data=task_data, seed=42, auto_reset=False
    )
    key = jax.random.PRNGKey(0)

    # 1. Get the pure, vmapped rollout function from the environment
    rollout_fn = env.get_rollout_fn(random_agent_policy, steps)
    jitted_rollout = jax.jit(lambda state, key: rollout_fn(state, key).working_grid)

    # 2. Obtain initial state and keys
    initial_state, _ = env.reset(key)
    if batch_size > 1:
        keys = jax.random.split(key, batch_size)
    else:
        keys = key

    # 3. Time compilation
    logger.info(f"Compiling wrapper for batch_size={batch_size}...")
    _, compile_time = _time_first_call(jitted_rollout, initial_state, keys)
    logger.info(f"Compilation finished in {compile_time:.2f}s")

    # 4. Time execution
    start_time = time.perf_counter()
    final_grid = jitted_rollout(initial_state, keys)
    final_grid.block_until_ready()
    run_time = time.perf_counter() - start_time

    # 5. Calculate SPS
    total_env_steps = steps * batch_size
    sps = total_env_steps / run_time if run_time > 0 else float("inf")
    return TimingResult(sps, run_time, compile_time, steps)


# ---------------------------------------------------------------------------
# CLI and Orchestration
# ---------------------------------------------------------------------------


def main(
    batch_sizes: str = typer.Option("1,10,100,1000,5000,10000,20000", "--batch-sizes"),
    num_steps: int = typer.Option(500, "--num-steps"),
):
    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO")

    config = create_minimal_config()
    task_data = create_dummy_task_data(config)
    batch_sizes_list = [int(x.strip()) for x in batch_sizes.split(",")]

    print("\n" + "=" * 60)
    print("ArcEnv High-Performance Benchmark (Apples-to-Apples)")
    print("=" * 60)

    results = {"functional_scan": {}, "wrapper_scan": {}}

    print("\n--- Functional Benchmark ---")
    for bs in batch_sizes_list:
        sps_result = benchmark_scan(config, task_data, bs, num_steps)
        results["functional_scan"][bs] = sps_result.to_dict()
        print(
            f"[functional] Batch Size: {bs:<6} | SPS: {sps_result.steps_per_second:,.2f}"
        )

    print("\n--- Wrapper Benchmark ---")
    for bs in batch_sizes_list:
        sps_result = benchmark_wrapper_scan(config, task_data, bs, num_steps)
        results["wrapper_scan"][bs] = sps_result.to_dict()
        print(
            f"[wrapper]    Batch Size: {bs:<6} | SPS: {sps_result.steps_per_second:,.2f}"
        )

    print("=" * 60)
    print("Benchmark complete.")

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"scan_benchmark_{timestamp}.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    typer.run(main)
