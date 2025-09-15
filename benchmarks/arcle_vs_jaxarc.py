#!/usr/bin/env python3
"""
ARCLE vs JaxARC Throughput Benchmark

This script produces a compact JSON with timings (seconds) and SPS (steps/sec)
for repeated runs per configuration.

Usage examples:
    pixi run -e bench python benchmarks/arcle_vs_jaxarc.py --batch-powers 0,1,2,3,4 --fixed-steps 1000 --runs 3
    pixi run -e bench python benchmarks/arcle_vs_jaxarc.py --parallel pmap --batch-powers 0,1,2,3 --fixed-steps 2048
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import timeit
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import numpy as np
import typer
from arcle.loaders import MiniARCLoader
from gymnasium import spaces

from jaxarc.configs import JaxArcConfig
from jaxarc.envs.action_wrappers import BboxActionWrapper
from jaxarc.registration import make
from jaxarc.utils.core import get_config

# Enable partitionable PRNG for better multi-device behavior
jax.config.update("jax_threefry_partitionable", True)

app = typer.Typer(
    add_completion=False,
    help="ARCLE vs JaxARC throughput benchmark",
    invoke_without_command=True,
)


# --------------------------- Utilities -------------------------------------


# BBox wrapper for ARCLE to mirror JaxARC's BboxActionWrapper
class BBoxWrapper(gym.ActionWrapper):
    """Convert bbox tuple (x1,y1,x2,y2,op) into ARCLE dict action.

    Action space: Tuple(H, W, H, W, num_ops)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.unwrapped.H),
                spaces.Discrete(self.unwrapped.W),
                spaces.Discrete(self.unwrapped.H),
                spaces.Discrete(self.unwrapped.W),
                spaces.Discrete(len(self.unwrapped.operations)),
            )
        )

    def action(self, action: tuple):
        x1, y1, x2, y2, op = action
        selection = np.zeros((self.unwrapped.H, self.unwrapped.W), dtype=np.int8)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        selection[x1 : x2 + 1, y1 : y2 + 1] = 1
        return {"selection": selection, "operation": op}


def system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split(" ")[0],
        "platform": sys.platform,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "gymnasium": getattr(gym, "__version__", "unknown"),
    }
    # JAX devices
    with contextlib.suppress(Exception):  # pragma: no cover
        devs = jax.devices()
        info["devices"] = [f"{d.platform}:{d.id}" for d in devs]
    return info


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_same_task() -> tuple[str, int]:
    """Ensure both frameworks use equivalent tasks (fixed id/index)."""
    task_id = "Most_Common_color_l6ab0lf3xztbyxsu3p"
    task_idx = 12
    return task_id, task_idx


def setup_jaxarc(task_id: str):
    """Create JaxARC env + params for a fixed MiniARC task; wrap with bbox actions."""
    overrides = [
        "dataset=mini_arc",
        "action=raw",
        "visualization.enabled=false",
        "logging.log_operations=false",
        "logging.log_rewards=false",
        "wandb.enabled=false",
    ]
    cfg = JaxArcConfig.from_hydra(get_config(overrides=overrides))
    env, params = make(f"Mini-{task_id}", config=cfg, auto_download=False)
    env = BboxActionWrapper(env)
    return env, params


def run_jaxarc(
    env,
    params,
    num_steps: int,
    num_envs: int,
    parallel: str = "jit",
    unroll: int = 1,
):
    """Return compiled JAX runner for throughput benchmarking.

    parallel:
        - "jit": single-device, vectorize envs via vmap
        - "pmap": multi-device, shard envs across devices then vmap within device
    """
    action_space = env.action_space(params)

    def _single(key):
        # Split key so initial reset and per-step RNG use disjoint streams
        reset_key, loop_key = jax.random.split(key)
        ts0 = env.reset(params, reset_key)

        def body(carry, _):
            ts, k = carry
            # Split for potential reset and action sampling
            k_reset, k_action, k_next = jax.random.split(k, 3)

            # Conditionally reset if previous timestep was terminal or truncated
            def do_reset(_ts):
                return env.reset(params, k_reset)

            def keep(_ts):
                return _ts

            ts = jax.lax.cond(ts.last(), do_reset, keep, ts)

            # Sample a fresh action and step
            act = action_space.sample(k_action)
            new_ts = env.step(params, ts, act)
            return (new_ts, k_next), ()

        # Scan over a dummy sequence of given length; carry TimeStep and RNG key
        (ts_final, _), _ = jax.lax.scan(body, (ts0, loop_key), xs=None, length=num_steps, unroll=unroll)
        return ts_final

    if parallel.lower() == "pmap":
        ndev = jax.local_device_count()
        if ndev < 1:
            msg = "No local devices available for pmap"
            raise RuntimeError(msg)
        if num_envs % ndev != 0:
            msg = f"num_envs={num_envs} must be divisible by number of devices {ndev} when using pmap"
            raise ValueError(msg)

        def _per_device(keys):  # keys shape (per_dev_envs, 2)
            return jax.vmap(_single)(keys)

        return jax.pmap(_per_device)

    def _batched(keys):  # keys shape (num_envs, 2)
        if hasattr(keys, "shape") and keys.shape[0] != num_envs:
            msg = f"Expected {num_envs} keys, got {keys.shape[0]}"
            raise ValueError(msg)
        return jax.vmap(_single)(keys)

    return jax.jit(_batched)


def time_compile_and_runtime(fn, arg, repeats: int) -> tuple[float, list[float]]:
    """Measure JAX compile time once, then runtime-only for repeated executes."""
    t0 = time.perf_counter()
    compiled = fn.lower(arg).compile()
    compile_time = time.perf_counter() - t0

    # Warm-up run to avoid inflated first runtime due to device/context init
    try:
        print("Running a warm-up execution…")
        res_warm = compiled(arg)
        leaves_warm = jax.tree_util.tree_leaves(res_warm)
        if leaves_warm:
            _ = leaves_warm[0].block_until_ready()
        print("Warm-up complete.")
    except Exception as e:  # pragma: no cover
        print(f"Warm-up failed (continuing without warm-up): {e}")

    def _call():
        res = compiled(arg)
        leaves = jax.tree_util.tree_leaves(res)
        if leaves:
            _ = leaves[0].block_until_ready()

    runtime_times = timeit.repeat(_call, number=1, repeat=repeats)
    return compile_time, runtime_times


# --------------------------- ARCLE Helpers ---------------------------------


def setup_arcle_env(num_envs: int, mode: str = "sync"):
    """Create ARCLE environments with different vectorization strategies.

    Args:
        num_envs: Number of environments
        mode: "sync" (SyncVectorEnv) or "async" (AsyncVectorEnv)
    """

    def make_env():
        def _init():
            env = gym.make(
                "ARCLE/RawARCEnv-v0", data_loader=MiniARCLoader(), max_grid_size=(5, 5)
            )
            return BBoxWrapper(env)

        return _init

    if num_envs == 1:
        # Single environment (no vectorization)
        return make_env()()

    if mode == "sync":
        # Sequential execution (SyncVectorEnv)
        return gym.vector.SyncVectorEnv([make_env() for _ in range(num_envs)])

    if mode == "async":
        # Multiprocessing parallelization (AsyncVectorEnv)
        return gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    raise ValueError(f"Unknown ARCLE mode: {mode}. Use 'sync' or 'async'")


def run_arcle(
    num_steps: int, num_envs: int, repeats: int, task_idx: int = 0, mode: str = "sync"
) -> list[float]:
    """Run ARCLE benchmark with specified vectorization mode."""
    env = setup_arcle_env(num_envs, mode=mode)

    def _run():
        if num_envs == 1:
            # Single environment
            obs, _info = env.reset(options={"prob_index": task_idx})
            for _ in range(num_steps):
                action = env.action_space.sample()
                obs, _, term, trunc, _info = env.step(action)
                if term or trunc:
                    obs, _info = env.reset(options={"prob_index": task_idx})
        else:
            # Vector environment - use single options dict for all envs
            obs, _info = env.reset(options={"prob_index": task_idx})
            for _ in range(num_steps):
                actions = env.action_space.sample()
                # Vectorized envs auto-reset individual envs; no manual reset needed
                obs, _, _term, _trunc, _info = env.step(actions)
        return obs

    times = timeit.repeat(_run, number=1, repeat=repeats)
    with contextlib.suppress(Exception):
        _run()
    env.close()
    return times


def benchmark_throughput(
    batch_powers: list[int], fixed_steps: int, runs: int, modes: list[str], unroll: int
) -> dict[int, dict[str, dict[str, Any]]]:
    """Benchmark throughput for specified modes.

    Args:
        batch_powers: Powers of 2 for number of environments
        fixed_steps: Steps per environment
        runs: Number of repeated runs
        modes: List of modes to benchmark (arcle-sync, arcle-async, jaxarc-jit, jaxarc-pmap)
        unroll: Scan unroll factor
    """
    results: dict[int, dict[str, dict[str, Any]]] = {}
    task_id, task_idx = ensure_same_task()

    # Setup JaxARC once if needed
    jaxarc_env, jaxarc_params = None, None
    if any(mode.startswith("jaxarc") for mode in modes):
        jaxarc_env, jaxarc_params = setup_jaxarc(task_id)

    for p in batch_powers:
        num_envs = 2**p
        results[num_envs] = {}

        for mode in modes:
            if mode.startswith("jaxarc"):
                # JaxARC benchmarking
                parallel = "pmap" if mode == "jaxarc-pmap" else "jit"
                run_fn = run_jaxarc(
                    jaxarc_env,
                    jaxarc_params,
                    fixed_steps,
                    num_envs,
                    parallel=parallel,
                    unroll=unroll,
                )

                # Prepare PRNG keys according to parallelization strategy
                if parallel == "pmap":
                    ndev = jax.local_device_count()
                    if num_envs % ndev != 0:
                        msg = f"num_envs={num_envs} must be divisible by number of devices {ndev} for pmap"
                        raise ValueError(msg)
                    per_dev_envs = num_envs // ndev
                    flat_keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
                    keys = flat_keys.reshape(ndev, per_dev_envs, 2)
                else:
                    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)

                try:
                    compile_time, times = time_compile_and_runtime(run_fn, keys, runs)
                    total_times = [compile_time + t for t in times]
                    total_steps = fixed_steps * num_envs
                    sps = [total_steps / t for t in times]

                    results[num_envs][mode] = {
                        "times": times,
                        "sps": sps,
                        "compile_time": compile_time,
                        "total_times": total_times,
                    }
                except Exception as e:
                    print(f"An error occurred during JaxARC benchmarking for {num_envs} envs: {e}")
                    results[num_envs][mode] = {
                        "error": str(e),
                        "times": [],
                        "sps": [],
                        "compile_time": 0.0,
                        "total_times": [],
                    }

            elif mode.startswith("arcle"):
                # ARCLE benchmarking
                arcle_mode = "async" if mode == "arcle-async" else "sync"
                try:
                    times = run_arcle(
                        fixed_steps, num_envs, runs, task_idx=task_idx, mode=arcle_mode
                    )
                    total_times = list(times)
                    total_steps = fixed_steps * num_envs
                    sps = [total_steps / t for t in times]

                    results[num_envs][mode] = {
                        "times": times,
                        "sps": sps,
                        "compile_time": 0.0,
                        "total_times": total_times,
                    }
                except Exception as e:
                    print(f"An error occurred during ARCLE benchmarking for {num_envs} envs: {e}")
                    results[num_envs][mode] = {
                        "error": str(e),
                        "times": [],
                        "sps": [],
                        "compile_time": 0.0,
                        "total_times": [],
                    }

            print(
                f"Envs={num_envs}, Mode={mode}: mean {np.mean(results[num_envs][mode]['times']):.4f}s"
            )

    return results


# --------------------------- Main -----------------------------------------


@app.callback()
def main(
    batch_powers: str = typer.Option(
        "0,1,2,3,4,5,6,7", help="Comma-separated powers of 2 for envs"
    ),
    fixed_steps: int = typer.Option(
        1000, "--fixed-steps", help="Steps per env for throughput"
    ),
    runs: int = typer.Option(3, "--runs", help="Repeated runs per config"),
    modes: str = typer.Option(
        "all",
        "--modes",
        help="Benchmark modes: all, arcle-sync, arcle-async, jaxarc-jit, jaxarc-pmap, or comma-separated list",
    ),
    unroll: int = typer.Option(
        1, "--unroll", help="Scan unroll factor (can affect performance)", min=1
    ),
    run_tag: str = typer.Option(
        "",
        "--run-tag",
        help="Custom tag for this run (e.g., cpu,a100,8xA100,tpus)",
        case_sensitive=False,
    ),
    output_dir: str = typer.Option(
        "benchmarks/results", "--output-dir", help="Directory to save JSON"
    ),
):
    outdir = Path(output_dir)
    ensure_output_dir(outdir)

    try:
        powers_envs = [int(x) for x in batch_powers.split(",") if x.strip()]
    except ValueError:
        typer.secho(
            "Invalid powers format. Use comma-separated integers.", fg=typer.colors.RED
        )
        raise typer.Exit(1) from None

    # Parse modes
    available_modes = ["arcle-sync", "arcle-async", "jaxarc-jit", "jaxarc-pmap"]
    if modes.lower() == "all":
        selected_modes = available_modes
    else:
        selected_modes = [m.strip() for m in modes.split(",") if m.strip()]
        invalid_modes = [m for m in selected_modes if m not in available_modes]
        if invalid_modes:
            typer.secho(
                f"Invalid modes: {invalid_modes}. Available modes: {available_modes}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1) from None

    # Validate pmap mode if selected
    if "jaxarc-pmap" in selected_modes:
        ndev = jax.local_device_count()
        if ndev < 2:
            typer.secho(
                f"Warning: jaxarc-pmap requires multiple devices but only {ndev} found. "
                f"Consider using jaxarc-jit instead.",
                fg=typer.colors.YELLOW,
            )

    # Metadata
    meta: dict[str, Any] = {
        "system": system_info(),
        "runs": runs,
        "batch_powers": powers_envs,
        "fixed_steps": fixed_steps,
        "modes": selected_modes,
        "num_devices": jax.local_device_count(),
        "unroll": unroll,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "run_tag": ("" if run_tag is None else run_tag),
        "task": {
            "task_id": ensure_same_task()[0],
            "task_idx": ensure_same_task()[1],
        },
    }

    print(f"Running throughput benchmark with modes: {selected_modes}")

    # Run benchmarks and save incrementally per mode and batch size
    for mode in selected_modes:
        print(f"\nBenchmarking mode: {mode}")

        # Generate filename with mode and tag (early)
        ts = meta["timestamp"]
        tag_raw = meta["run_tag"] or ""
        safe_tag = "".join(
            c if (c.isalnum() or c in ("-", "_")) else "-" for c in tag_raw.strip()
        )

        # Build filename: arcle_vs_jaxarc_{tag}_{mode}_{timestamp}.json
        base = "arcle_vs_jaxarc"
        if safe_tag:
            fname = f"{base}_{safe_tag}_{mode}_{ts}.json"
        else:
            fname = f"{base}_{mode}_{ts}.json"

        json_path = outdir / fname

        # Load existing results for resume if present
        results_for_mode: dict[str, Any] = {"meta": meta.copy(), "throughput": {}}
        if json_path.exists():
            print(f"Resuming from existing file: {json_path}")
            try:
                with json_path.open("r") as f:
                    results_for_mode = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load existing results ({e}), starting fresh.")

        # Ensure meta is correct for this mode
        results_for_mode["meta"]["mode"] = mode
        results_for_mode["meta"]["modes"] = [mode]

        for p in powers_envs:
            num_envs = 2**p
            key_num_envs = str(num_envs)

            # Skip if already present (resume support)
            if key_num_envs in results_for_mode.get("throughput", {}):
                print(f"Skipping Envs={num_envs}, Mode={mode} (already completed).")
                continue

            print(f"Running Envs={num_envs}, Mode={mode}…")

            # Run a single batch power via benchmark_throughput
            throughput = benchmark_throughput([p], fixed_steps, runs, [mode], unroll)

            # Merge the one-result data into our incremental dict
            if throughput and num_envs in throughput and mode in throughput[num_envs]:
                results_for_mode.setdefault("throughput", {})[key_num_envs] = throughput[num_envs][mode]
            else:
                # Store an explicit error/empty record to show attempted run
                results_for_mode.setdefault("throughput", {})[key_num_envs] = {
                    "error": "No data returned",
                    "times": [],
                    "sps": [],
                    "compile_time": 0.0,
                    "total_times": [],
                }

            # Save incrementally after each batch size
            try:
                with json_path.open("w") as f:
                    json.dump(results_for_mode, f, indent=2)
                print(f"Saved intermediate results for Envs={num_envs} to {json_path}")
            except Exception as e:
                print(f"Error saving results for Envs={num_envs}: {e}")

    print(f"\nCompleted benchmarking {len(selected_modes)} modes")


if __name__ == "__main__":
    app()
