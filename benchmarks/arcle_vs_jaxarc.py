#!/usr/bin/env python3
"""
ARCLE vs JaxARC Benchmark (KISS)

Two simple benchmarks inspired by NAVIX:
    1) Speed vs number of timesteps (single env)
    2) Throughput vs number of environments (batched)

Outputs a compact JSON with raw timings (seconds) for each run. Optional plotting.

Usage examples:
    pixi run -e bench python benchmarks/arcle_vs_jaxarc_simple.py --mode both
    pixi run -e bench python benchmarks/arcle_vs_jaxarc_simple.py --mode speed --timestep-powers 1,2,3,4 --runs 3
    pixi run -e bench python benchmarks/arcle_vs_jaxarc_simple.py --mode throughput --batch-powers 0,1,2,3,4 --fixed-steps 1000
"""

from __future__ import annotations

import contextlib
import json
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from arcle.loaders import MiniARCLoader
from gymnasium import spaces

from jaxarc.configs import JaxArcConfig
from jaxarc.envs.action_wrappers import BboxActionWrapper
from jaxarc.registration import make
from jaxarc.utils.core import get_config

app = typer.Typer(
    add_completion=False,
    help="ARCLE vs JaxARC benchmark (simple)",
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
        "matplotlib": getattr(plt, "__version__", "unknown"),
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


def run_jaxarc(env, params, num_steps: int, num_envs: int):
    """Return compiled JAX runner vmapped over envs (num_envs can be 1)."""
    action_space = env.action_space(params)

    def _single(key):
        ts = env.reset(params, key)
        keys = jax.random.split(key, num_steps)

        def body(ts, k):
            a = action_space.sample(k)
            ts = env.step(params, ts, a)
            return ts, ()

        ts, _ = jax.lax.scan(body, ts, keys, unroll=20)
        return ts

    def _batched(keys):
        # Expect shape (num_envs, 2) PRNGKey array; num_envs can be 1
        if hasattr(keys, "shape") and keys.shape[0] != num_envs:
            raise ValueError(f"Expected {num_envs} keys, got {keys.shape[0]}")
        return jax.vmap(_single)(keys)

    return jax.jit(_batched)


def time_runtime_only(fn, arg, repeats: int) -> list[float]:
    """Compile outside timing and measure runtime only (NAVIX-style)."""
    compiled = fn.lower(arg).compile()

    def _call():
        res = compiled(arg)
        leaves = jax.tree_util.tree_leaves(res)
        if leaves:
            _ = leaves[0].block_until_ready()

    return timeit.repeat(_call, number=1, repeat=repeats)


def setup_arcle_env(num_envs: int):
    """Create ARCLE RawARCEnv(s) for MiniARC with BBoxWrapper."""
    if num_envs == 1:
        env = gym.make(
            "ARCLE/RawARCEnv-v0", data_loader=MiniARCLoader(), max_grid_size=(5, 5)
        )
        return BBoxWrapper(env)

    def make_env():
        def _init():
            env = gym.make(
                "ARCLE/RawARCEnv-v0", data_loader=MiniARCLoader(), max_grid_size=(5, 5)
            )
            return BBoxWrapper(env)

        return _init

    return gym.vector.SyncVectorEnv([make_env() for _ in range(num_envs)])


def run_arcle(
    num_steps: int, num_envs: int, repeats: int, task_idx: int = 0
) -> list[float]:
    env = setup_arcle_env(num_envs)

    def _run():
        obs, _info = env.reset(options={"prob_index": task_idx})
        for _ in range(num_steps):
            actions = env.action_space.sample()
            obs, _, term, trunc, _info = env.step(actions)
            # VectorEnv returns arrays, handle either case
            terminated = np.any(term) if isinstance(term, np.ndarray) else term
            truncated = np.any(trunc) if isinstance(trunc, np.ndarray) else trunc
            if terminated or truncated:
                obs, _info = env.reset(options={"prob_index": task_idx})
        return obs

    times = timeit.repeat(_run, number=1, repeat=repeats)
    with contextlib.suppress(Exception):
        _run()
    env.close()
    return times


# --------------------------- Benchmarks ------------------------------------


@dataclass
class BenchOutput:
    # {config_value: {Framework: {"times": [...], "sps": [...]}}}
    speed: dict[int, dict[str, dict[str, list[float]]]]
    throughput: dict[int, dict[str, dict[str, list[float]]]]
    meta: dict[str, Any]


def benchmark_speed(
    timestep_powers: list[int], runs: int
) -> dict[int, dict[str, dict[str, list[float]]]]:
    results: dict[int, dict[str, dict[str, list[float]]]] = {}
    # JaxARC setup once
    task_id, task_idx = ensure_same_task()
    env, params = setup_jaxarc(task_id)

    for p in timestep_powers:
        steps = 10**p
        # JaxARC timing
        run_fn = run_jaxarc(env, params, steps, num_envs=1)
        keys = jax.random.split(jax.random.PRNGKey(0), 1)
        jax_times = time_runtime_only(run_fn, keys, runs)
        jax_sps = [steps / t for t in jax_times]

        # ARCLE timing (optional)
        arcle_times = run_arcle(steps, num_envs=1, repeats=runs, task_idx=task_idx)
        arcle_sps = [steps / t for t in arcle_times]
        results[steps] = {
            "JaxARC": {"times": jax_times, "sps": jax_sps},
            "ARCLE": {"times": arcle_times, "sps": arcle_sps},
        }
        print(
            f"Steps={steps}: JaxARC mean {np.mean(jax_times):.4f}s, ARCLE mean {np.mean(arcle_times):.4f}s"
        )
    return results


def benchmark_throughput(
    batch_powers: list[int], fixed_steps: int, runs: int
) -> dict[int, dict[str, dict[str, list[float]]]]:
    results: dict[int, dict[str, dict[str, list[float]]]] = {}
    task_id, task_idx = ensure_same_task()
    env, params = setup_jaxarc(task_id)

    for p in batch_powers:
        num_envs = 2**p
        run_fn = run_jaxarc(env, params, fixed_steps, num_envs)
        keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
        jax_times = time_runtime_only(run_fn, keys, runs)
        total_steps = fixed_steps * num_envs
        jax_sps = [total_steps / t for t in jax_times]

        arcle_times = run_arcle(fixed_steps, num_envs, runs, task_idx=task_idx)
        arcle_sps = [total_steps / t for t in arcle_times]
        results[num_envs] = {
            "JaxARC": {"times": jax_times, "sps": jax_sps},
            "ARCLE": {"times": arcle_times, "sps": arcle_sps},
        }
        print(
            f"Envs={num_envs}: JaxARC mean {np.mean(jax_times):.4f}s, ARCLE mean {np.mean(arcle_times):.4f}s"
        )
    return results


# --------------------------- Plotting (optional) ---------------------------


def plot_speed(
    results: dict[int, dict[str, dict[str, list[float]]]], outdir: Path
) -> Path | None:
    xs = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(6, 3), dpi=140)
    if any("ARCLE" in results[k] for k in xs):
        ys = jnp.asarray([results[k]["ARCLE"]["times"] for k in xs])
        ax.errorbar(
            xs,
            ys.mean(axis=-1),
            yerr=ys.std(axis=-1),
            label="ARCLE",
            color="black",
            marker="o",
        )
    ys = jnp.asarray([results[k]["JaxARC"]["times"] for k in xs])
    ax.errorbar(
        xs,
        ys.mean(axis=-1),
        yerr=ys.std(axis=-1),
        label="JaxARC",
        color="red",
        marker="s",
    )
    ax.set_title("Speed vs steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Time (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)
    ax.legend(loc="best")
    ensure_output_dir(outdir)
    path = outdir / "speed_vs_steps.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_throughput(
    results: dict[int, dict[str, dict[str, list[float]]]], outdir: Path
) -> Path | None:
    xs = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(6, 3), dpi=140)
    if any("ARCLE" in results[k] for k in xs):
        ys = jnp.asarray([results[k]["ARCLE"]["times"] for k in xs])
        ax.errorbar(
            xs,
            ys.mean(axis=-1),
            yerr=ys.std(axis=-1),
            label="ARCLE",
            color="black",
            marker="o",
        )
    ys = jnp.asarray([results[k]["JaxARC"]["times"] for k in xs])
    ax.errorbar(
        xs,
        ys.mean(axis=-1),
        yerr=ys.std(axis=-1),
        label="JaxARC",
        color="red",
        marker="s",
    )
    ax.set_title("Throughput vs envs")
    ax.set_xlabel("Num envs")
    ax.set_ylabel("Time (s)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)
    ax.legend(loc="best")
    ensure_output_dir(outdir)
    path = outdir / "throughput_vs_envs.png"
    fig.savefig(path, bbox_inches="tight")
    return path


# --------------------------- Main -----------------------------------------


@app.callback()
def main(
    mode: str = typer.Option(
        "both",
        "--mode",
        help="Benchmark mode: speed, throughput, or both",
        case_sensitive=False,
    ),
    timestep_powers: str = typer.Option(
        "1,2,3,4,5", help="Comma-separated powers of 10 for steps"
    ),
    batch_powers: str = typer.Option(
        "0,1,2,3,4,5,6,7", help="Comma-separated powers of 2 for envs"
    ),
    fixed_steps: int = typer.Option(
        1000, "--fixed-steps", help="Steps per env for throughput"
    ),
    runs: int = typer.Option(3, "--runs", help="Repeated runs per config"),
    # Task is fixed via ensure_same_task() for fair comparison
    output_dir: str = typer.Option(
        "benchmarks/results", "--output-dir", help="Directory to save JSON and plots"
    ),
    plot: bool = typer.Option(False, "--plot", help="Generate PNG plots"),
):
    outdir = Path(output_dir)
    ensure_output_dir(outdir)

    try:
        powers_steps = [int(x) for x in timestep_powers.split(",") if x.strip()]
        powers_envs = [int(x) for x in batch_powers.split(",") if x.strip()]
    except ValueError:
        typer.secho(
            "Invalid powers format. Use comma-separated integers.", fg=typer.colors.RED
        )
        raise typer.Exit(1) from None

    output: BenchOutput = BenchOutput(speed={}, throughput={}, meta={})
    output.meta = {
        "system": system_info(),
        "runs": runs,
        "timestep_powers": powers_steps,
        "batch_powers": powers_envs,
        "fixed_steps": fixed_steps,
        "task": {
            "task_id": ensure_same_task()[0],
            "task_idx": ensure_same_task()[1],
        },
    }

    mode = mode.lower()
    if mode in ("speed", "both"):
        print("Running speed vs steps…")
        output.speed = benchmark_speed(powers_steps, runs)

    if mode in ("throughput", "both"):
        print("Running throughput vs envs…")
        output.throughput = benchmark_throughput(
            powers_envs, fixed_steps, runs
        )

    # Save JSON
    json_path = outdir / "arcle_vs_jaxarc_results.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "meta": output.meta,
                "speed": output.speed,
                "throughput": output.throughput,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {json_path}")

    # Optional plots
    if plot:
        if output.speed:
            p = plot_speed(output.speed, outdir)
            if p:
                print(f"Saved plot: {p}")
        if output.throughput:
            p = plot_throughput(output.throughput, outdir)
            if p:
                print(f"Saved plot: {p}")


if __name__ == "__main__":
    app()
