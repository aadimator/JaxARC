"""Simple logging utilities for JAX environments.

Provides basic helpers for creating logging payloads from environment data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import numpy as np
from loguru import logger

try:
    from jaxarc.state import State
    from jaxarc.types import EnvParams, JaxArcTask
    from jaxarc.utils.task_manager import extract_task_id_from_index
except ImportError:
    # Handle missing imports gracefully
    State = Any
    EnvParams = Any
    JaxArcTask = Any
    extract_task_id_from_index = lambda x: f"task_{x}"


def to_python_int(x: Any) -> Optional[int]:
    """Convert scalar-like value to Python int or None."""
    if x is None:
        return None
    try:
        return int(np.asarray(x).item())
    except Exception:
        try:
            return int(x)
        except Exception:
            return None


def to_python_float(x: Any) -> Optional[float]:
    """Convert scalar-like value to Python float or None."""
    if x is None:
        return None
    try:
        return float(np.asarray(x).item())
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def to_python_scalar(x: Any) -> Any:
    """Convert JAX/numpy scalar to Python scalar, return as-is if conversion fails."""
    if x is None or isinstance(x, (int, float, bool, str)):
        return x

    try:
        arr = np.asarray(x)
        if arr.shape == ():
            return arr.item()
    except Exception:
        pass

    return x


def create_start_log(
    params: EnvParams,
    task_idx: Union[int, Any] = None,
    state: Optional[State] = None,
    episode_num: int = 0,
) -> Dict[str, Any]:
    """Extract task data for logging.

    Args:
        params: Environment parameters with task buffer
        task_idx: Task index (required if state not provided)
        state: Environment state containing task_idx
        episode_num: Episode number

    Returns:
        Dict with task data for logging
    """
    # Get task index
    if task_idx is None and state is not None:
        task_idx = state.task_idx
    if task_idx is None:
        raise ValueError("Either task_idx or state must be provided")

    idx = to_python_int(task_idx)
    if params is None or params.buffer is None:
        raise ValueError("EnvParams must contain a valid buffer")

    # Extract task from buffer
    try:
        single = jax.tree_util.tree_map(lambda x: x[idx], params.buffer)
        task_object = JaxArcTask(
            input_grids_examples=single.input_grids_examples,
            input_masks_examples=single.input_masks_examples,
            output_grids_examples=single.output_grids_examples,
            output_masks_examples=single.output_masks_examples,
            num_train_pairs=single.num_train_pairs,
            test_input_grids=single.test_input_grids,
            test_input_masks=single.test_input_masks,
            true_test_output_grids=single.true_test_output_grids,
            true_test_output_masks=single.true_test_output_masks,
            num_test_pairs=single.num_test_pairs,
            task_index=single.task_index,
        )

        # Get task ID
        try:
            task_id = task_object.get_task_id()
        except Exception:
            task_id = extract_task_id_from_index(idx)

    except Exception as e:
        logger.error(f"Failed to extract task data: {e}")
        raise ValueError(f"Could not extract task at index {idx}") from e

    return {
        "task_object": task_object,
        "task_idx": idx,
        "task_id": task_id,
        "num_train_pairs": to_python_int(single.num_train_pairs),
        "num_test_pairs": to_python_int(single.num_test_pairs),
        "episode_num": episode_num,
        "show_test": True,
    }


def create_step_log(
    timestep,
    action,
    step_num: int,
    episode_num: int,
    prev_state=None,
    env_params=None,
) -> Dict[str, Any]:
    """Create step logging payload.

    Args:
        timestep: TimeStep from environment
        action: Action taken
        step_num: Step number
        episode_num: Episode number
        prev_state: Previous state (optional)
        env_params: Environment parameters (optional)

    Returns:
        Dict for step logging
    """
    log_data = {
        "step_num": step_num,
        "episode_num": episode_num,
        "action": action,
        "reward": to_python_float(timestep.reward),
        "before_state": prev_state,
        "after_state": timestep.state,
        "params": env_params,
    }

    # Extract info
    info = {}
    if hasattr(timestep, "info") and timestep.info is not None:
        if isinstance(timestep.info, dict):
            info.update(timestep.info)

    # Add similarity metrics
    if timestep.state and hasattr(timestep.state, "similarity_score"):
        similarity = to_python_float(timestep.state.similarity_score)
        if similarity is not None:
            info["similarity"] = similarity

    # Calculate similarity improvement
    if (
        prev_state
        and hasattr(prev_state, "similarity_score")
        and timestep.state
        and hasattr(timestep.state, "similarity_score")
    ):
        prev_sim = to_python_float(prev_state.similarity_score) or 0.0
        curr_sim = to_python_float(timestep.state.similarity_score) or 0.0
        info["similarity_improvement"] = curr_sim - prev_sim

    log_data["info"] = info

    # Extract state metadata
    if timestep.state:
        if hasattr(timestep.state, "task_idx"):
            task_idx = to_python_int(timestep.state.task_idx)
            log_data["task_idx"] = task_idx
            try:
                log_data["task_id"] = extract_task_id_from_index(task_idx)
            except Exception:
                log_data["task_id"] = f"task_{task_idx}"

        if hasattr(timestep.state, "pair_idx"):
            log_data["task_pair_index"] = to_python_int(timestep.state.pair_idx)

        if hasattr(timestep.state, "step_count"):
            log_data["step_count"] = to_python_int(timestep.state.step_count)

    return log_data


def create_episode_summary(
    episode_num: int,
    step_logs: list[dict],
    env_params=None,
) -> Dict[str, Any]:
    """Create episode summary from step logs.

    Args:
        episode_num: Episode number
        step_logs: List of step log dicts
        env_params: Environment parameters (optional)

    Returns:
        Dict for episode summary logging
    """
    summary = {
        "episode_num": episode_num,
        "total_steps": len(step_logs),
        "step_data": step_logs,
        "params": env_params,
    }

    if not step_logs:
        summary.update(
            {
                "total_reward": 0.0,
                "final_similarity": 0.0,
                "reward_progression": [],
                "similarity_progression": [],
            }
        )
        return summary

    # Extract metrics from step logs
    rewards = []
    similarities = []

    for step in step_logs:
        reward = to_python_float(step.get("reward", 0.0)) or 0.0
        rewards.append(reward)

        # Get similarity from info or state
        similarity = 0.0
        info = step.get("info", {})
        if isinstance(info, dict) and "similarity" in info:
            similarity = to_python_float(info["similarity"]) or 0.0

        similarities.append(similarity)

    summary.update(
        {
            "total_reward": sum(rewards),
            "final_similarity": similarities[-1] if similarities else 0.0,
            "reward_progression": rewards,
            "similarity_progression": similarities,
        }
    )

    # Add task ID from first step
    if step_logs and "task_id" in step_logs[0]:
        summary["task_id"] = step_logs[0]["task_id"]

    return summary


__all__ = [
    "create_episode_summary",
    "create_start_log",
    "create_step_log",
    "to_python_float",
    "to_python_int",
    "to_python_scalar",
]
