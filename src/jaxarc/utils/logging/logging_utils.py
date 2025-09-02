"""
Host-side logging utilities for building enriched logging payloads.

These helpers are intended to be executed outside of JAX-transformed code (on the
host) and provide a consistent, best-effort way to construct the dictionaries
that logging handlers expect. The library avoids implicit discovery of large
objects (like `EnvParams.buffer`) inside logging code — callers should explicitly
pass `params` when task-level metadata is desired.

API:
- create_step_log(...) : Simple step logging with minimal parameters
- create_episode_summary(...) : Simple episode summary with automatic computation
- extract_task_for_logging(...) : Extract complete task data for logging

Usage Examples:


# Step logging - 50% fewer parameters
step_log = create_step_log(
    timestep=timestep,           # Contains state, reward, info
    action=action,
    step_num=step,
    episode_num=episode,
    prev_state=prev_state,       # optional for similarity improvement
)
experiment_logger.log_step(step_log)

# Episode summary - 62% fewer parameters
summary = create_episode_summary(
    episode_num=episode,
    step_logs=all_step_logs,     # Automatically computes totals
    env_params=env_params,       # optional for task metadata
)
experiment_logger.log_episode_summary(summary)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import numpy as np
from loguru import logger

# Import lightweight helpers from project (host-side safe)
from jaxarc.utils.task_manager import extract_task_id_from_index, get_task_id_globally

try:
    # Import types for type hints only; these imports should be fine on host side
    from jaxarc.state import State
    from jaxarc.types import EnvParams, JaxArcTask
except Exception:  # pragma: no cover - defensively handle import environments
    EnvParams = Any  # type: ignore
    State = Any  # type: ignore
    JaxArcTask = Any  # type: ignore


def _to_python_int(x: Any) -> Optional[int]:
    """Convert various scalar-like values (numpy, JAX, Python) to Python int.

    Returns None if conversion fails.
    """
    if x is None:
        return None
    try:
        # numpy scalars, python ints
        return int(np.asarray(x).item())
    except Exception:
        try:
            return int(x)
        except Exception:
            return None


def to_python_int(x: Any) -> Optional[int]:
    """Public helper: convert scalar-like value to Python int or return None.

    This is a small, well-tested helper intended for use across logging handlers
    and other host-side utilities to normalize scalar values coming from JAX /
    numpy / Python objects.
    """
    return _to_python_int(x)


def _to_python_float(x: Any) -> Optional[float]:
    """Convert various scalar-like values (numpy, JAX, Python) to Python float.

    Returns None if conversion fails.
    """
    if x is None:
        return None
    try:
        return float(np.asarray(x).item())
    except Exception:
        try:
            return float(x)
        except Exception:
            return None


def to_python_float(x: Any) -> Optional[float]:
    """Public helper: convert scalar-like value to Python float or return None."""
    return _to_python_float(x)


def to_python_scalar(x: Any) -> Optional[Any]:
    """Coerce a scalar-like value to a native Python scalar when possible.

    This function attempts to extract a Python scalar from JAX / numpy scalar
    wrappers, falling back to common coercions. It returns None on failure.

    Ordering:
      - Try numpy/JAX .item() extraction first (covers numpy and jax scalars).
      - Then try int(), float(), bool() coercions in that order.
      - Finally return the original value if it is already a Python scalar.
    """
    if x is None:
        return None

    # Fast path for common Python scalars
    if isinstance(x, (int, float, bool, str)):
        return x

    # Try numpy / JAX extraction
    try:
        arr = np.asarray(x)
        # Only attempt .item() on 0-d arrays
        if hasattr(arr, "shape") and arr.shape == ():
            return arr.item()
    except Exception:
        pass

    # Try primitive coercions as a last resort
    try:
        return int(x)
    except Exception:
        pass

    try:
        return float(x)
    except Exception:
        pass

    try:
        return bool(x)
    except Exception:
        pass

    # Give up
    return None


def create_start_log(
    params: EnvParams,
    task_idx: Union[int, Any] = None,
    state: Optional[State] = None,
    episode_num: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract complete task data for logging from EnvParams buffer.

    This function extracts the actual JaxArcTask object and metadata needed for
    comprehensive logging and visualization. It performs the same extraction logic
    as the environment's reset function to ensure consistency.

    Args:
        params: EnvParams containing the stacked task buffer
        task_idx: Task index to extract (optional if state is provided)
        state: Environment state containing task_idx (optional if task_idx provided)
        episode_num: Episode number to include in metadata (optional, defaults to 0)

    Returns:
        Dict containing:
            - task_object: Complete JaxArcTask object ready for visualization
            - task_idx: Task index (int)
            - task_id: Human-readable task identifier
            - num_train_pairs: Number of training pairs
            - num_test_pairs: Number of test pairs
            - episode_num: Episode number (from parameter or defaults to 0)
            - show_test: Whether to show test examples (defaults to True)

    Example:
        ```python
        # Using with state (most common)
        task_data = create_start_log(env_params, state=timestep.state)

        # Using with explicit task_idx
        task_data = create_start_log(env_params, task_idx=0)
        ```
    """
    # Determine task_idx from either parameter or state
    if task_idx is None and state is not None:
        task_idx = state.task_idx
    elif task_idx is None:
        raise ValueError("Either task_idx or state must be provided")

    # Convert to python int for indexing
    idx_int = _to_python_int(task_idx)

    if params is None or params.buffer is None:
        raise ValueError("EnvParams must contain a valid buffer for task extraction")

    # Extract single task from buffer (same logic as env.reset())
    single = jax.tree_util.tree_map(lambda x: x[idx_int], params.buffer)

    # Build the complete JaxArcTask object (same as env.reset())
    try:
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
    except Exception as e:
        logger.error(f"Failed to create JaxArcTask object: {e}")
        raise ValueError(
            f"Could not extract valid task data from buffer at index {idx_int}"
        ) from e

    # Get task ID
    try:
        task_id = task_object.get_task_id()
    except Exception:
        try:
            task_id = get_task_id_globally(_to_python_int(single.task_index))
        except Exception:
            task_id = f"task_{idx_int}"

    # Extract counts as Python ints
    num_train_pairs = _to_python_int(single.num_train_pairs)
    num_test_pairs = _to_python_int(single.num_test_pairs)

    return {
        "task_object": task_object,
        "task_idx": idx_int,
        "task_id": task_id,
        "num_train_pairs": num_train_pairs,
        "num_test_pairs": num_test_pairs,
        "episode_num": episode_num if episode_num is not None else 0,
        "show_test": True,  # Default to showing test examples
    }


def create_step_log(
    timestep,
    action,
    step_num: int,
    episode_num: int,
    prev_state=None,
    env_params=None,
) -> Dict[str, Any]:
    """
    Create a step logging payload with minimal parameters.

    This is a simplified, more idiomatic version of build_step_logging_payload.
    It takes the essential data and extracts everything else automatically.

    Args:
        timestep: The TimeStep object from the environment step
        action: The action that was taken
        step_num: Step number within the episode
        episode_num: Episode number
        prev_state: Optional previous state (for similarity improvement calculation)
        env_params: Optional environment parameters (for task metadata)

    Returns:
        Dictionary suitable for logging handlers

    Example:
        ```python
        step_log = create_step_log(
            timestep=timestep,
            action=action,
            step_num=step,
            episode_num=episode,
            prev_state=prev_state,
        )
        experiment_logger.log_step(step_log)
        ```
    """
    payload = {
        "step_num": int(step_num),
        "episode_num": int(episode_num),
        "before_state": prev_state,
        "after_state": timestep.state,
        "action": action,
        "reward": to_python_float(timestep.reward),
        "params": env_params,
    }

    # Extract info from timestep
    info_dict = {}
    if hasattr(timestep, "info") and timestep.info is not None:
        if isinstance(timestep.info, dict):
            info_dict = dict(timestep.info)
        else:
            # Extract common attributes from info object
            for attr in ("similarity", "similarity_improvement", "step_count"):
                if hasattr(timestep.info, attr):
                    info_dict[attr] = getattr(timestep.info, attr)

    # Add metrics from state
    metrics = {}
    if timestep.state is not None and hasattr(timestep.state, "similarity_score"):
        metrics["similarity"] = to_python_float(timestep.state.similarity_score)

    # Calculate similarity improvement if prev_state available
    if (
        prev_state is not None
        and hasattr(prev_state, "similarity_score")
        and hasattr(timestep.state, "similarity_score")
    ):
        try:
            prev_sim = to_python_float(prev_state.similarity_score) or 0.0
            curr_sim = to_python_float(timestep.state.similarity_score) or 0.0
            metrics["similarity_improvement"] = curr_sim - prev_sim
        except Exception:
            pass

    if metrics:
        info_dict["metrics"] = metrics

    payload["info"] = info_dict

    # Extract task metadata from state
    if timestep.state is not None:
        if hasattr(timestep.state, "task_idx"):
            payload["task_idx"] = to_python_int(timestep.state.task_idx)
        if hasattr(timestep.state, "pair_idx"):
            payload["task_pair_index"] = to_python_int(timestep.state.pair_idx)
            payload["current_pair_index"] = payload["task_pair_index"]
        if hasattr(timestep.state, "step_count"):
            payload["step_count"] = to_python_int(timestep.state.step_count)

    # Extract task ID if possible
    task_idx = payload.get("task_idx")
    if task_idx is not None:
        try:
            task_id = extract_task_id_from_index(task_idx)
            payload["task_id"] = task_id or f"task_{task_idx}"
        except Exception:
            payload["task_id"] = f"task_{task_idx}"

    return payload


def create_episode_summary(
    episode_num: int,
    step_logs: list[dict],
    env_params=None,
) -> Dict[str, Any]:
    """
    Create an episode summary with minimal parameters.

    This is a simplified, more idiomatic version of build_episode_summary_payload.
    It computes all metrics from the provided step logs.

    Args:
        episode_num: Episode number
        step_logs: List of step log dictionaries (from create_step_log)
        env_params: Optional environment parameters for task metadata

    Returns:
        Dictionary suitable for episode summary logging

    Example:
        ```python
        summary = create_episode_summary(
            episode_num=0, step_logs=episode_steps, env_params=env_params
        )
        experiment_logger.log_episode_summary(summary)
        ```
    """
    payload = {
        "episode_num": int(episode_num),
        "step_data": step_logs,
        "total_steps": len(step_logs),
        "params": env_params,
    }

    if not step_logs:
        # Empty episode
        payload.update(
            {
                "total_reward": 0.0,
                "final_similarity": 0.0,
                "reward_progression": [0.0, 0.0],
                "similarity_progression": [0.0, 0.0],
                "key_moments": [],
            }
        )
        return payload

    # Calculate totals from step logs
    rewards = []
    similarities = []

    for step in step_logs:
        # Extract reward
        reward = step.get("reward", 0.0)
        rewards.append(to_python_float(reward) or 0.0)

        # Extract similarity
        similarity = None
        info = step.get("info", {})
        if isinstance(info, dict) and "metrics" in info:
            metrics = info["metrics"]
            if isinstance(metrics, dict):
                similarity = metrics.get("similarity")

        # Fallback to after_state similarity if available
        if similarity is None:
            after_state = step.get("after_state")
            if after_state and hasattr(after_state, "similarity_score"):
                similarity = after_state.similarity_score

        similarities.append(to_python_float(similarity) or 0.0)

    # Fill forward similarities for missing values
    filled_similarities = []
    last_similarity = 0.0
    for sim in similarities:
        if sim is not None and sim != 0.0:
            last_similarity = sim
        filled_similarities.append(last_similarity)

    # Ensure minimum length for visualization
    if len(rewards) < 2:
        rewards = rewards + [rewards[-1] if rewards else 0.0] * (2 - len(rewards))
    if len(filled_similarities) < 2:
        filled_similarities = filled_similarities + [
            filled_similarities[-1] if filled_similarities else 0.0
        ] * (2 - len(filled_similarities))

    payload.update(
        {
            "total_reward": sum(rewards),
            "final_similarity": filled_similarities[-1] if filled_similarities else 0.0,
            "reward_progression": rewards,
            "similarity_progression": filled_similarities,
        }
    )

    # Find key moments (large reward changes)
    key_moments = []
    if len(rewards) > 1:
        deltas = [abs(rewards[i] - rewards[i - 1]) for i in range(1, len(rewards))]
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            threshold = avg_delta * 1.5
            for i, delta in enumerate(deltas, 1):
                if delta >= threshold:
                    key_moments.append(i)

    payload["key_moments"] = key_moments

    # Extract task_id from first step if available
    if step_logs and isinstance(step_logs[0], dict):
        payload["task_id"] = step_logs[0].get("task_id")

    return payload


# Expose a small API tuple for convenience
__all__ = [
    # New simplified API
    "create_step_log",
    "create_episode_summary",
    "create_start_log",
    # Exported scalar coercion helpers for reuse across logging handlers
    "to_python_int",
    "to_python_float",
    "to_python_scalar",
]
