"""
Reward calculation utilities for JaxARC environments.

This module contains JAX-friendly reward functions extracted from the
monolithic functional API to improve modularity and testability.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxarc.types import EnvParams
from jaxarc.state import State
from jaxarc.utils.jax_types import RewardValue


def _calculate_reward(
    old_state: State,
    new_state: State,
    params: EnvParams,
    *,
    is_submit_step: jnp.ndarray | None = None,
    episode_mode: int | None = None,
) -> RewardValue:
    """Submit-aware reward with optional episode mode selection.

    This implementation mirrors the previous logic while working with the new API:
    - Uses similarity improvement as shaped reward during training
    - Adds success bonus when solved (optionally only on submit)
    - Adds efficiency bonus for fast solutions
    - Applies step penalty on every step
    - Applies unsolved submission penalty when submitting without solving
    - Selects training vs evaluation composition via optional episode_mode

    Args:
        old_state: Previous environment state
        new_state: New environment state after action
        config: Environment configuration
        is_submit_step: Optional boolean array indicating if this step is a Submit action
        episode_mode: Optional episode mode (0=train, 1=test). When None, treated as train.

    Returns:
        JAX scalar array containing the calculated reward
    """
    reward_cfg = params.reward

    # Resolve optional flags with safe defaults
    submit_flag = (
        is_submit_step if is_submit_step is not None else jnp.asarray(False)
    )
    is_training = (
        jnp.asarray(True) if episode_mode is None else jnp.asarray(episode_mode == 0)
    )

    # 1) Components
    similarity_improvement = new_state.similarity_score - old_state.similarity_score
    is_solved = new_state.similarity_score >= 1.0

    step_penalty = jnp.asarray(reward_cfg.step_penalty, dtype=jnp.float32)
    similarity_reward = reward_cfg.similarity_weight * similarity_improvement

    success_bonus = jnp.where(is_solved, reward_cfg.success_bonus, 0.0)
    # Optionally award success bonus only on submit step
    success_bonus = jnp.where(
        reward_cfg.reward_on_submit_only,
        jnp.where(submit_flag, success_bonus, 0.0),
        success_bonus,
    )

    efficiency_bonus = jnp.where(
        is_solved & (new_state.step_count <= reward_cfg.efficiency_bonus_threshold),
        reward_cfg.efficiency_bonus,
        0.0,
    )

    # Penalty for submitting without solving
    submission_penalty = jnp.where(
        submit_flag & ~is_solved, reward_cfg.unsolved_submission_penalty, 0.0
    )

    # 2) Mode-specific totals (training includes similarity shaping)
    training_reward = (
        similarity_reward + step_penalty + success_bonus + efficiency_bonus + submission_penalty
    )
    evaluation_reward = step_penalty + success_bonus + efficiency_bonus + submission_penalty

    # 3) Select by mode
    return jnp.where(is_training, training_reward, evaluation_reward)
