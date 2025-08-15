"""
Reward calculation utilities for JaxARC environments.

This module contains JAX-friendly reward functions extracted from the
monolithic functional API to improve modularity and testability.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxarc.configs import JaxArcConfig
from jaxarc.state import ArcEnvState
from jaxarc.utils.jax_types import RewardValue


def _calculate_reward(
    old_state: ArcEnvState, new_state: ArcEnvState, config: JaxArcConfig
) -> RewardValue:
    """Calculate reward with a unified, compiler-friendly structure.

    Control operations were removed; reward depends only on transition dynamics
    and episode mode.

    Args:
        old_state: Previous environment state
        new_state: New environment state after action
        config: Environment configuration

    Returns:
        JAX scalar array containing the calculated reward
    """
    reward_cfg = config.reward

    # 1) Components
    similarity_improvement = new_state.similarity_score - old_state.similarity_score
    is_solved = new_state.similarity_score >= 1.0
    is_training = new_state.episode_mode == 0

    is_submit_step = new_state.episode_done & ~old_state.episode_done

    step_penalty = jnp.array(reward_cfg.step_penalty, dtype=jnp.float32)

    training_similarity_reward = (
        reward_cfg.training_similarity_weight * similarity_improvement
    )
    progress_bonus = jnp.where(
        similarity_improvement > 0, reward_cfg.progress_bonus, 0.0
    )

    success_bonus = jnp.where(is_solved, reward_cfg.success_bonus, 0.0)
    success_bonus = jnp.where(
        reward_cfg.reward_on_submit_only, 
        jnp.where(is_submit_step, success_bonus, 0.0), 
        success_bonus
    )
    
    efficiency_bonus = jnp.where(
        is_solved & (new_state.step_count <= reward_cfg.efficiency_bonus_threshold),
        reward_cfg.efficiency_bonus,
        0.0,
    )

    # 2) Mode-specific totals
    training_reward = (
        training_similarity_reward
        + progress_bonus
        + step_penalty
        + success_bonus
        + efficiency_bonus
    )

    evaluation_reward = (
        step_penalty + success_bonus + efficiency_bonus
    )

    # 3) Select by mode
    return jnp.where(is_training, training_reward, evaluation_reward)
