"""Stoix-compatible environment factory for JaxARC.

Provides ``make_jaxarc_env()`` that creates JaxARC environments with the
standard wrapper chain expected by Stoix, and ``jaxarc_custom_metrics()``
for processing domain-specific episode metrics.
"""

from __future__ import annotations

from omegaconf import DictConfig
from stoa.environment import Environment

from jaxarc.configs import JaxArcConfig
from jaxarc.envs import (
    AnswerObservationWrapper,
    BboxActionWrapper,
    ContextualObservationWrapper,
    FlattenActionWrapper,
    InputGridObservationWrapper,
    PointActionWrapper,
)
from jaxarc.registration import make
from jaxarc.wrappers import (
    AllowedOperationsWrapper,
    EvalMetricsWrapper,
    SelectionAwareFlattenWrapper,
)


def make_jaxarc_env(
    config: DictConfig,
) -> tuple[Environment, Environment]:
    """Create JaxARC train and eval environments for Stoix.

    Follows the standard Stoix make_*_env() pattern:
    1. Create base environments via jaxarc.registration.make()
    2. Apply action wrappers (Point/Bbox) + SelectionAwareFlatten (FlattenAction for mask)
    3. Apply observation wrappers (Answer/Input/Contextual)
    4. Return (train_env, eval_env) ready for Stoix's core wrapper chain

    Does NOT apply Stoix core wrappers (AddRNGKey, RecordEpisodeMetrics,
    AutoResetWrapper, VmapWrapper) or ExtendedMetrics. Those must be applied
    by the caller in the correct order: AddRNGKey → RecordEpisodeMetrics →
    ExtendedMetrics → AutoResetWrapper → VmapWrapper.

    Args:
        config: OmegaConf DictConfig with env.scenario.name, env.action.mode,
                env.observation_wrappers settings. The eval env uses test pairs
                unless env.eval_episode_mode is set to 0, and the train scenario
                unless env.eval_scenario is set. The answer channel is off unless
                env.observation_wrappers.answer_grid is true. The eval env
                reports each episode's outcome in extras["eval_metrics"].

    Returns:
        Tuple of (train_env, eval_env) implementing stoa.Environment.

    Raises:
        ValueError: If action.mode is not one of "point", "bbox", "mask".
    """
    jaxarc_config = JaxArcConfig.from_hydra(config)
    scenario = config.env.scenario.name

    eval_scenario = config.env.get("eval_scenario") or scenario
    eval_episode_mode = config.env.get("eval_episode_mode", 1)

    # The mode is set explicitly because make() otherwise infers test mode from
    # a scenario name like "AGI1-eval".
    env, _ = make(scenario, config=jaxarc_config, auto_download=True, episode_mode=0)
    eval_env, _ = make(
        eval_scenario,
        config=jaxarc_config,
        auto_download=True,
        episode_mode=eval_episode_mode,
    )

    # Action wrappers
    action_mode = config.env.action.mode
    if action_mode == "point":
        env = PointActionWrapper(env)
        eval_env = PointActionWrapper(eval_env)
    elif action_mode == "bbox":
        env = BboxActionWrapper(env)
        eval_env = BboxActionWrapper(eval_env)
    elif action_mode != "mask":
        msg = f"Unknown action mode: {action_mode}"
        raise ValueError(msg)

    if action_mode != "mask" and jaxarc_config.action.allowed_operations is not None:
        env = AllowedOperationsWrapper(env)
        eval_env = AllowedOperationsWrapper(eval_env)

    # Point and bbox get one action per selection-independent operation.
    if action_mode == "mask":
        env = FlattenActionWrapper(env)
        eval_env = FlattenActionWrapper(eval_env)
    else:
        env = SelectionAwareFlattenWrapper(env)
        eval_env = SelectionAwareFlattenWrapper(eval_env)

    # Observation wrappers
    obs_wrappers = config.env.get("observation_wrappers", {})
    if obs_wrappers.get("answer_grid", False):
        env = AnswerObservationWrapper(env)
        eval_env = AnswerObservationWrapper(eval_env)
    if obs_wrappers.get("input_grid", True):
        env = InputGridObservationWrapper(env)
        eval_env = InputGridObservationWrapper(eval_env)
    if obs_wrappers.get("contextual", True):
        env = ContextualObservationWrapper(env)
        eval_env = ContextualObservationWrapper(eval_env)

    # Held-out scoring, for jaxarc.evaluation and Stoix's evaluator.
    eval_env = EvalMetricsWrapper(eval_env)

    return env, eval_env


def jaxarc_custom_metrics(metrics: dict) -> dict:
    """Process JaxARC extended metrics for Stoix logging.

    Computes derived statistics from raw episode metrics emitted by the
    ExtendedMetrics wrapper. Safe to call on metrics dicts that don't
    contain JaxARC-specific keys (returns unchanged).

    Args:
        metrics: Raw metrics dict from Stoix training loop.

    Returns:
        Metrics dict with added derived fields:
        - success_rate: float (0-100)
        - avg_steps_to_solve: float
        - truncation_rate: float (0-100)
        - best_similarity_mean: float (0-1)
        - final_similarity_mean: float (0-1)
    """
    import jax.numpy as jnp

    if "solved" not in metrics:
        return metrics

    solved = metrics["solved"]
    success_rate = jnp.mean(solved.astype(jnp.float32)) * 100.0

    steps_to_solve = metrics.get("steps_to_solve", jnp.array(0, dtype=jnp.int32))
    solved_mask = solved.astype(jnp.float32)
    solved_count = jnp.sum(solved_mask)
    avg_steps = jnp.where(
        solved_count > 0,
        jnp.sum(steps_to_solve * solved_mask) / solved_count,
        0.0,
    )

    was_truncated = metrics.get("was_truncated", jnp.array(False))
    truncation_rate = jnp.mean(was_truncated.astype(jnp.float32)) * 100.0

    best_similarity = metrics.get(
        "best_similarity", jnp.array(0.0, dtype=jnp.float32)
    )
    final_similarity = metrics.get(
        "final_similarity", jnp.array(0.0, dtype=jnp.float32)
    )

    return {
        **metrics,
        "success_rate": success_rate,
        "avg_steps_to_solve": avg_steps,
        "truncation_rate": truncation_rate,
        "best_similarity_mean": jnp.mean(best_similarity),
        "final_similarity_mean": jnp.mean(final_similarity),
    }
