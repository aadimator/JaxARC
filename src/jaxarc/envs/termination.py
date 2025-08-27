"""
Termination utilities for JaxARC environments.

This module provides episode termination checks extracted from the functional API.
"""

from __future__ import annotations

from jaxarc.configs import JaxArcConfig
from jaxarc.state import ArcEnvState
from jaxarc.utils.jax_types import EpisodeDone


def _is_episode_done(state: ArcEnvState, config: JaxArcConfig) -> EpisodeDone:
    """Basic termination: max steps, or submitted."""
    max_steps_reached = state.step_count >= config.environment.max_episode_steps
    submitted = state.episode_done
    return max_steps_reached | submitted
