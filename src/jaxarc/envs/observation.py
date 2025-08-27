"""
Observation utilities for JaxARC environments.

This module contains functions for constructing agent observations from
ArcEnvState, extracted from the monolithic functional API.
"""

from __future__ import annotations

from jaxarc.configs import JaxArcConfig
from jaxarc.state import ArcEnvState
from jaxarc.utils.jax_types import ObservationArray


def _get_observation(state: ArcEnvState, config: JaxArcConfig) -> ObservationArray:
    """Extract observation from state.

    Currently returns the working grid; kept separate for future expansion.
    """
    # Prevent unused-argument warnings while keeping extensible signature
    _ = config
    return state.working_grid


def create_observation(state: ArcEnvState, config: JaxArcConfig) -> ObservationArray:
    """Create agent observation from environment state.

    For now, returns the working grid to maintain backward compatibility.
    """
    return _get_observation(state, config)
