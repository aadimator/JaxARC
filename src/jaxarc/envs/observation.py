"""
Observation utilities for JaxARC environments.

This module contains functions for constructing agent observations from
ArcEnvState, extracted from the monolithic functional API.
"""

from __future__ import annotations

from typing import Any
from jaxarc.configs import JaxArcConfig
from jaxarc.state import ArcEnvState
from jaxarc.types import EnvParams
from jaxarc.utils.jax_types import ObservationArray


def _get_observation(state: ArcEnvState, _unused: Any) -> ObservationArray:
    """Extract observation from state.

    Currently returns the working grid; kept separate for future expansion.
    """
    return state.working_grid


def create_observation(state: ArcEnvState, config: JaxArcConfig) -> ObservationArray:
    """Create agent observation from environment state.

    For now, returns the working grid to maintain backward compatibility.
    """
    return _get_observation(state, config)


def create_observation_from_params(state: ArcEnvState, params: EnvParams) -> ObservationArray:
    """Create agent observation using EnvParams-based API.

    Kept separate to support new functional signatures while maintaining legacy compatibility.
    """
    return _get_observation(state, params)
