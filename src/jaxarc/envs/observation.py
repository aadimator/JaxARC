"""
Observation utilities for JaxARC environments.

This module contains functions for constructing agent observations from
State, extracted from the monolithic functional API.
"""

from __future__ import annotations

from typing import Any

from jaxarc.state import State
from jaxarc.types import EnvParams
from jaxarc.utils.jax_types import ObservationArray


def _get_observation(state: State, _unused: Any) -> ObservationArray:
    """Extract observation from state.

    Currently returns the working grid; kept separate for future expansion.
    """
    return state.working_grid


def create_observation(state: State, params: EnvParams) -> ObservationArray:
    """Create agent observation using EnvParams-based API.

    Kept separate to support new functional signatures while maintaining legacy compatibility.
    """
    return _get_observation(state, params)
