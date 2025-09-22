"""JaxARC: Single-Agent Reinforcement Learning environment for ARC dataset in JAX.

JaxARC provides a JAX-native environment for training agents on ARC (Abstraction and
Reasoning Corpus) tasks with focus on single-agent reinforcement learning, designed
for high performance and extensibility.

Key Features:
- JAX-native implementation with full jit/vmap/pmap support
- Single-agent RL focus with extensible architecture
- Multiple action formats (point, bbox, mask)
- Comprehensive configuration system with Hydra integration
- Rich visualization and debugging utilities

Examples:
    ```python
    import jax
    import jax.numpy as jnp
    from jaxarc import JaxArcConfig, create_mask_action
    from jaxarc.registration import make

    # Create environment
    config = JaxArcConfig()
    env, env_params = make("Mini", config=config)

    # Reset environment
    key = jax.random.PRNGKey(42)
    state, timestep = env.reset(key, env_params=env_params)

    # Create mask action (core action format)
    mask = jnp.zeros((10, 10), dtype=jnp.bool_).at[5, 5].set(True)
    action = create_action(operation=15, selection=mask)
    state, timestep = env.step(state, action, env_params=env_params)
    ```
"""

from __future__ import annotations

from ._version import version as __version__

# Unified configuration system
from .configs import JaxArcConfig

# Action system (actions are the core format)
from .envs.actions import Action, create_action

# Core environment and state
# Functional API
from .state import State

# Core types
from .types import EnvParams, Grid, JaxArcTask, StepType, TaskPair, TimeStep

__all__ = [
    "Action",
    "EnvParams",
    "Grid",
    "JaxArcConfig",
    "JaxArcTask",
    "State",
    "StepType",
    "TaskPair",
    "TimeStep",
    "__version__",
    "create_action",
]
