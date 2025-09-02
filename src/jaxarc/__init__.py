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
    timestep = env.reset(env_params, key)

    # Create mask action (core action format)
    mask = jnp.zeros((10, 10), dtype=jnp.bool_).at[5, 5].set(True)
    action = create_mask_action(operation=15, selection=mask)
    timestep = env.step(env_params, timestep, action)
    ```
"""

from __future__ import annotations

from ._version import version as __version__

# Unified configuration system
from .configs import JaxArcConfig

# Action system (mask-based actions are the core format)
from .envs.actions import MaskAction, create_mask_action

# Core environment and state
# Functional API
from .state import State

# Core types
from .types import EnvParams, Grid, JaxArcTask, TaskPair, TimeStep

__all__ = [
    "EnvParams",
    "Grid",
    "JaxArcConfig",
    "JaxArcTask",
    "MaskAction",
    "State",
    "TaskPair",
    "TimeStep",
    "__version__",
    "create_mask_action",
]
