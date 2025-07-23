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
    from jaxarc import ArcEnvironment
    from jaxarc.envs.config import JaxArcConfig

    # Create environment with unified configuration
    config = JaxArcConfig()  # Uses defaults
    env = ArcEnvironment(config)

    # Reset environment
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)

    # Take a step
    action = {"operation": 0, "selection": [5, 5, 7, 7]}
    state, obs, reward, info = env.step(action)
    ```
"""

from __future__ import annotations

from ._version import version as __version__

# Configuration package utilities for downstream consumption
from .config_pkg import (
    create_config_template,
    extend_jaxarc_config,
    get_jaxarc_config_dir,
    list_available_configs,
    load_jaxarc_config,
)

# Core environment and state
# Configuration system - most commonly used factory functions
# Functional API
from .envs import (
    ArcEnvironment,
    arc_reset,
    arc_step,
)

# Unified configuration system
from .envs.config import JaxArcConfig
from .state import ArcEnvState, create_arc_env_state

# Core types
from .types import ARCLEAction, Grid, JaxArcTask, TaskPair

__all__ = [
    # Core types
    "ARCLEAction",
    # Configuration
    "ArcEnvState",
    # Core environment and state
    "ArcEnvironment",
    "Grid",
    "JaxArcConfig",
    "JaxArcTask",
    "TaskPair",
    # Version
    "__version__",
    # Functional API
    "arc_reset",
    "arc_step",
    # Configuration package utilities
    "create_config_template",
    "create_arc_env_state",
    "extend_jaxarc_config",
    "get_jaxarc_config_dir",
    "list_available_configs",
    "load_jaxarc_config",
]
