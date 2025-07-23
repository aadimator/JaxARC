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
    from jaxarc.envs.equinox_config import JaxArcConfig
    from jaxarc.envs.factory import create_standard_config
    from jaxarc.envs.equinox_config import convert_arc_env_config_to_jax_arc_config

    # Create environment with unified configuration
    legacy_config = create_standard_config()
    config = convert_arc_env_config_to_jax_arc_config(legacy_config)
    env = ArcEnvironment(config)

    # Or create directly with JaxArcConfig
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
    ArcEnvConfig,
    ArcEnvironment,
    arc_reset,
    arc_step,
    create_bbox_config,
    create_full_config,
    create_point_config,
    create_raw_config,
    create_standard_config,
)

# Unified configuration system
from .envs.equinox_config import JaxArcConfig, convert_arc_env_config_to_jax_arc_config
from .state import ArcEnvState

# Core types
from .types import ARCLEAction, Grid, JaxArcTask, TaskPair

__all__ = [
    # Core types
    "ARCLEAction",
    # Configuration
    "ArcEnvConfig",
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
    # Unified configuration
    "convert_arc_env_config_to_jax_arc_config",
    "create_bbox_config",
    # Configuration package utilities
    "create_config_template",
    "create_full_config",
    "create_point_config",
    "create_raw_config",
    "create_standard_config",
    "extend_jaxarc_config",
    "get_jaxarc_config_dir",
    "list_available_configs",
    "load_jaxarc_config",
]
