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
    from jaxarc import ArcEnvironment, create_standard_config
    
    # Create environment with standard configuration
    config = create_standard_config()
    env = ArcEnvironment(config)
    
    # Reset environment
    key = jax.random.PRNGKey(42)
    state = env.reset(key)
    
    # Take a step
    action = {"operation": 0, "point": [5, 5]}
    state, reward, done, info = env.step(state, action, key)
    ```
"""

from __future__ import annotations

from ._version import version as __version__

# Core environment and state
from .envs import ArcEnvironment
from .state import ArcEnvState

# Core types
from .types import Grid, JaxArcTask, ARCLEAction, TaskPair

# Configuration system - most commonly used factory functions
from .envs import (
    create_standard_config,
    create_raw_config,
    create_full_config,
    create_point_config,
    create_bbox_config,
    ArcEnvConfig,
)

# Functional API
from .envs import arc_reset, arc_step

__all__ = [
    # Version
    "__version__",
    
    # Core environment and state
    "ArcEnvironment", 
    "ArcEnvState",
    
    # Core types
    "Grid",
    "JaxArcTask", 
    "ARCLEAction",
    "TaskPair",
    
    # Configuration
    "ArcEnvConfig",
    "create_standard_config",
    "create_raw_config", 
    "create_full_config",
    "create_point_config",
    "create_bbox_config",
    
    # Functional API
    "arc_reset",
    "arc_step",
]
