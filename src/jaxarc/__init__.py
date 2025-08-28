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
    from jaxarc.configs import JaxArcConfig

    # Create environment with unified configuration
    config = JaxArcConfig()  # Uses defaults
    env = ArcEnvironment(config)

    # Reset environment
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)

    # Take a step with structured action
    from jaxarc.envs import create_bbox_action

    action = create_bbox_action(operation=0, r1=5, c1=5, r2=7, c2=7)
    state, obs, reward, info = env.step(action)
    ```
"""

from __future__ import annotations

from ._version import version as __version__

# Unified configuration system
from .configs import JaxArcConfig

# Core environment and state
# Functional API

from .state import State

# Core types
from .types import ARCAction, Grid, JaxArcTask, TaskPair, EnvParams, TimeStep

__all__ = [
    # Core types
    "ARCAction",
    "EnvParams",
    "TimeStep",
    # Configuration
    "State",
    "Grid",
    "JaxArcConfig",
    "JaxArcTask",
    "TaskPair",
    # Version
    "__version__",
    # Functional API


]
