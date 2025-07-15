"""Clean class-based API for JaxARC environments.

This module provides a simple class-based interface that wraps the functional API,
without any backward compatibility bloat.
"""

from typing import Any, Dict, Optional, Tuple

import chex
import jax.numpy as jnp
from loguru import logger

from jaxarc.envs.actions import get_action_handler
from jaxarc.envs.config import ArcEnvConfig
from jaxarc.envs.functional import ArcEnvState, arc_reset, arc_step
from jaxarc.utils.visualization import _clear_output_directory
from jaxarc.types import JaxArcTask


class ArcEnvironment:
    """Simple class-based interface for ARC environments.

    This class provides a clean, stateful interface that wraps the functional API.
    It's designed to work seamlessly with the new config system.

    Example:
        ```python
        config = ArcEnvConfig(max_episode_steps=100)
        env = ArcEnvironment(config)

        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        action = {"selection": jnp.array([0, 0, 2, 2]), "operation": 0}
        next_state, next_obs, reward, info = env.step(action)
        ```
    """

    def __init__(self, config: ArcEnvConfig):
        """Initialize environment with configuration.

        Args:
            config: Typed configuration for the environment
        """
        self.config = config
        self._state: Optional[ArcEnvState] = None

        # Get the action handler for this environment's action format
        self.action_handler = get_action_handler(config.action.action_format)

        logger.info(f"ArcEnvironment initialized with config: {self.config}")
        logger.info(f"Using {config.action.action_format} action format")

    def reset(
        self,
        key: chex.PRNGKey,
        task_data: Optional[JaxArcTask] = None
    ) -> Tuple[ArcEnvState, jnp.ndarray]:
        """Reset environment to initial state.

        Args:
            key: Random key for initialization
            task_data: Optional specific task data

        Returns:
            Tuple of (initial_state, initial_observation)
        """
        # Clear visualization directory if enabled
        if self.config.debug.log_rl_steps and self.config.debug.clear_output_dir:
            self._clear_visualization_directory()

        self._state, obs = arc_reset(key, self.config, task_data)
        return self._state, obs

    def step(
        self,
        action: Any
    ) -> Tuple[ArcEnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Step environment with given action.

        Args:
            action: Action to take (format depends on config.action.action_format)
                   Expected format: {"selection": action_data, "operation": operation_id}

        Returns:
            Tuple of (next_state, observation, reward, info)

        Raises:
            RuntimeError: If environment hasn't been reset
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")

        # Convert action to mask using the configured handler
        action_mask = self.action_handler(action["selection"], self._state.working_grid_mask)

        # Create standardized action dict for downstream processing
        standardized_action = {
            "selection": action_mask,
            "operation": action["operation"]
        }

        self._state, obs, reward, done, info = arc_step(self._state, standardized_action, self.config)
        return self._state, obs, reward, info

    def _clear_visualization_directory(self) -> None:
        """Clear the visualization output directory.

        This method safely clears the RL steps visualization directory
        to ensure clean output for each episode.
        """
        try:
            _clear_output_directory(self.config.debug.rl_steps_output_dir)
            logger.debug(f"Cleared visualization directory: {self.config.debug.rl_steps_output_dir}")
        except Exception as e:
            logger.warning(f"Failed to clear visualization directory: {e}")

    @property
    def state(self) -> Optional[ArcEnvState]:
        """Get current environment state."""
        return self._state

    @property
    def is_done(self) -> bool:
        """Check if current episode is done."""
        return self._state is None or self._state.episode_done

    def get_observation_space_info(self) -> Dict[str, Any]:
        """Get observation space information."""
        return {
            "grid_shape": (self.config.grid.max_grid_height, self.config.grid.max_grid_width),
            "max_colors": self.config.grid.max_colors,
            "action_format": self.config.action.action_format,
        }

    def get_action_space_info(self) -> Dict[str, Any]:
        """Get action space information."""
        if self.config.action.action_format == "selection_operation":
            return {
                "type": "dict",
                "selection_shape": (4,),  # [x1, y1, x2, y2]
                "selection_bounds": (0, max(self.config.grid.max_grid_height, self.config.grid.max_grid_width)),
                "operation_range": (0, self.config.action.num_operations),
            }
        elif self.config.action.action_format == "point":
            return {
                "type": "array",
                "shape": (3,),  # [x, y, operation]
                "bounds": (0, max(
                    self.config.grid.max_grid_height,
                    self.config.grid.max_grid_width,
                    self.config.action.num_operations
                )),
            }
        else:  # bbox
            return {
                "type": "dict",
                "bbox_shape": (4,),  # [x1, y1, x2, y2]
                "bbox_bounds": (0, max(self.config.grid.max_grid_height, self.config.grid.max_grid_width)),
                "operation_range": (0, self.config.action.num_operations),
            }
