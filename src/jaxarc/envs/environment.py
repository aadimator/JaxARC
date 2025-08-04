"""Clean class-based API for JaxARC environments.

This module provides a simple class-based interface that wraps the functional API,
with enhanced visualization and logging capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import chex
import jax.numpy as jnp
from loguru import logger

from jaxarc.envs.actions import get_action_handler
from jaxarc.envs.config import JaxArcConfig
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask
from jaxarc.utils.logging.experiment_logger import ExperimentLogger
from jaxarc.utils.visualization import (
    _clear_output_directory,
)


class ArcEnvironment:
    """Simple class-based interface for ARC environments.

    This class provides a clean, stateful interface that wraps the functional API
    with enhanced visualization and logging capabilities.

    Example:
        ```python
        from jaxarc.envs.config import JaxArcConfig

        config = JaxArcConfig()  # Uses default configuration
        env = ArcEnvironment(config)

        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        from jaxarc.envs import create_bbox_action
        action = create_bbox_action(operation=0, r1=0, c1=0, r2=2, c2=2)
        next_state, next_obs, reward, info = env.step(action)
        ```
    """

    config: JaxArcConfig
    _state: Optional[ArcEnvState]
    _episode_count: int
    action_handler: Any  # Action handler type depends on selection format
    _logger: Optional[ExperimentLogger]

    def __init__(self, config: JaxArcConfig):
        """Initialize environment with unified configuration.

        Args:
            config: Unified JaxArcConfig containing all configuration parameters
        """
        self.config = config
        self._state: Optional[ArcEnvState] = None
        self._episode_count = 0

        # Setup error handling and debugging
        self._setup_error_handling()

        # Get the action handler for this environment's selection format
        self.action_handler = get_action_handler(config.action.selection_format)

        # Initialize logging system
        self._logger: Optional[ExperimentLogger] = None
        self._setup_experiment_logger()

        logger.info("ArcEnvironment initialized with unified JaxArcConfig")
        logger.info(f"Using {config.action.selection_format} selection format")

        if self._logger is not None:
            logger.info("Experiment logging system enabled")

    def _setup_error_handling(self) -> None:
        """Setup error handling and debugging based on configuration."""
        from ..utils.error_handling import JAXErrorHandler
        from ..utils.debugging import configure_debugging
        
        # Setup error handling environment
        JAXErrorHandler.setup_error_environment()
        
        # Configure debugging based on debug level
        debug_level = self.config.environment.debug_level
        
        if debug_level == "off":
            error_mode = "raise"
            enable_nan_checks = False
        elif debug_level == "minimal":
            error_mode = "raise"
            enable_nan_checks = False
        elif debug_level == "standard":
            error_mode = "raise"
            enable_nan_checks = True
        elif debug_level == "verbose":
            error_mode = "raise"
            enable_nan_checks = True
        elif debug_level == "research":
            error_mode = "breakpoint"  # Enable interactive debugging for research
            enable_nan_checks = True
        else:
            error_mode = "raise"
            enable_nan_checks = True
        
        # Configure debugging
        configure_debugging(
            error_mode=error_mode,
            breakpoint_frames=3,
            enable_nan_checks=enable_nan_checks
        )
        
        logger.debug(f"Error handling configured: mode={error_mode}, nan_checks={enable_nan_checks}")

    def _setup_experiment_logger(self) -> None:
        """Setup experiment logging system based on configuration."""
        # Check if logging is enabled - always enabled unless debug level is "off"
        logging_enabled = (hasattr(self.config, 'environment') and 
                          self.config.environment.debug_level != "off")

        if not logging_enabled:
            return

        try:
            # Create ExperimentLogger with current configuration
            self._logger = ExperimentLogger(self.config)
            logger.debug("ExperimentLogger initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to setup experiment logger: {e}")
            logger.warning("Continuing without logging")
            self._logger = None



    def reset(
        self, key: chex.PRNGKey, task_data: Optional[JaxArcTask] = None
    ) -> Tuple[ArcEnvState, jnp.ndarray]:
        """Reset environment to initial state.

        Args:
            key: Random key for initialization
            task_data: Optional specific task data

        Returns:
            Tuple of (initial_state, initial_observation)
        """
        # Start new episode
        self._episode_count += 1
        
        # Legacy visualization directory clearing if needed
        if (
            self._logger is None and
            self.config.environment.debug_level != "off"
            and self.config.storage.clear_output_on_start
        ):
            self._clear_visualization_directory()

        self._state, obs = arc_reset(key, self.config, task_data)

        # Log episode start
        if self._logger is not None:
            episode_start_data = {
                'episode_num': self._episode_count,
                'task_id': getattr(self._state.task_data, 'task_id', 'unknown'),
                'initial_state': self._state,
                'task_data': self._state.task_data,
            }
            # Note: We don't have a log_episode_start method, but we can use this for initialization
            # The actual episode logging will happen in log_episode_summary

        return self._state, obs

    def step(
        self, action: Any
    ) -> Tuple[ArcEnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Step environment with given action.

        Args:
            action: Structured action to take (PointAction, BboxAction, or MaskAction)
                   Use create_point_action(), create_bbox_action(), or create_mask_action() 
                   factory functions to create actions

        Returns:
            Tuple of (next_state, observation, reward, info)

        Raises:
            RuntimeError: If environment hasn't been reset
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")

        # Store previous state for logging
        prev_state = self._state

        # Delegate action processing to arc_step which handles all formats
        self._state, obs, reward, done, info = arc_step(
            self._state, action, self.config
        )

        # Log step through JAX callback mechanism
        if self._logger is not None:
            # Serialize action for safe callback usage
            from jaxarc.utils.serialization_utils import serialize_action
            serialized_action = serialize_action(action)
            
            # Create step data for logging
            step_data = {
                'step_num': int(self._state.step_count),
                'before_state': prev_state,
                'after_state': self._state,
                'action': serialized_action,
                'reward': float(reward),
                'info': info,
            }
            
            # Use JAX callback to log step data
            from jax import debug
            debug.callback(self._log_step_callback, step_data)

            # Log episode summary if done
            if done:
                summary_data = {
                    'episode_num': self._episode_count,
                    'total_steps': int(self._state.step_count),
                    'total_reward': info.get("total_reward", float(reward)),
                    'final_similarity': float(self._state.similarity_score),
                    'success': info.get("success", False),
                    'task_id': getattr(self._state.task_data, 'task_id', 'unknown'),
                    'final_state': self._state,
                }
                
                # Use JAX callback to log episode summary
                debug.callback(self._log_episode_summary_callback, summary_data)

        return self._state, obs, reward, info



    def _log_step_callback(self, step_data: Dict[str, Any]) -> None:
        """JAX callback for logging step data.
        
        This method is called from within JAX transformations via jax.debug.callback.
        It safely passes the step data to the experiment logger.
        
        Args:
            step_data: Dictionary containing step information
        """
        if self._logger is not None:
            try:
                self._logger.log_step(step_data)
            except Exception as e:
                logger.warning(f"Failed to log step data: {e}")
    
    def _log_episode_summary_callback(self, summary_data: Dict[str, Any]) -> None:
        """JAX callback for logging episode summary.
        
        This method is called from within JAX transformations via jax.debug.callback.
        It safely passes the episode summary to the experiment logger.
        
        Args:
            summary_data: Dictionary containing episode summary information
        """
        if self._logger is not None:
            try:
                self._logger.log_episode_summary(summary_data)
            except Exception as e:
                logger.warning(f"Failed to log episode summary: {e}")

    def _clear_visualization_directory(self) -> None:
        """Clear the visualization output directory.

        This method safely clears the visualization directory
        to ensure clean output for each episode.
        """
        try:
            viz_dir = f"{self.config.storage.base_output_dir}/{self.config.storage.visualization_dir}"
            _clear_output_directory(viz_dir)
            logger.debug(f"Cleared visualization directory: {viz_dir}")
        except Exception as e:
            logger.warning(f"Failed to clear visualization directory: {e}")

    def close(self) -> None:
        """Clean up resources when environment is no longer needed."""
        if self._logger is not None:
            try:
                self._logger.close()
            except Exception as e:
                logger.warning(f"Error closing experiment logger: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

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
            "grid_shape": (
                self.config.dataset.max_grid_height,
                self.config.dataset.max_grid_width,
            ),
            "max_colors": self.config.dataset.max_colors,
            "selection_format": self.config.action.selection_format,
        }

    def get_action_space_info(self) -> Dict[str, Any]:
        """Get action space information."""
        if self.config.action.selection_format == "mask":
            return {
                "type": "dict",
                "selection_shape": (4,),  # [x1, y1, x2, y2]
                "selection_bounds": (
                    0,
                    max(
                        self.config.dataset.max_grid_height,
                        self.config.dataset.max_grid_width,
                    ),
                ),
                "operation_range": (0, self.config.action.max_operations),
            }
        if self.config.action.selection_format == "point":
            return {
                "type": "array",
                "shape": (3,),  # [x, y, operation]
                "bounds": (
                    0,
                    max(
                        self.config.dataset.max_grid_height,
                        self.config.dataset.max_grid_width,
                        self.config.action.max_operations,
                    ),
                ),
            }
        # bbox
        return {
            "type": "dict",
            "bbox_shape": (4,),  # [x1, y1, x2, y2]
            "bbox_bounds": (
                0,
                max(
                    self.config.dataset.max_grid_height,
                    self.config.dataset.max_grid_width,
                ),
            ),
            "operation_range": (0, self.config.action.max_operations),
        }
