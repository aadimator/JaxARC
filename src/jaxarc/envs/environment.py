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
from jaxarc.envs.equinox_config import JaxArcConfig
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask
from jaxarc.utils.visualization import (
    AsyncLogger,
    AsyncLoggerConfig,
    EnhancedVisualizer,
    EpisodeConfig,
    EpisodeManager,
    VisualizationConfig,
    WandbIntegration,
    _clear_output_directory,
)


class ArcEnvironment:
    """Simple class-based interface for ARC environments.

    This class provides a clean, stateful interface that wraps the functional API
    with enhanced visualization and logging capabilities.

    Example:
        ```python
        from jaxarc.envs.equinox_config import JaxArcConfig

        config = JaxArcConfig()  # Uses default configuration
        env = ArcEnvironment(config)

        key = jax.random.PRNGKey(42)
        state, obs = env.reset(key)

        action = {"selection": jnp.array([0, 0, 2, 2]), "operation": 0}
        next_state, next_obs, reward, info = env.step(action)
        ```
    """

    config: JaxArcConfig
    _state: Optional[ArcEnvState]
    _episode_count: int
    action_handler: Any  # Action handler type depends on selection format
    _enhanced_visualizer: Optional[EnhancedVisualizer]
    _episode_manager: Optional[EpisodeManager]
    _async_logger: Optional[AsyncLogger]
    _wandb_integration: Optional[WandbIntegration]

    def __init__(self, config: JaxArcConfig):
        """Initialize environment with unified configuration.

        Args:
            config: Unified JaxArcConfig containing all configuration parameters
        """
        self.config = config
        self._state: Optional[ArcEnvState] = None
        self._episode_count = 0

        # Get the action handler for this environment's selection format
        self.action_handler = get_action_handler(config.action.selection_format)

        # Initialize enhanced visualization system if enabled
        self._enhanced_visualizer: Optional[EnhancedVisualizer] = None
        self._episode_manager: Optional[EpisodeManager] = None
        self._async_logger: Optional[AsyncLogger] = None
        self._wandb_integration: Optional[WandbIntegration] = None

        self._setup_enhanced_visualization()

        logger.info("ArcEnvironment initialized with unified JaxArcConfig")
        logger.info(f"Using {config.action.selection_format} selection format")

        if self._enhanced_visualizer is not None:
            logger.info("Enhanced visualization system enabled")

    def _setup_enhanced_visualization(self) -> None:
        """Setup enhanced visualization system based on configuration."""
        # Check if enhanced visualization is enabled based on unified config
        enhanced_enabled = self.config.visualization.enabled

        # Also enable if debug level indicates visualization should be active
        if not enhanced_enabled and self.config.environment.debug_level in [
            "standard",
            "verbose",
            "research",
        ]:
            enhanced_enabled = True

        if not enhanced_enabled:
            return

        try:
            # Create visualization configuration
            vis_config = self._create_visualization_config()

            # Create episode manager
            episode_config = self._create_episode_config()
            self._episode_manager = EpisodeManager(episode_config)

            # Create async logger
            async_config = self._create_async_logger_config()
            self._async_logger = AsyncLogger(async_config)

            # Create wandb integration if configured
            self._wandb_integration = self._create_wandb_integration()

            # Create enhanced visualizer
            self._enhanced_visualizer = EnhancedVisualizer(
                vis_config=vis_config,
                episode_manager=self._episode_manager,
                async_logger=self._async_logger,
                wandb_integration=self._wandb_integration,
            )

        except Exception as e:
            logger.warning(f"Failed to setup enhanced visualization: {e}")
            logger.warning("Falling back to legacy visualization")
            self._enhanced_visualizer = None

    def _create_visualization_config(self) -> VisualizationConfig:
        """Create visualization configuration from unified config."""
        vis_cfg = self.config.visualization
        return VisualizationConfig(
            debug_level=self.config.environment.computed_visualization_level,
            output_formats=vis_cfg.output_formats or ["svg"],
            image_quality=vis_cfg.image_quality,
            show_coordinates=vis_cfg.show_coordinates,
            show_operation_names=vis_cfg.show_operation_names,
            highlight_changes=vis_cfg.highlight_changes,
            include_metrics=vis_cfg.include_metrics,
            color_scheme=vis_cfg.color_scheme,
        )

    def _create_episode_config(self) -> EpisodeConfig:
        """Create episode configuration from unified config."""
        storage_cfg = self.config.storage
        base_dir = f"{storage_cfg.base_output_dir}/{storage_cfg.episodes_dir}"

        return EpisodeConfig(
            base_output_dir=base_dir,
            cleanup_policy=storage_cfg.cleanup_policy,
            max_storage_gb=storage_cfg.max_storage_gb,
        )

    def _create_async_logger_config(self) -> AsyncLoggerConfig:
        """Create async logger configuration from unified config."""
        log_cfg = self.config.logging
        return AsyncLoggerConfig(
            queue_size=log_cfg.queue_size,
            worker_threads=log_cfg.worker_threads,
            enable_compression=log_cfg.enable_compression,
        )

    def _create_wandb_integration(self) -> Optional[WandbIntegration]:
        """Create wandb integration if configured."""
        wandb_cfg = self.config.wandb
        if wandb_cfg.enabled:
            return WandbIntegration(wandb_cfg)
        return None

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
        # Start new episode in enhanced visualization system
        if self._enhanced_visualizer is not None:
            self._episode_count += 1
            self._enhanced_visualizer.start_episode(self._episode_count)
        # Legacy visualization directory clearing
        elif (
            self.config.environment.debug_level != "off"
            and self.config.storage.clear_output_on_start
        ):
            self._clear_visualization_directory()

        self._state, obs = arc_reset(key, self.config, task_data)

        # Log episode start with enhanced visualization
        if self._enhanced_visualizer is not None:
            self._enhanced_visualizer.log_episode_start(
                episode_num=self._episode_count,
                task_data=self._state.task_data,
                initial_state=self._state,
            )

        return self._state, obs

    def step(
        self, action: Any
    ) -> Tuple[ArcEnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Step environment with given action.

        Args:
            action: Action to take (format depends on config.action.selection_format)
                   For point: {"point": [row, col], "operation": operation_id}
                   For bbox: {"bbox": [r1, c1, r2, c2], "operation": operation_id}
                   For mask: {"mask": mask_array, "operation": operation_id}

        Returns:
            Tuple of (next_state, observation, reward, info)

        Raises:
            RuntimeError: If environment hasn't been reset
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")

        # Store previous state for visualization
        prev_state = self._state

        # Delegate action processing to arc_step which handles all formats
        self._state, obs, reward, done, info = arc_step(
            self._state, action, self.config
        )

        # Enhanced visualization step logging
        if self._enhanced_visualizer is not None:
            self._enhanced_visualizer.log_step(
                step_num=int(self._state.step_count),
                before_state=prev_state,
                action=action,
                after_state=self._state,
                reward=float(reward),
                info=info,
            )

            # Log episode end if done
            if done:
                self._enhanced_visualizer.log_episode_end(
                    episode_num=self._episode_count,
                    final_state=self._state,
                    total_reward=info.get("total_reward", float(reward)),
                    success=info.get("success", False),
                )

        return self._state, obs, reward, info

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
        if self._async_logger is not None:
            try:
                self._async_logger.close()
            except Exception as e:
                logger.warning(f"Error closing async logger: {e}")

        if self._wandb_integration is not None:
            try:
                self._wandb_integration.finish_run()
            except Exception as e:
                logger.warning(f"Error finishing wandb run: {e}")

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
