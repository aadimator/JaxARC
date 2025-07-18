"""Clean class-based API for JaxARC environments.

This module provides a simple class-based interface that wraps the functional API,
with enhanced visualization and logging capabilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import chex
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig

from jaxarc.envs.actions import get_action_handler
from jaxarc.envs.config import ArcEnvConfig
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.state import ArcEnvState
from jaxarc.types import JaxArcTask
from jaxarc.utils.visualization import (
    _clear_output_directory,
    EnhancedVisualizer,
    VisualizationConfig,
    EpisodeManager,
    EpisodeConfig,
    AsyncLogger,
    AsyncLoggerConfig,
    WandbIntegration,
    WandbConfig,
    create_development_wandb_config,
)


class ArcEnvironment:
    """Simple class-based interface for ARC environments.

    This class provides a clean, stateful interface that wraps the functional API
    with enhanced visualization and logging capabilities.

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

    def __init__(self, config: ArcEnvConfig, hydra_config: Optional[DictConfig] = None):
        """Initialize environment with configuration.

        Args:
            config: Typed configuration for the environment
            hydra_config: Optional Hydra configuration for enhanced visualization
        """
        self.config = config
        self.hydra_config = hydra_config
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

        logger.info(f"ArcEnvironment initialized with config: {self.config}")
        logger.info(f"Using {config.action.selection_format} selection format")
        
        if self._enhanced_visualizer is not None:
            logger.info("Enhanced visualization system enabled")

    def _setup_enhanced_visualization(self) -> None:
        """Setup enhanced visualization system based on configuration."""
        # Check if enhanced visualization is enabled
        enhanced_enabled = False
        
        if self.hydra_config is not None:
            enhanced_enabled = self.hydra_config.get("enhanced_visualization", {}).get("enabled", False)
        
        # Fallback to legacy debug settings if no enhanced config
        if not enhanced_enabled and self.config.debug.log_rl_steps:
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
                wandb_integration=self._wandb_integration
            )
            
        except Exception as e:
            logger.warning(f"Failed to setup enhanced visualization: {e}")
            logger.warning("Falling back to legacy visualization")
            self._enhanced_visualizer = None

    def _create_visualization_config(self) -> VisualizationConfig:
        """Create visualization configuration from Hydra config or defaults."""
        if self.hydra_config is not None and "visualization" in self.hydra_config:
            vis_cfg = self.hydra_config.visualization
            return VisualizationConfig(
                debug_level=vis_cfg.get("debug_level", "standard"),
                output_formats=vis_cfg.get("output_formats", ["svg"]),
                image_quality=vis_cfg.get("image_quality", "high"),
                show_coordinates=vis_cfg.get("show_coordinates", False),
                show_operation_names=vis_cfg.get("show_operation_names", True),
                highlight_changes=vis_cfg.get("highlight_changes", True),
                include_metrics=vis_cfg.get("include_metrics", True),
                color_scheme=vis_cfg.get("color_scheme", "default"),
            )
        else:
            # Use defaults based on legacy debug settings
            debug_level = "standard" if self.config.debug.log_rl_steps else "off"
            return VisualizationConfig(debug_level=debug_level)

    def _create_episode_config(self) -> EpisodeConfig:
        """Create episode configuration from Hydra config or defaults."""
        base_dir = self.config.debug.rl_steps_output_dir
        
        if self.hydra_config is not None and "storage" in self.hydra_config:
            storage_cfg = self.hydra_config.storage
            return EpisodeConfig(
                base_output_dir=storage_cfg.get("base_output_dir", base_dir),
                cleanup_policy=storage_cfg.get("cleanup_policy", "size_based"),
                max_storage_gb=storage_cfg.get("max_storage_gb", 5.0),
            )
        else:
            return EpisodeConfig(base_output_dir=base_dir)

    def _create_async_logger_config(self) -> AsyncLoggerConfig:
        """Create async logger configuration from Hydra config or defaults."""
        if self.hydra_config is not None and "logging" in self.hydra_config:
            log_cfg = self.hydra_config.logging
            return AsyncLoggerConfig(
                queue_size=log_cfg.get("queue_size", 1000),
                worker_threads=log_cfg.get("worker_threads", 2),
                enable_compression=log_cfg.get("enable_compression", True),
            )
        else:
            return AsyncLoggerConfig()

    def _create_wandb_integration(self) -> Optional[WandbIntegration]:
        """Create wandb integration if configured."""
        if self.hydra_config is not None and "wandb" in self.hydra_config:
            wandb_cfg = self.hydra_config.wandb
            if wandb_cfg.get("enabled", False):
                config = WandbConfig(
                    enabled=True,
                    project_name=wandb_cfg.get("project_name", "jaxarc-experiments"),
                    entity=wandb_cfg.get("entity"),
                    tags=wandb_cfg.get("tags", []),
                    log_frequency=wandb_cfg.get("log_frequency", 10),
                )
                return WandbIntegration(config)
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
        else:
            # Legacy visualization directory clearing
            if self.config.debug.log_rl_steps and self.config.debug.clear_output_dir:
                self._clear_visualization_directory()

        self._state, obs = arc_reset(key, self.config, task_data)
        
        # Log episode start with enhanced visualization
        if self._enhanced_visualizer is not None:
            self._enhanced_visualizer.log_episode_start(
                episode_num=self._episode_count,
                task_data=self._state.task_data,
                initial_state=self._state
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
                info=info
            )
            
            # Log episode end if done
            if done:
                self._enhanced_visualizer.log_episode_end(
                    episode_num=self._episode_count,
                    final_state=self._state,
                    total_reward=info.get("total_reward", float(reward)),
                    success=info.get("success", False)
                )
        
        return self._state, obs, reward, info

    def _clear_visualization_directory(self) -> None:
        """Clear the visualization output directory.

        This method safely clears the RL steps visualization directory
        to ensure clean output for each episode.
        """
        try:
            _clear_output_directory(self.config.debug.rl_steps_output_dir)
            logger.debug(
                f"Cleared visualization directory: {self.config.debug.rl_steps_output_dir}"
            )
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
                self.config.grid.max_grid_height,
                self.config.grid.max_grid_width,
            ),
            "max_colors": self.config.grid.max_colors,
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
                        self.config.grid.max_grid_height,
                        self.config.grid.max_grid_width,
                    ),
                ),
                "operation_range": (0, self.config.action.num_operations),
            }
        if self.config.action.selection_format == "point":
            return {
                "type": "array",
                "shape": (3,),  # [x, y, operation]
                "bounds": (
                    0,
                    max(
                        self.config.grid.max_grid_height,
                        self.config.grid.max_grid_width,
                        self.config.action.num_operations,
                    ),
                ),
            }
        # bbox
        return {
            "type": "dict",
            "bbox_shape": (4,),  # [x1, y1, x2, y2]
            "bbox_bounds": (
                0,
                max(self.config.grid.max_grid_height, self.config.grid.max_grid_width),
            ),
            "operation_range": (0, self.config.action.num_operations),
        }
