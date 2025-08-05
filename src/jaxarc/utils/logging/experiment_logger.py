"""Central logging coordinator for JaxARC experiments.

This module provides the ExperimentLogger class, which serves as the central
coordinator for all logging operations in JaxARC. It manages a set of focused
handlers for different logging concerns (file, SVG, console, wandb) and provides
error isolation between handlers.

The ExperimentLogger follows the design principles:
- Single entry point for all logging operations
- Handler-based architecture with single responsibility
- Graceful degradation when handlers fail
- JAX compatibility through existing callback mechanisms
- Configuration-driven handler initialization
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from ..config import get_config
from ...envs.config import JaxArcConfig


class ExperimentLogger:
    """Central logging coordinator with handler-based architecture.
    
    This class serves as the single entry point for all logging operations
    in JaxARC. It manages a set of handlers for different logging concerns
    and provides error isolation to ensure that failures in one handler
    don't affect others.
    
    Note: This is a regular Python class (not equinox.Module) because it needs
    mutable state to manage handlers and doesn't need to be JAX-compatible.
    
    Attributes:
        config: JaxARC configuration object
        handlers: Dictionary of active handler instances
        
    Examples:
        ```python
        from jaxarc.utils.logging import ExperimentLogger
        from jaxarc.envs.config import JaxArcConfig
        
        # Initialize logger with configuration
        config = JaxArcConfig(...)
        logger = ExperimentLogger(config)
        
        # Log step data
        step_data = {
            'step_num': 1,
            'before_state': state,
            'after_state': new_state,
            'action': action,
            'reward': 0.5,
            'info': {'metrics': {'similarity': 0.8}}
        }
        logger.log_step(step_data)
        
        # Log episode summary
        summary_data = {
            'episode_num': 1,
            'total_steps': 50,
            'total_reward': 10.0,
            'final_similarity': 0.95,
            'success': True
        }
        logger.log_episode_summary(summary_data)
        
        # Clean shutdown
        logger.close()
        ```
    """
    
    def __init__(self, config: JaxArcConfig):
        """Initialize logger with handlers based on configuration.
        
        Args:
            config: JaxARC configuration object containing logging settings
        """
        self.config = config
        self.handlers = self._initialize_handlers()
        
        logger.info(f"ExperimentLogger initialized with {len(self.handlers)} handlers")
    
    def _initialize_handlers(self) -> Dict[str, Any]:
        """Initialize handlers based on configuration settings.
        
        This method creates handler instances based on the configuration,
        with graceful fallback if handler initialization fails.
        
        Returns:
            Dictionary mapping handler names to handler instances
        """
        handlers = {}
        
        # Get debug level from environment config
        debug_level = "off"
        if hasattr(self.config, 'environment') and hasattr(self.config.environment, 'debug_level'):
            debug_level = self.config.environment.debug_level
        
        # Early return if logging is completely disabled
        if debug_level == "off":
            logger.debug("Logging disabled (debug_level='off')")
            return handlers
        
        try:
            # File logging handler - enabled unless debug level is "off"
            if debug_level != "off":
                handlers['file'] = self._create_file_handler()
                logger.debug("FileHandler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize FileHandler: {e}")
        
        try:
            # SVG visualization handler - enabled for standard and above debug levels
            if debug_level in ["standard", "verbose", "research"]:
                handlers['svg'] = self._create_svg_handler()
                logger.debug("SVGHandler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SVGHandler: {e}")
        
        try:
            # Console output handler - enabled unless debug level is "off"
            if debug_level != "off":
                handlers['rich'] = self._create_rich_handler()
                logger.debug("RichHandler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize RichHandler: {e}")
        
        try:
            # Wandb integration handler - enabled if wandb config exists and is enabled
            # Only initialize if explicitly enabled in config
            if (hasattr(self.config, 'wandb') and 
                hasattr(self.config.wandb, 'enabled') and 
                self.config.wandb.enabled is True):
                handlers['wandb'] = self._create_wandb_handler()
                logger.debug("WandbHandler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize WandbHandler: {e}")
        
        return handlers
    
    def _create_file_handler(self) -> Any:
        """Create FileHandler instance.
        
        Returns:
            FileHandler instance
            
        Raises:
            ImportError: If FileHandler cannot be imported
        """
        # Import here to avoid circular imports and allow graceful fallback
        from .file_handler import FileHandler
        return FileHandler(self.config)
    
    def _create_svg_handler(self) -> Any:
        """Create SVGHandler instance.
        
        Returns:
            SVGHandler instance
            
        Raises:
            ImportError: If SVGHandler cannot be imported
        """
        # Import here to avoid circular imports and allow graceful fallback
        from ..visualization.svg_handler import SVGHandler
        return SVGHandler(self.config)
    
    def _create_rich_handler(self) -> Any:
        """Create RichHandler instance.
        
        Returns:
            RichHandler instance
            
        Raises:
            ImportError: If RichHandler cannot be imported
        """
        # Import here to avoid circular imports and allow graceful fallback
        from .rich_handler import RichHandler
        return RichHandler(self.config)
    
    def _create_wandb_handler(self) -> Any:
        """Create WandbHandler instance.
        
        Returns:
            WandbHandler instance
            
        Raises:
            ImportError: If WandbHandler cannot be imported
        """
        # Import here to avoid circular imports and allow graceful fallback
        from .wandb_handler import WandbHandler
        return WandbHandler(self.config.wandb)
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log step data through all active handlers.
        
        This method calls the log_step method on all active handlers,
        with error isolation to ensure that failures in one handler
        don't affect others.
        
        Args:
            step_data: Dictionary containing step information with keys:
                - step_num: Step number within episode
                - before_state: State before action
                - after_state: State after action  
                - action: Action taken
                - reward: Reward received
                - info: Additional information including metrics
        """
        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, 'log_step'):
                    handler.log_step(step_data)
            except Exception as e:
                # Log error but continue with other handlers
                logger.warning(f"Handler {handler_name} failed in log_step: {e}")
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Log episode summary through all active handlers.
        
        This method calls the log_episode_summary method on all active handlers,
        with error isolation to ensure that failures in one handler
        don't affect others.
        
        Args:
            summary_data: Dictionary containing episode summary with keys:
                - episode_num: Episode number
                - total_steps: Total steps in episode
                - total_reward: Total reward accumulated
                - final_similarity: Final similarity score
                - success: Whether episode was successful
                - task_id: Task identifier
        """
        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, 'log_episode_summary'):
                    handler.log_episode_summary(summary_data)
            except Exception as e:
                # Log error but continue with other handlers
                logger.warning(f"Handler {handler_name} failed in log_episode_summary: {e}")
    
    def close(self) -> None:
        """Clean shutdown of all handlers.
        
        This method calls the close method on all active handlers to ensure
        proper cleanup of resources like file handles, network connections, etc.
        """
        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, 'close'):
                    handler.close()
                    logger.debug(f"Handler {handler_name} closed successfully")
            except Exception as e:
                logger.warning(f"Handler {handler_name} failed to close: {e}")
        
        logger.info("ExperimentLogger shutdown complete")

