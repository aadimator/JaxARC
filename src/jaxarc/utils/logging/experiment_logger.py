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
        self._episode_counter = 0  # Sequential episode counter for batched logging
        
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
        from .svg_handler import SVGHandler
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
    
    def log_task_start(self, task_data: Dict[str, Any], show_test: bool = True) -> None:
        """Log task information when an episode starts.
        
        This method calls the log_task_start method on all active handlers,
        with error isolation to ensure that failures in one handler
        don't affect others.
        
        Args:
            task_data: Dictionary containing task information with keys:
                - task_id: Task identifier
                - task_object: The JaxArcTask object
                - episode_num: Episode number
                - num_train_pairs: Number of training pairs
                - num_test_pairs: Number of test pairs
                - task_stats: Additional task statistics
            show_test: Whether to show test examples in visualizations (default: True)
        """
        # Add show_test parameter to task_data for handlers
        enhanced_task_data = {**task_data, 'show_test': show_test}
        
        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, 'log_task_start'):
                    handler.log_task_start(enhanced_task_data)
            except Exception as e:
                # Log error but continue with other handlers
                logger.warning(f"Handler {handler_name} failed in log_task_start: {e}")

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
    
    def log_batch_step(self, batch_data: Dict[str, Any]) -> None:
        """Log data from a batched training step.
        
        This method handles batched training data by aggregating metrics and
        sampling episodes for detailed logging. It provides frequency-based
        control for both aggregation and sampling to minimize performance impact.
        
        Args:
            batch_data: Dictionary containing:
                - update_step: Current training update number
                - episode_returns: Array of episode returns [batch_size]
                - episode_lengths: Array of episode lengths [batch_size]
                - similarity_scores: Array of similarity scores [batch_size]
                - policy_loss: Scalar policy loss
                - value_loss: Scalar value loss
                - gradient_norm: Scalar gradient norm
                - success_mask: Boolean array of episode successes [batch_size]
                - Optional: task_ids, initial_states, final_states for detailed logging
        """
        if not hasattr(self.config, 'logging'):
            logger.warning("No logging configuration found, skipping batch logging")
            return
            
        update_step = batch_data.get("update_step", 0)

        # Log aggregated metrics at specified frequency
        if (self.config.logging.batched_logging_enabled and 
            update_step % self.config.logging.log_frequency == 0):
            
            try:
                aggregated_metrics = self._aggregate_batch_metrics(batch_data)
                
                for handler_name, handler in self.handlers.items():
                    try:
                        if hasattr(handler, 'log_aggregated_metrics'):
                            handler.log_aggregated_metrics(aggregated_metrics, update_step)
                    except Exception as e:
                        logger.warning(f"Handler {handler_name} failed in log_aggregated_metrics: {e}")
            except Exception as e:
                logger.warning(f"Failed to aggregate batch metrics: {e}")

        # Log sampled episode summaries at specified frequency
        if (self.config.logging.sampling_enabled and 
            update_step % self.config.logging.sample_frequency == 0):
            
            try:
                sampled_episodes = self._sample_episodes_from_batch(batch_data)
                
                for episode_data in sampled_episodes:
                    self.log_episode_summary(episode_data)  # Reuse existing method
            except Exception as e:
                logger.warning(f"Failed to sample episodes from batch: {e}")

    def _aggregate_batch_metrics(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate aggregate metrics from batch data using JAX operations.
        
        This method computes statistical aggregations (mean, std, min, max) for
        episode-level metrics and includes scalar training metrics. It handles both
        known metrics with specific configuration flags and unknown metrics generically
        for extensibility with downstream algorithms.
        
        Args:
            batch_data: Dictionary containing batch training data
            
        Returns:
            Dictionary of aggregated metrics with float values
        """
        import jax.numpy as jnp
        
        metrics = {}
        
        # Known episode-level metrics with specific configuration flags
        episode_metrics = {
            'episode_returns': ('reward', self.config.logging.log_aggregated_rewards),
            'similarity_scores': ('similarity', self.config.logging.log_aggregated_similarity),
            'episode_lengths': ('episode_length', self.config.logging.log_episode_lengths),
        }
        
        for key, (prefix, enabled) in episode_metrics.items():
            if key in batch_data and enabled:
                values = batch_data[key]
                # Skip empty arrays to avoid JAX errors
                if jnp.asarray(values).size == 0:
                    logger.debug(f"Skipping empty metric '{key}'")
                    continue
                metrics.update({
                    f'{prefix}_mean': float(jnp.mean(values)),
                    f'{prefix}_std': float(jnp.std(values)),
                    f'{prefix}_max': float(jnp.max(values)),
                    f'{prefix}_min': float(jnp.min(values))
                })
        
        # Success rate calculation (special case - boolean array)
        if 'success_mask' in batch_data and self.config.logging.log_success_rates:
            success_mask = batch_data['success_mask']
            # Skip empty arrays
            if jnp.asarray(success_mask).size == 0:
                logger.debug("Skipping empty success_mask")
            else:
                metrics['success_rate'] = float(jnp.mean(success_mask))
        
        # Known scalar training metrics with specific configuration flags
        scalar_metrics = {
            'policy_loss': self.config.logging.log_loss_metrics,
            'value_loss': self.config.logging.log_loss_metrics,
            'gradient_norm': self.config.logging.log_gradient_norms,
            'entropy': self.config.logging.log_loss_metrics,
            'explained_variance': self.config.logging.log_loss_metrics,
            'learning_rate': self.config.logging.log_loss_metrics,
        }
        
        for key, enabled in scalar_metrics.items():
            if key in batch_data and enabled:
                metrics[key] = float(batch_data[key])
        
        # Generic handling for unknown metrics (extensibility for downstream algorithms)
        # Skip known keys and metadata keys to avoid double processing
        known_keys = set(episode_metrics.keys()) | set(scalar_metrics.keys()) | {
            'success_mask', 'update_step', 'task_ids', 'initial_states', 'final_states'
        }
        
        for key, value in batch_data.items():
            if key in known_keys:
                continue
                
            try:
                # Convert to JAX array for shape inspection
                jax_value = jnp.asarray(value)
                
                # Skip empty arrays
                if jax_value.size == 0:
                    logger.debug(f"Skipping empty metric '{key}'")
                    continue
                
                # Determine if this is an array metric (needs aggregation) or scalar metric
                if jax_value.ndim == 0:
                    # Scalar metric - pass through directly
                    metrics[key] = float(jax_value)
                elif jax_value.ndim == 1 and jax_value.shape[0] > 1:
                    # Array metric - apply aggregation
                    metrics.update({
                        f'{key}_mean': float(jnp.mean(jax_value)),
                        f'{key}_std': float(jnp.std(jax_value)),
                        f'{key}_max': float(jnp.max(jax_value)),
                        f'{key}_min': float(jnp.min(jax_value))
                    })
                else:
                    # Single element array or other - treat as scalar
                    metrics[key] = float(jnp.mean(jax_value))
                    
            except Exception as e:
                # Skip metrics that can't be converted to JAX arrays
                logger.debug(f"Skipping metric '{key}' due to conversion error: {e}")
                continue
        
        return metrics

    def _sample_episodes_from_batch(self, batch_data: Dict[str, Any]) -> list[Dict[str, Any]]:
        """Extract sample of episode summaries from batch for detailed logging.
        
        This method performs deterministic sampling based on the update step for
        reproducibility. It reconstructs episode summary data compatible with
        the existing log_episode_summary method.
        
        Note: This focuses on episode-level data rather than step-by-step data,
        as intermediate states are not typically stored in batched training for
        performance reasons.
        
        Args:
            batch_data: Dictionary containing batch training data
            
        Returns:
            List of episode summary dictionaries for sampled episodes
        """
        import jax
        import jax.numpy as jnp
        from typing import List
        
        num_samples = self.config.logging.num_samples
        batch_size = batch_data['episode_returns'].shape[0]
        
        # Sample all if requested samples >= batch size
        if num_samples >= batch_size:
            sample_indices = jnp.arange(batch_size)
        else:
            # Deterministic sampling based on update step for reproducibility
            key = jax.random.PRNGKey(batch_data.get("update_step", 0))
            sample_indices = jax.random.choice(
                key, batch_size, shape=(num_samples,), replace=False
            )
        
        sampled_episodes = []
        for i in sample_indices:
            # Reconstruct episode summary data for existing log_episode_summary method
            # Use sequential episode counter to avoid exceeding max_episodes_per_run limits
            self._episode_counter += 1
            unique_episode_num = self._episode_counter
            episode_summary = {
                "episode_num": unique_episode_num,
                "total_reward": float(batch_data['episode_returns'][i]),
                "total_steps": int(batch_data['episode_lengths'][i]) if 'episode_lengths' in batch_data else 0,
                "final_similarity": float(batch_data['similarity_scores'][i]) if 'similarity_scores' in batch_data else 0.0,
                "success": bool(batch_data['success_mask'][i]) if 'success_mask' in batch_data else False,
                "environment_id": int(i),
                "task_id": batch_data.get('task_ids', [f"batch_task_{i}"])[i] if 'task_ids' in batch_data else f"batch_task_{i}",
                
                # Optional: Include final states if available for SVG visualization
                "initial_state": batch_data.get('initial_states', [None])[i] if 'initial_states' in batch_data else None,
                "final_state": batch_data.get('final_states', [None])[i] if 'final_states' in batch_data else None,
            }
            sampled_episodes.append(episode_summary)
        
        return sampled_episodes

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

