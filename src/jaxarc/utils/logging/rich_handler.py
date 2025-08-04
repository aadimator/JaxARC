from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from ...envs.config import JaxArcConfig


class RichHandlerWrapper:
    """Temporary wrapper around existing rich_display functionality.
    
    This is a temporary implementation that wraps the existing rich_display
    functions to provide a handler interface. This will be replaced with
    a proper RichHandler in a future task.
    
    Note: This is a regular Python class (not equinox.Module) for consistency
    with the handler architecture.
    """
    
    def __init__(self, config: JaxArcConfig):
        """Initialize wrapper with configuration.
        
        Args:
            config: JaxARC configuration object
        """
        self.config = config
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log step data to console using rich display.
        
        Args:
            step_data: Step data dictionary
        """
        # For now, just log basic step information
        # This will be enhanced in a future task
        step_num = step_data.get('step_num', 0)
        reward = step_data.get('reward', 0.0)
        
        # Extract metrics from info if available
        metrics_info = ""
        if 'info' in step_data and 'metrics' in step_data['info']:
            metrics = step_data['info']['metrics']
            if 'similarity' in metrics:
                metrics_info = f", similarity: {metrics['similarity']:.3f}"
        
        logger.info(f"Step {step_num}: reward={reward:.3f}{metrics_info}")
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Log episode summary to console.
        
        Args:
            summary_data: Episode summary data dictionary
        """
        episode_num = summary_data.get('episode_num', 0)
        total_steps = summary_data.get('total_steps', 0)
        total_reward = summary_data.get('total_reward', 0.0)
        final_similarity = summary_data.get('final_similarity', 0.0)
        success = summary_data.get('success', False)
        
        status = "SUCCESS" if success else "INCOMPLETE"
        logger.info(
            f"Episode {episode_num} {status}: "
            f"{total_steps} steps, "
            f"reward: {total_reward:.3f}, "
            f"similarity: {final_similarity:.3f}"
        )
    
    def close(self) -> None:
        """Clean shutdown - no resources to clean up for console output."""
        pass