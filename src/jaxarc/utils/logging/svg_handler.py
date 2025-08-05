"""SVG visualization generation handler for JaxARC logging.

This module provides the SVGHandler class that consolidates SVG generation logic
from rl_visualization.py and episode_visualization.py into a single handler for
the simplified logging architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ...types import Grid
from ..serialization_utils import serialize_jax_array
from ..visualization.episode_manager import EpisodeConfig, EpisodeManager
from ..visualization.episode_visualization import draw_enhanced_episode_summary_svg
from ..visualization.rl_visualization import (
    draw_rl_step_svg_enhanced,
    get_operation_display_name,
)
from ..visualization.utils import detect_changed_cells


class SVGHandler:
    """SVG visualization generation handler.
    
    Note: Regular Python class that can freely use file I/O,
    string manipulation, and SVG generation libraries.
    
    This handler consolidates SVG generation logic from the visualization
    modules and provides a clean interface for the ExperimentLogger.
    """
    
    def __init__(self, config: Any):
        """Initialize SVG handler with configuration.
        
        Args:
            config: JaxArcConfig containing visualization and storage settings
        """
        self.config = config
        
        # Initialize episode manager for file path management
        # Create episode config from storage config
        episode_config = EpisodeConfig(
            base_output_dir=getattr(config.storage, 'base_output_dir', 'outputs'),
            run_name=getattr(config.storage, 'run_name', None),
            max_episodes_per_run=getattr(config.storage, 'max_episodes_per_run', 1000),
            max_storage_gb=getattr(config.storage, 'max_storage_gb', 10.0),
            cleanup_policy=getattr(config.storage, 'cleanup_policy', 'size_based'),
        )
        
        self.episode_manager = EpisodeManager(episode_config)
        
        # Track current episode for file management
        self.current_episode_num: int | None = None
        self.current_run_started = False
        
        logger.debug("SVGHandler initialized")
    
    def start_run(self, run_name: str | None = None) -> None:
        """Start a new run for SVG generation.
        
        Args:
            run_name: Optional custom run name
        """
        try:
            self.episode_manager.start_new_run(run_name)
            self.current_run_started = True
            logger.info(f"SVGHandler started new run: {self.episode_manager.current_run_name}")
        except Exception as e:
            logger.error(f"Failed to start SVG run: {e}")
    
    def start_episode(self, episode_num: int) -> None:
        """Start a new episode for SVG generation.
        
        Args:
            episode_num: Episode number
        """
        try:
            if not self.current_run_started:
                self.start_run()
            
            self.episode_manager.start_new_episode(episode_num)
            self.current_episode_num = episode_num
            logger.debug(f"SVGHandler started episode {episode_num}")
        except Exception as e:
            logger.error(f"Failed to start SVG episode {episode_num}: {e}")
    
    def log_step(self, step_data: dict[str, Any]) -> None:
        """Generate and save step visualization.
        
        Args:
            step_data: Dictionary containing step information including:
                - step_num: Step number
                - before_state: Environment state before action
                - after_state: Environment state after action
                - action: Action taken
                - reward: Reward received
                - info: Additional information dictionary
        """
        # Check if step visualization is enabled
        if not self._should_generate_step_svg():
            return
        
        try:
            step_num = step_data.get('step_num', 0)
            episode_num = step_data.get('episode_num')
            
            # Ensure episode is started
            if episode_num is not None and self.current_episode_num != episode_num:
                self.start_episode(episode_num)
            elif self.current_episode_num is None:
                # Default to episode 0 if not specified
                self.start_episode(0)
            
            # Extract required data
            before_state = step_data.get('before_state')
            after_state = step_data.get('after_state')
            action = step_data.get('action')
            reward = step_data.get('reward', 0.0)
            info = step_data.get('info', {})
            
            if before_state is None or after_state is None or action is None:
                logger.warning(f"Missing required data for step {step_num} SVG generation")
                return
            
            # Create Grid objects from states
            before_grid = self._extract_grid_from_state(before_state)
            after_grid = self._extract_grid_from_state(after_state)
            
            if before_grid is None or after_grid is None:
                logger.warning(f"Could not extract grids for step {step_num}")
                return
            
            # Detect changed cells
            changed_cells = detect_changed_cells(before_grid, after_grid)
            
            # Get operation name
            operation_id = self._extract_operation_id(action)
            operation_name = get_operation_display_name(operation_id, action)
            
            # Extract additional context
            task_id = step_data.get('task_id', '')
            task_pair_index = step_data.get('task_pair_index', 0)
            total_task_pairs = step_data.get('total_task_pairs', 1)
            
            # Filter info dictionary to only include known visualization keys
            # This makes SVGHandler ignore unknown keys gracefully
            filtered_info = self._filter_info_for_visualization(info)
            
            # Generate SVG content
            svg_content = draw_rl_step_svg_enhanced(
                before_grid=before_grid,
                after_grid=after_grid,
                action=action,
                reward=reward,
                info=filtered_info,
                step_num=step_num,
                operation_name=operation_name,
                changed_cells=changed_cells,
                config=self.config,
                task_id=task_id,
                task_pair_index=task_pair_index,
                total_task_pairs=total_task_pairs,
            )
            
            # Save SVG file
            svg_path = self.episode_manager.get_step_path(step_num, "svg")
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Path.open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            logger.debug(f"Saved step {step_num} SVG to {svg_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate step {step_data.get('step_num', 'unknown')} SVG: {e}")
    
    def log_episode_summary(self, summary_data: dict[str, Any]) -> None:
        """Generate and save episode summary visualization.
        
        Args:
            summary_data: Dictionary containing episode summary information including:
                - episode_num: Episode number
                - total_steps: Total number of steps
                - total_reward: Total reward received
                - final_similarity: Final similarity score
                - success: Whether episode was successful
                - task_id: Task identifier
                - step_data: List of step data for visualization
        """
        # Check if episode summary visualization is enabled
        if not self._should_generate_episode_summary():
            return
        
        try:
            episode_num = summary_data.get('episode_num', 0)
            
            # Ensure episode is started
            if self.current_episode_num != episode_num:
                self.start_episode(episode_num)
            
            # Extract step data if available
            step_data = summary_data.get('step_data', [])
            
            # Convert dictionary to object for compatibility with visualization function
            class SummaryDataWrapper:
                def __init__(self, data_dict):
                    for key, value in data_dict.items():
                        setattr(self, key, value)
            
            summary_obj = SummaryDataWrapper(summary_data)
            
            # Generate episode summary SVG
            svg_content = draw_enhanced_episode_summary_svg(
                summary_data=summary_obj,
                step_data=step_data,
                config=self.config,
            )
            
            # Save summary SVG
            summary_path = self.episode_manager.get_episode_summary_path("svg")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Path.open(summary_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            logger.debug(f"Saved episode {episode_num} summary SVG to {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate episode summary SVG: {e}")
    
    def close(self) -> None:
        """Clean shutdown of SVG handler."""
        try:
            # Perform any cleanup if needed
            logger.debug("SVGHandler closed")
        except Exception as e:
            logger.error(f"Error during SVGHandler shutdown: {e}")
    
    def _should_generate_step_svg(self) -> bool:
        """Check if step SVG generation is enabled based on configuration."""
        try:
            # Check visualization config
            if hasattr(self.config, 'visualization'):
                viz_config = self.config.visualization
                if not getattr(viz_config, 'enabled', True):
                    return False
                
                level = getattr(viz_config, 'level', 'standard')
                if level == 'off':
                    return False
                
                # Only generate step SVGs for verbose and full levels
                if level in ['verbose', 'full']:
                    return getattr(viz_config, 'step_visualizations', True)
                else:
                    # For standard, minimal levels, don't generate step SVGs
                    return False
            
            # Check environment debug level as fallback
            if hasattr(self.config, 'environment'):
                debug_level = getattr(self.config.environment, 'debug_level', 'standard')
                return debug_level in ['verbose', 'research']
            
            return False
        except Exception:
            return False
    
    def _should_generate_episode_summary(self) -> bool:
        """Check if episode summary generation is enabled based on configuration."""
        try:
            # Check visualization config
            if hasattr(self.config, 'visualization'):
                viz_config = self.config.visualization
                if not getattr(viz_config, 'enabled', True):
                    return False
                
                level = getattr(viz_config, 'level', 'standard')
                if level == 'off':
                    return False
                
                return getattr(viz_config, 'episode_summaries', True)
            
            # Check environment debug level as fallback
            if hasattr(self.config, 'environment'):
                debug_level = getattr(self.config.environment, 'debug_level', 'standard')
                return debug_level != 'off'
            
            return True
        except Exception:
            return True
    
    def _extract_grid_from_state(self, state: Any) -> Grid | None:
        """Extract Grid object from environment state.
        
        Args:
            state: Environment state object
            
        Returns:
            Grid object or None if extraction fails
        """
        try:
            # Handle different state formats
            if hasattr(state, 'working_grid') and hasattr(state, 'working_grid_mask'):
                # ArcEnvState format
                return Grid(
                    data=np.asarray(state.working_grid),
                    mask=np.asarray(state.working_grid_mask),
                )
            elif hasattr(state, 'data') and hasattr(state, 'mask'):
                # Already a Grid object
                return state
            elif isinstance(state, (np.ndarray, list)):
                # Raw grid data
                return Grid(data=np.asarray(state), mask=None)
            else:
                logger.warning(f"Unknown state format: {type(state)}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract grid from state: {e}")
            return None
    
    def _extract_operation_id(self, action: Any) -> int:
        """Extract operation ID from action.
        
        Args:
            action: Action object or dictionary
            
        Returns:
            Operation ID as integer
        """
        try:
            # Handle structured action objects
            if hasattr(action, 'operation'):
                op_val = action.operation
                return int(op_val) if hasattr(op_val, 'item') else int(op_val)
            
            # Handle dictionary format
            elif isinstance(action, dict) and 'operation' in action:
                op_val = action['operation']
                return int(op_val) if hasattr(op_val, 'item') else int(op_val)
            
            else:
                logger.warning(f"Could not extract operation from action: {type(action)}")
                return 0
        except Exception as e:
            logger.error(f"Failed to extract operation ID: {e}")
            return 0
    
    def _filter_info_for_visualization(self, info: dict[str, Any]) -> dict[str, Any]:
        """Filter info dictionary to only include keys relevant for visualization.
        
        This method makes SVGHandler ignore unknown keys gracefully by only
        passing through keys that are known to be used by the visualization functions.
        
        Args:
            info: Original info dictionary
            
        Returns:
            Filtered info dictionary with only visualization-relevant keys
        """
        # Known keys used by visualization functions
        known_viz_keys = {
            'success', 'similarity', 'similarity_improvement', 'step_count',
            'is_control_operation', 'operation_type', 'episode_mode',
            'current_pair_index', 'metrics'  # Include metrics for potential future use
        }
        
        # Filter to only include known keys, gracefully ignoring unknown ones
        filtered_info = {}
        for key, value in info.items():
            if key in known_viz_keys:
                filtered_info[key] = value
            # Unknown keys are silently ignored (graceful handling)
        
        return filtered_info
    
    def get_current_run_info(self) -> dict[str, Any]:
        """Get information about the current run.
        
        Returns:
            Dictionary with current run information
        """
        return self.episode_manager.get_current_run_info()