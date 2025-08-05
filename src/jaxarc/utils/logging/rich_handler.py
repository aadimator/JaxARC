"""Rich console output handler for JaxARC logging system.

This module provides the RichHandler class that handles console output using
the Rich library for enhanced terminal formatting and styling.
"""

from __future__ import annotations

from typing import Any, Dict

from rich.console import Console

from ..visualization.rich_display import (
    log_grid_to_console,
    visualize_grid_rich,
    visualize_task_pair_rich,
)


class RichHandler:
    """Console output handler using Rich library.
    
    Note: Regular Python class that can freely use Rich library,
    console I/O, and terminal formatting.
    """
    
    def __init__(self, config):
        """Initialize RichHandler with configuration.
        
        Args:
            config: Configuration object with debug level settings
        """
        self.config = config
        self.console = self._setup_rich_console()
    
    def _setup_rich_console(self) -> Console:
        """Initialize Rich console with configuration."""
        return Console()
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Display step information to console.
        
        Args:
            step_data: Dictionary containing step information including:
                - step_num: Current step number
                - before_state: State before action
                - after_state: State after action
                - action: Action taken
                - reward: Reward received
                - info: Additional information dictionary
        """
        # Only display step info for verbose and full debug levels
        if hasattr(self.config, 'environment') and hasattr(self.config.environment, 'debug_level'):
            debug_level = self.config.environment.debug_level
        elif hasattr(self.config, 'debug_level'):
            debug_level = self.config.debug_level
        else:
            debug_level = "standard"
            
        if debug_level in ["verbose", "research"]:
            self._display_step_info(step_data)
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Display episode summary to console.
        
        Args:
            summary_data: Dictionary containing episode summary information including:
                - episode_num: Episode number
                - total_steps: Total steps in episode
                - total_reward: Total reward accumulated
                - final_similarity: Final similarity score
                - success: Whether episode was successful
        """
        self._display_episode_summary(summary_data)
    
    def close(self) -> None:
        """Clean shutdown."""
        pass
    
    def _display_step_info(self, step_data: Dict[str, Any]) -> None:
        """Display step information using Rich formatting.
        
        Args:
            step_data: Step data dictionary
        """
        step_num = step_data.get('step_num', 0)
        reward = step_data.get('reward', 0.0)
        info = step_data.get('info', {})
        
        # Display step header
        self.console.print(f"\n[bold blue]Step {step_num}[/bold blue]")
        
        # Display reward
        reward_color = "green" if reward > 0 else "red" if reward < 0 else "yellow"
        self.console.print(f"Reward: [{reward_color}]{reward:.3f}[/{reward_color}]")
        
        # Display grids if available
        before_state = step_data.get('before_state')
        after_state = step_data.get('after_state')
        
        if before_state is not None and hasattr(before_state, 'grid'):
            # Display before and after grids side by side
            if after_state is not None and hasattr(after_state, 'grid'):
                visualize_task_pair_rich(
                    input_grid=before_state.grid,
                    output_grid=after_state.grid,
                    title=f"Step {step_num}",
                    console=self.console
                )
            else:
                # Display only before grid
                grid_table = visualize_grid_rich(
                    before_state.grid,
                    title=f"Step {step_num} - Before",
                    border_style="input"
                )
                self.console.print(grid_table)
        
        # Display metrics if available (automatic extraction from info['metrics'])
        if 'metrics' in info and isinstance(info['metrics'], dict):
            self.console.print("[bold]Metrics:[/bold]")
            for key, value in info['metrics'].items():
                if isinstance(value, (int, float)):
                    self.console.print(f"  {key}: {value:.3f}")
                else:
                    self.console.print(f"  {key}: {value}")
        
        # Display other known info keys, gracefully ignoring unknown ones
        known_display_keys = {'success', 'similarity_improvement', 'is_control_operation'}
        for key in known_display_keys:
            if key in info:
                value = info[key]
                if isinstance(value, (int, float)):
                    self.console.print(f"[bold]{key}:[/bold] {value:.3f}")
                else:
                    self.console.print(f"[bold]{key}:[/bold] {value}")
        # Unknown keys in info are silently ignored (graceful handling)
        
        # Display action information if available
        action = step_data.get('action')
        if action is not None:
            self.console.print(f"[bold]Action:[/bold] {action}")
    
    def _display_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Display episode summary using Rich formatting.
        
        Args:
            summary_data: Episode summary data dictionary
        """
        episode_num = summary_data.get('episode_num', 0)
        total_steps = summary_data.get('total_steps', 0)
        total_reward = summary_data.get('total_reward', 0.0)
        final_similarity = summary_data.get('final_similarity', 0.0)
        success = summary_data.get('success', False)
        
        # Display episode header
        self.console.print(f"\n[bold magenta]Episode {episode_num} Summary[/bold magenta]")
        self.console.rule()
        
        # Display basic stats
        self.console.print(f"Total Steps: [bold]{total_steps}[/bold]")
        
        # Display reward with color coding
        reward_color = "green" if total_reward > 0 else "red" if total_reward < 0 else "yellow"
        self.console.print(f"Total Reward: [{reward_color}]{total_reward:.3f}[/{reward_color}]")
        
        # Display similarity with color coding
        similarity_color = "green" if final_similarity > 0.8 else "yellow" if final_similarity > 0.5 else "red"
        self.console.print(f"Final Similarity: [{similarity_color}]{final_similarity:.3f}[/{similarity_color}]")
        
        # Display success status
        success_color = "green" if success else "red"
        success_text = "SUCCESS" if success else "FAILED"
        self.console.print(f"Status: [{success_color}]{success_text}[/{success_color}]")
        
        # Display task information if available
        task_id = summary_data.get('task_id')
        if task_id:
            self.console.print(f"Task ID: [bold]{task_id}[/bold]")
        
        self.console.rule()