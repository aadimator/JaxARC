"""Simplified Weights & Biases integration handler for JaxARC experiments.

This module provides the WandbHandler class for simplified Weights & Biases
integration. It removes custom retry logic, offline caching, and network
connectivity checks in favor of using official wandb features.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from loguru import logger


class WandbHandler:
    """Simplified Weights & Biases integration handler.
    
    This is a regular Python class (not equinox.Module) that can freely use
    the wandb library, network requests, and standard error handling.
    
    Key simplifications:
    - Uses official wandb offline mode via WANDB_MODE environment variable
    - Relies on wandb's built-in error handling instead of custom retry logic
    - Uses simple wandb.init() and wandb.log() calls
    - Automatically extracts metrics from info['metrics'] dictionary
    """
    
    def __init__(self, wandb_config):
        """Initialize wandb handler with configuration.
        
        Args:
            wandb_config: Wandb configuration object with attributes like
                         enabled, project_name, entity, tags, etc.
        """
        self.config = wandb_config
        self.run = None
        self._initialize_wandb()
    
    def _initialize_wandb(self) -> None:
        """Initialize wandb run with simple configuration."""
        if not self.config.enabled:
            logger.info("Wandb integration disabled in config")
            return
            
        try:
            import wandb
            self._wandb = wandb
            
            # Set offline mode if configured - use official wandb environment variable
            if getattr(self.config, 'offline_mode', False):
                os.environ["WANDB_MODE"] = "offline"
            
            # Simple wandb.init() call - let wandb handle offline mode and errors
            # Convert config attributes to basic types to handle Mock objects in tests
            project_name = str(getattr(self.config, 'project_name', 'jaxarc-test'))
            entity = getattr(self.config, 'entity', None)
            if entity is not None:
                entity = str(entity)
            
            tags = getattr(self.config, 'tags', [])
            if hasattr(tags, '__iter__') and not isinstance(tags, str):
                tags = [str(tag) for tag in tags]
            else:
                tags = [str(tags)] if tags else []
            
            notes = getattr(self.config, 'notes', None)
            if notes is not None:
                notes = str(notes)
                
            group = getattr(self.config, 'group', None)
            if group is not None:
                group = str(group)
                
            job_type = getattr(self.config, 'job_type', None)
            if job_type is not None:
                job_type = str(job_type)
            
            self.run = wandb.init(
                project=project_name,
                entity=entity,
                tags=tags,
                notes=notes,
                group=group,
                job_type=job_type,
                save_code=getattr(self.config, 'save_code', True),
                # Let wandb handle config saving automatically
            )
            
            logger.info(f"Wandb run initialized: {self.run.name} ({self.run.id})")
            
        except ImportError:
            logger.warning("wandb not available - skipping wandb logging")
            self.run = None
        except Exception as e:
            # Simple error handling - just print and continue
            logger.warning(f"wandb initialization failed: {e}")
            self.run = None
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log step metrics to wandb.
        
        Automatically extracts metrics from info['metrics'] if available,
        and adds standard step metrics like reward and step_num.
        
        Args:
            step_data: Step data dictionary containing step information
        """
        if self.run is None:
            return
        
        try:
            # Extract metrics from info['metrics'] if available
            metrics = {}
            if 'info' in step_data and 'metrics' in step_data['info']:
                metrics.update(step_data['info']['metrics'])
            
            # Add standard step metrics
            if 'reward' in step_data:
                metrics['reward'] = step_data['reward']
            if 'step_num' in step_data:
                metrics['step'] = step_data['step_num']
            
            # Add any other scalar metrics from the top level (excluding step_num since we use 'step')
            for key, value in step_data.items():
                if key not in ['info', 'before_state', 'after_state', 'action', 'step_num'] and isinstance(value, (int, float)):
                    metrics[key] = value
            
            # Simple wandb.log() call - let wandb handle retries and errors
            if metrics:
                self.run.log(metrics)
                
        except Exception as e:
            # Simple error handling - just print and continue
            logger.warning(f"wandb step logging failed: {e}")
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Log episode summary to wandb.
        
        Args:
            summary_data: Episode summary data dictionary
        """
        if self.run is None:
            return
        
        try:
            # Log episode-level metrics
            episode_metrics = {}
            
            # Standard episode metrics
            standard_keys = [
                'episode_num', 'total_reward', 'total_steps', 
                'final_similarity', 'success'
            ]
            
            for key in standard_keys:
                if key in summary_data:
                    episode_metrics[key] = summary_data[key]
            
            # Add any other scalar metrics
            for key, value in summary_data.items():
                if key not in standard_keys and isinstance(value, (int, float, bool)):
                    episode_metrics[key] = value
            
            # Log metrics
            if episode_metrics:
                self.run.log(episode_metrics)
            
            # Log summary visualization if available
            if 'summary_svg_path' in summary_data:
                try:
                    import wandb
                    self.run.log({
                        "episode_summary": wandb.Image(summary_data['summary_svg_path'])
                    })
                except Exception as e:
                    logger.warning(f"Failed to log summary image: {e}")
                    
        except Exception as e:
            # Simple error handling - just print and continue
            logger.warning(f"wandb episode logging failed: {e}")
    
    def close(self) -> None:
        """Clean shutdown of wandb run."""
        if self.run is not None:
            try:
                self.run.finish()
                logger.info("Wandb run finished successfully")
            except Exception as e:
                logger.warning(f"wandb finish failed: {e}")
            finally:
                self.run = None