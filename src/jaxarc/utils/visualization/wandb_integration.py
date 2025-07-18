"""Weights & Biases integration for JaxARC visualization and experiment tracking."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chex

logger = logging.getLogger(__name__)


@chex.dataclass
class WandbConfig:
    """Configuration for Weights & Biases integration.
    
    This configuration controls how JaxARC integrates with wandb for experiment
    tracking, including what gets logged, how often, and in what format.
    """
    
    # Core wandb settings
    enabled: bool = False
    project_name: str = "jaxarc-experiments"
    entity: Optional[str] = None
    
    # Run configuration
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    
    # Logging frequency and format
    log_frequency: int = 10  # Log every N steps
    image_format: str = "png"  # "png", "svg", "both"
    max_image_size: tuple[int, int] = (800, 600)
    
    # Advanced logging options
    log_gradients: bool = False
    log_model_topology: bool = False
    log_system_metrics: bool = True
    
    # Offline and error handling
    offline_mode: bool = False
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Storage settings
    save_code: bool = True
    save_config: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.image_format not in {"png", "svg", "both"}:
            raise ValueError(f"Invalid image_format: {self.image_format}. Must be 'png', 'svg', or 'both'")
        
        if self.log_frequency <= 0:
            raise ValueError(f"log_frequency must be positive, got {self.log_frequency}")
        
        if len(self.max_image_size) != 2 or any(s <= 0 for s in self.max_image_size):
            raise ValueError(f"max_image_size must be tuple of two positive integers, got {self.max_image_size}")


class WandbIntegration:
    """Weights & Biases integration for JaxARC.
    
    This class provides a clean interface for logging JaxARC experiments to wandb,
    with graceful fallback when wandb is unavailable and comprehensive error handling.
    """
    
    def __init__(self, config: WandbConfig) -> None:
        """Initialize wandb integration.
        
        Args:
            config: Configuration for wandb integration
        """
        self.config = config
        self.run = None
        self._wandb_available = False
        self._step_count = 0
        self._last_log_step = -1
        
        # Try to import wandb
        self._initialize_wandb()
    
    def _initialize_wandb(self) -> None:
        """Initialize wandb library if available."""
        if not self.config.enabled:
            logger.info("Wandb integration disabled in config")
            return
        
        try:
            import wandb
            self._wandb = wandb
            self._wandb_available = True
            logger.info("Wandb successfully imported and available")
        except ImportError:
            logger.warning(
                "wandb not available. Install with 'pip install wandb' to enable experiment tracking. "
                "Falling back to local logging only."
            )
            self._wandb_available = False
    
    def initialize_run(
        self,
        experiment_config: Dict[str, Any],
        run_name: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> bool:
        """Initialize wandb run with experiment configuration.
        
        Args:
            experiment_config: Complete experiment configuration to log
            run_name: Optional name for the run
            run_id: Optional run ID for resuming runs
            
        Returns:
            True if wandb run was successfully initialized, False otherwise
        """
        if not self._wandb_available:
            logger.info("Wandb not available, skipping run initialization")
            return False
        
        try:
            # Set offline mode if configured
            if self.config.offline_mode:
                os.environ["WANDB_MODE"] = "offline"
            
            # Initialize the run
            self.run = self._wandb.init(
                project=self.config.project_name,
                entity=self.config.entity,
                name=run_name,
                id=run_id,
                tags=self.config.tags,
                notes=self.config.notes,
                group=self.config.group,
                job_type=self.config.job_type,
                config=experiment_config if self.config.save_config else None,
                save_code=self.config.save_code,
                resume="allow" if run_id else None
            )
            
            logger.info(f"Wandb run initialized: {self.run.name} ({self.run.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize wandb run: {e}")
            self._wandb_available = False
            return False
    
    def log_step(
        self,
        step_num: int,
        metrics: Dict[str, Union[int, float]],
        images: Optional[Dict[str, Any]] = None,
        force_log: bool = False
    ) -> bool:
        """Log step metrics and visualizations.
        
        Args:
            step_num: Current step number
            metrics: Dictionary of metrics to log
            images: Optional dictionary of images to log
            force_log: Force logging even if not at log_frequency interval
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not self._wandb_available or self.run is None:
            return False
        
        # Check if we should log this step
        if not force_log and (step_num - self._last_log_step) < self.config.log_frequency:
            return True
        
        try:
            log_data = {"step": step_num, **metrics}
            
            # Add images if provided
            if images:
                processed_images = self._process_images(images)
                log_data.update(processed_images)
            
            # Log to wandb with retry logic
            success = self._log_with_retry(log_data, step_num)
            
            if success:
                self._last_log_step = step_num
            
            return success
            
        except Exception as e:
            logger.error(f"Error logging step {step_num}: {e}")
            return False
    
    def log_episode_summary(
        self,
        episode_num: int,
        summary_data: Dict[str, Any],
        summary_image: Optional[Any] = None
    ) -> bool:
        """Log episode summary with key metrics.
        
        Args:
            episode_num: Episode number
            summary_data: Dictionary containing episode summary metrics
            summary_image: Optional summary visualization
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not self._wandb_available or self.run is None:
            return False
        
        try:
            log_data = {
                "episode": episode_num,
                **summary_data
            }
            
            # Add summary image if provided
            if summary_image is not None:
                processed_image = self._process_single_image(summary_image, "episode_summary")
                if processed_image:
                    log_data["episode_summary_image"] = processed_image
            
            # Always force log episode summaries
            return self._log_with_retry(log_data, episode_num)
            
        except Exception as e:
            logger.error(f"Error logging episode {episode_num} summary: {e}")
            return False
    
    def log_config_update(self, config_update: Dict[str, Any]) -> bool:
        """Log configuration updates during the run.
        
        Args:
            config_update: Dictionary of configuration changes
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not self._wandb_available or self.run is None:
            return False
        
        try:
            self.run.config.update(config_update)
            return True
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False
    
    def finish_run(self) -> None:
        """Properly close wandb run."""
        if self.run is not None:
            try:
                self.run.finish()
                logger.info("Wandb run finished successfully")
            except Exception as e:
                logger.error(f"Error finishing wandb run: {e}")
            finally:
                self.run = None
    
    def _process_images(self, images: Dict[str, Any]) -> Dict[str, Any]:
        """Process images for wandb logging.
        
        Args:
            images: Dictionary of images to process
            
        Returns:
            Dictionary of processed images ready for wandb
        """
        processed = {}
        
        for name, image in images.items():
            processed_image = self._process_single_image(image, name)
            if processed_image:
                processed[name] = processed_image
        
        return processed
    
    def _process_single_image(self, image: Any, name: str) -> Optional[Any]:
        """Process a single image for wandb logging.
        
        Args:
            image: Image to process (can be PIL Image, numpy array, or file path)
            name: Name/identifier for the image
            
        Returns:
            Processed image ready for wandb, or None if processing failed
        """
        try:
            # Handle different image types
            if isinstance(image, (str, Path)):
                # File path - let wandb handle it
                return self._wandb.Image(str(image), caption=name)
            elif hasattr(image, 'save'):
                # PIL Image
                return self._wandb.Image(image, caption=name)
            elif hasattr(image, 'shape'):
                # Numpy array
                return self._wandb.Image(image, caption=name)
            else:
                logger.warning(f"Unsupported image type for {name}: {type(image)}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing image {name}: {e}")
            return None
    
    def _log_with_retry(self, data: Dict[str, Any], step: Optional[int] = None) -> bool:
        """Log data with retry logic for network errors.
        
        Args:
            data: Data to log
            step: Optional step number
            
        Returns:
            True if logging was successful, False otherwise
        """
        for attempt in range(self.config.retry_attempts):
            try:
                if step is not None:
                    self.run.log(data, step=step)
                else:
                    self.run.log(data)
                return True
                
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(f"Wandb log attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All wandb log attempts failed: {e}")
                    return False
        
        return False
    
    @property
    def is_available(self) -> bool:
        """Check if wandb is available and configured."""
        return self._wandb_available and self.config.enabled
    
    @property
    def is_initialized(self) -> bool:
        """Check if wandb run is initialized."""
        return self.run is not None
    
    @property
    def run_url(self) -> Optional[str]:
        """Get the URL of the current wandb run."""
        if self.run is not None:
            return self.run.get_url()
        return None
    
    @property
    def run_id(self) -> Optional[str]:
        """Get the ID of the current wandb run."""
        if self.run is not None:
            return self.run.id
        return None
    
    @property
    def run_name(self) -> Optional[str]:
        """Get the name of the current wandb run."""
        if self.run is not None:
            return self.run.name
        return None


def create_wandb_config(
    enabled: bool = False,
    project_name: str = "jaxarc-experiments",
    **kwargs: Any
) -> WandbConfig:
    """Create a WandbConfig with common defaults.
    
    Args:
        enabled: Whether to enable wandb integration
        project_name: Name of the wandb project
        **kwargs: Additional configuration options
        
    Returns:
        Configured WandbConfig instance
    """
    return WandbConfig(
        enabled=enabled,
        project_name=project_name,
        **kwargs
    )


def create_research_wandb_config(
    project_name: str = "jaxarc-research",
    entity: Optional[str] = None
) -> WandbConfig:
    """Create a WandbConfig optimized for research use.
    
    Args:
        project_name: Name of the wandb project
        entity: Wandb entity (username or team)
        
    Returns:
        Research-optimized WandbConfig instance
    """
    return WandbConfig(
        enabled=True,
        project_name=project_name,
        entity=entity,
        log_frequency=5,  # More frequent logging for research
        image_format="both",  # Log both PNG and SVG
        log_system_metrics=True,
        save_code=True,
        save_config=True,
        tags=["research", "jaxarc"]
    )


def create_development_wandb_config(
    project_name: str = "jaxarc-dev"
) -> WandbConfig:
    """Create a WandbConfig optimized for development use.
    
    Args:
        project_name: Name of the wandb project
        
    Returns:
        Development-optimized WandbConfig instance
    """
    return WandbConfig(
        enabled=True,
        project_name=project_name,
        log_frequency=20,  # Less frequent logging for development
        image_format="png",  # Faster PNG only
        log_system_metrics=False,
        save_code=False,  # Don't save code during development
        offline_mode=True,  # Work offline during development
        tags=["development", "jaxarc"]
    )