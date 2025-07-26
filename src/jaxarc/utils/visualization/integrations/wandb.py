"""Weights & Biases integration for JaxARC visualization and experiment tracking."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.error import URLError

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
    max_retry_delay: float = 30.0  # Maximum delay between retries
    network_timeout: float = 30.0  # Network timeout in seconds
    auto_offline_on_error: bool = (
        True  # Automatically switch to offline mode on network errors
    )

    # Offline caching settings
    offline_cache_dir: Optional[str] = (
        None  # Directory for offline cache (auto-generated if None)
    )
    max_cache_size_gb: float = 1.0  # Maximum cache size in GB
    cache_compression: bool = True  # Compress cached data

    # Sync settings
    auto_sync_on_reconnect: bool = True  # Automatically sync when network is restored
    sync_batch_size: int = 50  # Number of entries to sync at once

    # Storage settings
    save_code: bool = True
    save_config: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.image_format not in {"png", "svg", "both"}:
            raise ValueError(
                f"Invalid image_format: {self.image_format}. Must be 'png', 'svg', or 'both'"
            )

        if self.log_frequency <= 0:
            raise ValueError(
                f"log_frequency must be positive, got {self.log_frequency}"
            )

        if len(self.max_image_size) != 2 or any(s <= 0 for s in self.max_image_size):
            raise ValueError(
                f"max_image_size must be tuple of two positive integers, got {self.max_image_size}"
            )


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
        self._offline_mode_active = False
        self._offline_cache_dir = None
        self._cached_entries = []

        # Initialize offline cache if needed
        self._setup_offline_cache()

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
        run_id: Optional[str] = None,
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

            # Generate enhanced tags and organization
            enhanced_tags = self._generate_experiment_tags(experiment_config)
            enhanced_group = self._generate_experiment_group(experiment_config)
            enhanced_job_type = self._generate_job_type(experiment_config)
            enhanced_name = self._generate_run_name(experiment_config, run_name)

            # Initialize the run
            self.run = self._wandb.init(
                project=self.config.project_name,
                entity=self.config.entity,
                name=enhanced_name,
                id=run_id,
                tags=enhanced_tags,
                notes=self.config.notes,
                group=enhanced_group,
                job_type=enhanced_job_type,
                config=experiment_config if self.config.save_config else None,
                save_code=self.config.save_code,
                resume="allow" if run_id else None,
            )

            logger.info(f"Wandb run initialized: {self.run.name} ({self.run.id})")
            logger.info(f"Tags: {enhanced_tags}")
            logger.info(f"Group: {enhanced_group}")
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
        force_log: bool = False,
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
        if (
            not force_log
            and (step_num - self._last_log_step) < self.config.log_frequency
        ):
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
        summary_image: Optional[Any] = None,
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
            log_data = {"episode": episode_num, **summary_data}

            # Add summary image if provided
            if summary_image is not None:
                processed_image = self._process_single_image(
                    summary_image, "episode_summary"
                )
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
        """Process a single image for wandb logging with optimization.

        Args:
            image: Image to process (can be PIL Image, numpy array, or file path)
            name: Name/identifier for the image

        Returns:
            Processed image ready for wandb, or None if processing failed
        """
        try:
            # Import PIL for image processing
            try:
                import numpy as np
                from PIL import Image as PILImage
            except ImportError:
                logger.warning("PIL not available, using basic image processing")
                return self._process_basic_image(image, name)

            # Convert input to PIL Image for processing
            pil_image = self._convert_to_pil(image, PILImage, np)
            if pil_image is None:
                return None

            # Apply image optimizations
            optimized_image = self._optimize_image(pil_image, PILImage)

            # Create wandb image with proper format
            return self._create_wandb_image(optimized_image, name)

        except Exception as e:
            logger.error(f"Error processing image {name}: {e}")
            # Fallback to basic processing
            return self._process_basic_image(image, name)

    def _process_basic_image(self, image: Any, name: str) -> Optional[Any]:
        """Basic image processing fallback when PIL is not available.

        Args:
            image: Image to process
            name: Name/identifier for the image

        Returns:
            Basic wandb image or None if processing failed
        """
        try:
            # Handle different image types without optimization
            if isinstance(image, (str, Path)):
                return self._wandb.Image(str(image), caption=name)
            if hasattr(image, "save"):
                # PIL Image
                return self._wandb.Image(image, caption=name)
            if hasattr(image, "shape"):
                # NumPy array
                return self._wandb.Image(image, caption=name)
            logger.warning(f"Unsupported image type for {name}: {type(image)}")
            return None
        except Exception as e:
            logger.error(f"Error in basic image processing for {name}: {e}")
            return None

    def _convert_to_pil(self, image: Any, PILImage: Any, np: Any) -> Optional[Any]:
        """Convert various image formats to PIL Image.

        Args:
            image: Input image in various formats
            PILImage: PIL Image class
            np: numpy module

        Returns:
            PIL Image or None if conversion failed
        """
        try:
            if isinstance(image, (str, Path)):
                # Load from file path
                return PILImage.open(str(image))
            if hasattr(image, "save") and hasattr(image, "size"):
                # Already a PIL Image
                return image
            if hasattr(image, "shape"):
                # NumPy array
                if len(image.shape) == 2:
                    # Grayscale
                    return PILImage.fromarray((image * 255).astype(np.uint8), mode="L")
                if len(image.shape) == 3:
                    if image.shape[2] == 3:
                        # RGB
                        return PILImage.fromarray(
                            (image * 255).astype(np.uint8), mode="RGB"
                        )
                    if image.shape[2] == 4:
                        # RGBA
                        return PILImage.fromarray(
                            (image * 255).astype(np.uint8), mode="RGBA"
                        )
                logger.warning(f"Unsupported numpy array shape: {image.shape}")
                return None
            logger.warning(f"Cannot convert image type to PIL: {type(image)}")
            return None
        except Exception as e:
            logger.error(f"Error converting image to PIL: {e}")
            return None

    def _optimize_image(self, pil_image: Any, PILImage: Any) -> Any:
        """Optimize PIL image based on configuration.

        Args:
            pil_image: PIL Image to optimize
            PILImage: PIL Image class

        Returns:
            Optimized PIL Image
        """
        try:
            # Resize image if it exceeds max_image_size
            max_width, max_height = self.config.max_image_size
            current_width, current_height = pil_image.size

            if current_width > max_width or current_height > max_height:
                # Calculate scaling factor to maintain aspect ratio
                scale_factor = min(
                    max_width / current_width, max_height / current_height
                )
                new_width = int(current_width * scale_factor)
                new_height = int(current_height * scale_factor)

                # Use high-quality resampling
                pil_image = pil_image.resize(
                    (new_width, new_height), PILImage.Resampling.LANCZOS
                )
                logger.debug(
                    f"Resized image from {current_width}x{current_height} to {new_width}x{new_height}"
                )

            # Convert to RGB if needed (for PNG format compatibility)
            if self.config.image_format in ("png", "both") and pil_image.mode not in (
                "RGB",
                "RGBA",
                "L",
            ):
                pil_image = pil_image.convert("RGB")

            return pil_image

        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            return pil_image  # Return original if optimization fails

    def _create_wandb_image(self, pil_image: Any, name: str) -> Any:
        """Create wandb image from optimized PIL image.

        Args:
            pil_image: Optimized PIL Image
            name: Name/identifier for the image

        Returns:
            wandb.Image object
        """
        try:
            # Create wandb image with caption
            return self._wandb.Image(pil_image, caption=name)
        except Exception as e:
            logger.error(f"Error creating wandb image for {name}: {e}")
            return None

    def _generate_experiment_tags(self, experiment_config: Dict[str, Any]) -> List[str]:
        """Generate enhanced tags based on experiment configuration.

        Args:
            experiment_config: Complete experiment configuration

        Returns:
            List of tags for the experiment
        """
        tags = list(self.config.tags)  # Start with configured tags

        try:
            # Add dataset-specific tags
            if "dataset" in experiment_config:
                dataset_config = experiment_config["dataset"]
                if isinstance(dataset_config, dict):
                    if "name" in dataset_config:
                        tags.append(f"dataset:{dataset_config['name']}")
                    if "split" in dataset_config:
                        tags.append(f"split:{dataset_config['split']}")
                elif isinstance(dataset_config, str):
                    tags.append(f"dataset:{dataset_config}")

            # Add action format tags
            if "action" in experiment_config:
                action_config = experiment_config["action"]
                if isinstance(action_config, dict) and "format" in action_config:
                    tags.append(f"action:{action_config['format']}")
                elif isinstance(action_config, str):
                    tags.append(f"action:{action_config}")

            # Add environment tags
            if "environment" in experiment_config:
                env_config = experiment_config["environment"]
                if isinstance(env_config, dict):
                    if "type" in env_config:
                        tags.append(f"env:{env_config['type']}")
                elif isinstance(env_config, str):
                    tags.append(f"env:{env_config}")

            # Add debug level tags
            if "debug" in experiment_config:
                debug_config = experiment_config["debug"]
                if isinstance(debug_config, dict) and "level" in debug_config:
                    tags.append(f"debug:{debug_config['level']}")
                elif isinstance(debug_config, str):
                    tags.append(f"debug:{debug_config}")

            # Add algorithm/agent tags if present
            if "algorithm" in experiment_config:
                algo_config = experiment_config["algorithm"]
                if isinstance(algo_config, dict) and "name" in algo_config:
                    tags.append(f"algo:{algo_config['name']}")
                elif isinstance(algo_config, str):
                    tags.append(f"algo:{algo_config}")

            # Add visualization tags
            if "visualization" in experiment_config:
                vis_config = experiment_config["visualization"]
                if isinstance(vis_config, dict):
                    if "debug_level" in vis_config:
                        tags.append(f"vis:{vis_config['debug_level']}")
                    if "wandb" in vis_config and isinstance(vis_config["wandb"], dict):
                        if vis_config["wandb"].get("enabled", False):
                            tags.append("wandb:enabled")

            # Add generic jaxarc tag
            if "jaxarc" not in tags:
                tags.append("jaxarc")

        except Exception as e:
            logger.warning(f"Error generating experiment tags: {e}")

        return tags

    def _generate_experiment_group(
        self, experiment_config: Dict[str, Any]
    ) -> Optional[str]:
        """Generate experiment group based on configuration.

        Args:
            experiment_config: Complete experiment configuration

        Returns:
            Group name for organizing related experiments
        """
        if self.config.group:
            return self.config.group

        try:
            # Generate group based on key experiment parameters
            group_parts = []

            # Add dataset to group
            if "dataset" in experiment_config:
                dataset_config = experiment_config["dataset"]
                if isinstance(dataset_config, dict) and "name" in dataset_config:
                    group_parts.append(dataset_config["name"])
                elif isinstance(dataset_config, str):
                    group_parts.append(dataset_config)

            # Add algorithm to group
            if "algorithm" in experiment_config:
                algo_config = experiment_config["algorithm"]
                if isinstance(algo_config, dict) and "name" in algo_config:
                    group_parts.append(algo_config["name"])
                elif isinstance(algo_config, str):
                    group_parts.append(algo_config)

            # Add action format to group
            if "action" in experiment_config:
                action_config = experiment_config["action"]
                if isinstance(action_config, dict) and "format" in action_config:
                    group_parts.append(action_config["format"])
                elif isinstance(action_config, str):
                    group_parts.append(action_config)

            if group_parts:
                return "-".join(group_parts)

        except Exception as e:
            logger.warning(f"Error generating experiment group: {e}")

        return None

    def _generate_job_type(self, experiment_config: Dict[str, Any]) -> Optional[str]:
        """Generate job type based on configuration.

        Args:
            experiment_config: Complete experiment configuration

        Returns:
            Job type for categorizing the experiment
        """
        if self.config.job_type:
            return self.config.job_type

        try:
            # Determine job type based on configuration
            if "debug" in experiment_config:
                debug_config = experiment_config["debug"]
                if isinstance(debug_config, dict):
                    debug_level = debug_config.get("level", "standard")
                    if debug_level in ("off", "minimal"):
                        return "training"
                    return "debugging"
                if isinstance(debug_config, str) and debug_config in ("off", "minimal"):
                    return "training"
                return "debugging"

            # Check if this looks like a training run
            if "algorithm" in experiment_config or "agent" in experiment_config:
                return "training"

            # Check if this looks like evaluation
            if "evaluation" in experiment_config or "eval" in experiment_config:
                return "evaluation"

            # Default to experiment
            return "experiment"

        except Exception as e:
            logger.warning(f"Error generating job type: {e}")
            return "experiment"

    def _generate_run_name(
        self, experiment_config: Dict[str, Any], provided_name: Optional[str]
    ) -> Optional[str]:
        """Generate enhanced run name based on configuration.

        Args:
            experiment_config: Complete experiment configuration
            provided_name: User-provided run name (takes precedence)

        Returns:
            Enhanced run name or None to use wandb default
        """
        if provided_name:
            return provided_name

        try:
            # Generate descriptive run name
            name_parts = []

            # Add dataset
            if "dataset" in experiment_config:
                dataset_config = experiment_config["dataset"]
                if isinstance(dataset_config, dict) and "name" in dataset_config:
                    name_parts.append(dataset_config["name"])
                elif isinstance(dataset_config, str):
                    name_parts.append(dataset_config)

            # Add algorithm
            if "algorithm" in experiment_config:
                algo_config = experiment_config["algorithm"]
                if isinstance(algo_config, dict) and "name" in algo_config:
                    name_parts.append(algo_config["name"])
                elif isinstance(algo_config, str):
                    name_parts.append(algo_config)

            # Add debug level if not standard
            if "debug" in experiment_config:
                debug_config = experiment_config["debug"]
                debug_level = None
                if isinstance(debug_config, dict):
                    debug_level = debug_config.get("level", "standard")
                elif isinstance(debug_config, str):
                    debug_level = debug_config

                if debug_level and debug_level != "standard":
                    name_parts.append(f"debug-{debug_level}")

            if name_parts:
                return "-".join(name_parts)

        except Exception as e:
            logger.warning(f"Error generating run name: {e}")

        return None

    def _setup_offline_cache(self) -> None:
        """Set up offline caching directory and initialize cache."""
        if self.config.offline_cache_dir:
            self._offline_cache_dir = Path(self.config.offline_cache_dir)
        else:
            # Use default cache directory
            self._offline_cache_dir = Path.home() / ".jaxarc" / "wandb_cache"

        # Create cache directory if it doesn't exist
        self._offline_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache metadata
        self._cache_metadata_file = self._offline_cache_dir / "cache_metadata.json"
        self._load_cache_metadata()

    def _load_cache_metadata(self) -> None:
        """Load cache metadata from disk."""
        try:
            if self._cache_metadata_file.exists():
                with open(self._cache_metadata_file) as f:
                    metadata = json.load(f)
                    self._cached_entries = metadata.get("entries", [])
            else:
                self._cached_entries = []
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
            self._cached_entries = []

    def _save_cache_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            metadata = {"entries": self._cached_entries, "last_updated": time.time()}
            with open(self._cache_metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")

    def _is_network_error(self, exception: Exception) -> bool:
        """Check if an exception is a network-related error.

        Args:
            exception: Exception to check

        Returns:
            True if the exception is network-related
        """
        network_error_types = (
            ConnectionError,
            TimeoutError,
            URLError,
            socket.error,
            socket.timeout,
            OSError,  # Can include network-related OS errors
        )

        # Check exception type
        if isinstance(exception, network_error_types):
            return True

        # Check exception message for common network error patterns
        error_msg = str(exception).lower()
        network_keywords = [
            "connection",
            "timeout",
            "network",
            "unreachable",
            "dns",
            "resolve",
            "refused",
            "reset",
            "broken pipe",
        ]

        return any(keyword in error_msg for keyword in network_keywords)

    def _cache_log_entry(
        self, data: Dict[str, Any], step: Optional[int] = None
    ) -> None:
        """Cache log entry for offline sync.

        Args:
            data: Data to cache
            step: Optional step number
        """
        try:
            # Create cache entry
            cache_entry = {
                "timestamp": time.time(),
                "data": data,
                "step": step,
                "run_id": self.run_id if self.run else None,
            }

            # Save to cache file
            cache_filename = f"entry_{len(self._cached_entries):06d}.json"
            cache_file = self._offline_cache_dir / cache_filename

            if self.config.cache_compression:
                # Save compressed
                import gzip

                with gzip.open(f"{cache_file}.gz", "wt") as f:
                    json.dump(cache_entry, f)
                cache_filename += ".gz"
            else:
                # Save uncompressed
                with open(cache_file, "w") as f:
                    json.dump(cache_entry, f, indent=2)

            # Add to metadata
            self._cached_entries.append(
                {
                    "filename": cache_filename,
                    "timestamp": cache_entry["timestamp"],
                    "step": step,
                    "run_id": cache_entry["run_id"],
                }
            )

            # Save metadata
            self._save_cache_metadata()

            # Clean up cache if needed
            self._cleanup_cache()

            logger.debug(f"Cached log entry: {cache_filename}")

        except Exception as e:
            logger.error(f"Error caching log entry: {e}")

    def _cleanup_cache(self) -> None:
        """Clean up cache based on size limits."""
        try:
            # Calculate total cache size
            total_size = 0
            for entry in self._cached_entries:
                cache_file = self._offline_cache_dir / entry["filename"]
                if cache_file.exists():
                    total_size += cache_file.stat().st_size

            # Convert to GB
            total_size_gb = total_size / (1024**3)

            if total_size_gb > self.config.max_cache_size_gb:
                logger.info(
                    f"Cache size ({total_size_gb:.2f} GB) exceeds limit ({self.config.max_cache_size_gb} GB). Cleaning up..."
                )

                # Sort by timestamp (oldest first)
                sorted_entries = sorted(
                    self._cached_entries, key=lambda x: x["timestamp"]
                )

                # Remove oldest entries until under limit
                while (
                    total_size_gb > self.config.max_cache_size_gb * 0.8
                    and sorted_entries
                ):  # Clean to 80% of limit
                    entry = sorted_entries.pop(0)
                    cache_file = self._offline_cache_dir / entry["filename"]

                    if cache_file.exists():
                        file_size = cache_file.stat().st_size
                        cache_file.unlink()
                        total_size -= file_size
                        total_size_gb = total_size / (1024**3)
                        logger.debug(f"Removed cached entry: {entry['filename']}")

                # Update cached entries list
                self._cached_entries = sorted_entries
                self._save_cache_metadata()

        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")

    def _log_with_retry(self, data: Dict[str, Any], step: Optional[int] = None) -> bool:
        """Log data with retry logic and offline caching for network errors.

        Args:
            data: Data to log
            step: Optional step number

        Returns:
            True if logging was successful, False otherwise
        """
        if self._offline_mode_active:
            # Cache the entry for later sync
            self._cache_log_entry(data, step)
            return True

        for attempt in range(self.config.retry_attempts):
            try:
                # Set timeout for the request
                if hasattr(self._wandb, "Settings"):
                    # Configure timeout if supported
                    pass  # wandb handles timeouts internally

                if step is not None:
                    self.run.log(data, step=step)
                else:
                    self.run.log(data)
                return True

            except Exception as e:
                is_network_error = self._is_network_error(e)

                if is_network_error and self.config.auto_offline_on_error:
                    logger.warning(
                        f"Network error detected: {e}. Switching to offline mode."
                    )
                    self._offline_mode_active = True
                    self._cache_log_entry(data, step)
                    return True

                if attempt < self.config.retry_attempts - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        self.config.retry_delay * (2**attempt),
                        self.config.max_retry_delay,
                    )
                    # Add jitter to avoid thundering herd
                    jitter = delay * 0.1 * (0.5 - time.time() % 1)
                    actual_delay = delay + jitter

                    logger.warning(
                        f"Wandb log attempt {attempt + 1} failed: {e}. Retrying in {actual_delay:.1f}s..."
                    )
                    time.sleep(actual_delay)
                else:
                    logger.error(f"All wandb log attempts failed: {e}")

                    # Cache the entry if it's a network error
                    if is_network_error:
                        logger.info(
                            "Caching entry for offline sync due to network error"
                        )
                        self._cache_log_entry(data, step)
                        return True

                    return False

        return False

    def sync_offline_data(self, force: bool = False) -> Dict[str, Any]:
        """Sync cached offline data to wandb.

        Args:
            force: Force sync even if not in auto-sync mode

        Returns:
            Dictionary with sync results
        """
        if not self._wandb_available or self.run is None:
            return {
                "success": False,
                "error": "Wandb not available or run not initialized",
                "synced_count": 0,
                "failed_count": 0,
            }

        if not force and not self.config.auto_sync_on_reconnect:
            return {
                "success": False,
                "error": "Auto-sync disabled and force=False",
                "synced_count": 0,
                "failed_count": 0,
            }

        logger.info(f"Starting sync of {len(self._cached_entries)} cached entries...")

        synced_count = 0
        failed_count = 0
        errors = []

        # Process entries in batches
        for i in range(0, len(self._cached_entries), self.config.sync_batch_size):
            batch = self._cached_entries[i : i + self.config.sync_batch_size]

            for entry_meta in batch:
                try:
                    # Load cached entry
                    cache_file = self._offline_cache_dir / entry_meta["filename"]

                    if not cache_file.exists():
                        logger.warning(
                            f"Cache file not found: {entry_meta['filename']}"
                        )
                        failed_count += 1
                        continue

                    # Load entry data
                    if entry_meta["filename"].endswith(".gz"):
                        import gzip

                        with gzip.open(cache_file, "rt") as f:
                            cache_entry = json.load(f)
                    else:
                        with open(cache_file) as f:
                            cache_entry = json.load(f)

                    # Sync to wandb
                    data = cache_entry["data"]
                    step = cache_entry.get("step")

                    if step is not None:
                        self.run.log(data, step=step)
                    else:
                        self.run.log(data)

                    # Remove synced file
                    cache_file.unlink()
                    synced_count += 1

                    logger.debug(f"Synced cached entry: {entry_meta['filename']}")

                except Exception as e:
                    error_msg = f"Error syncing {entry_meta['filename']}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    failed_count += 1

            # Small delay between batches to avoid overwhelming the API
            if i + self.config.sync_batch_size < len(self._cached_entries):
                time.sleep(0.1)

        # Update cached entries list (remove synced entries)
        if synced_count > 0:
            self._cached_entries = [
                entry
                for entry in self._cached_entries
                if (self._offline_cache_dir / entry["filename"]).exists()
            ]
            self._save_cache_metadata()

        # Reset offline mode if sync was successful
        if synced_count > 0 and failed_count == 0:
            self._offline_mode_active = False
            logger.info("Sync completed successfully. Resuming online mode.")

        result = {
            "success": failed_count == 0,
            "synced_count": synced_count,
            "failed_count": failed_count,
            "total_entries": len(self._cached_entries) + synced_count,
            "errors": errors,
        }

        logger.info(f"Sync completed: {synced_count} synced, {failed_count} failed")
        return result

    def check_network_connectivity(self) -> bool:
        """Check if wandb is reachable.

        Returns:
            True if wandb is reachable, False otherwise
        """
        try:
            import socket

            # Try to connect to wandb API
            sock = socket.create_connection(
                ("api.wandb.ai", 443), timeout=self.config.network_timeout
            )
            sock.close()
            return True

        except Exception as e:
            logger.debug(f"Network connectivity check failed: {e}")
            return False

    def get_offline_status(self) -> Dict[str, Any]:
        """Get current offline status and cache information.

        Returns:
            Dictionary with offline status information
        """
        cache_size = 0
        if self._offline_cache_dir and self._offline_cache_dir.exists():
            for entry in self._cached_entries:
                cache_file = self._offline_cache_dir / entry["filename"]
                if cache_file.exists():
                    cache_size += cache_file.stat().st_size

        return {
            "offline_mode_active": self._offline_mode_active,
            "cached_entries_count": len(self._cached_entries),
            "cache_size_bytes": cache_size,
            "cache_size_mb": cache_size / (1024**2),
            "cache_directory": str(self._offline_cache_dir)
            if self._offline_cache_dir
            else None,
            "network_connectivity": self.check_network_connectivity()
            if self._wandb_available
            else False,
        }

    def clear_offline_cache(self, confirm: bool = False) -> bool:
        """Clear all cached offline data.

        Args:
            confirm: Must be True to actually clear the cache

        Returns:
            True if cache was cleared, False otherwise
        """
        if not confirm:
            logger.warning(
                "Cache clear requested but not confirmed. Set confirm=True to actually clear."
            )
            return False

        try:
            # Remove all cache files
            removed_count = 0
            for entry in self._cached_entries:
                cache_file = self._offline_cache_dir / entry["filename"]
                if cache_file.exists():
                    cache_file.unlink()
                    removed_count += 1

            # Clear metadata
            self._cached_entries = []
            self._save_cache_metadata()

            logger.info(f"Cleared offline cache: {removed_count} files removed")
            return True

        except Exception as e:
            logger.error(f"Error clearing offline cache: {e}")
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

    @property
    def is_offline(self) -> bool:
        """Check if currently in offline mode."""
        return self._offline_mode_active or self.config.offline_mode

    @property
    def cached_entries_count(self) -> int:
        """Get the number of cached entries."""
        return len(self._cached_entries)

    def force_offline_mode(self) -> None:
        """Force switch to offline mode."""
        self._offline_mode_active = True
        logger.info("Forced switch to offline mode")

    def force_online_mode(self) -> None:
        """Force switch to online mode (will attempt to sync cached data)."""
        if self._offline_mode_active:
            self._offline_mode_active = False
            logger.info("Forced switch to online mode")

            # Attempt to sync cached data if auto-sync is enabled
            if self.config.auto_sync_on_reconnect and self._cached_entries:
                logger.info("Attempting to sync cached data...")
                sync_result = self.sync_offline_data()
                if sync_result["success"]:
                    logger.info(
                        f"Successfully synced {sync_result['synced_count']} cached entries"
                    )
                else:
                    logger.warning(
                        f"Sync partially failed: {sync_result['failed_count']} entries failed"
                    )

    def get_sync_status(self) -> Dict[str, Any]:
        """Get detailed sync status information.

        Returns:
            Dictionary with sync status details
        """
        offline_status = self.get_offline_status()

        return {
            **offline_status,
            "auto_sync_enabled": self.config.auto_sync_on_reconnect,
            "sync_batch_size": self.config.sync_batch_size,
            "retry_attempts": self.config.retry_attempts,
            "auto_offline_on_error": self.config.auto_offline_on_error,
            "wandb_available": self._wandb_available,
            "run_initialized": self.is_initialized,
        }


def create_wandb_config(
    enabled: bool = False, project_name: str = "jaxarc-experiments", **kwargs: Any
) -> WandbConfig:
    """Create a WandbConfig with common defaults.

    Args:
        enabled: Whether to enable wandb integration
        project_name: Name of the wandb project
        **kwargs: Additional configuration options

    Returns:
        Configured WandbConfig instance
    """
    return WandbConfig(enabled=enabled, project_name=project_name, **kwargs)


def create_research_wandb_config(
    project_name: str = "jaxarc-research", entity: Optional[str] = None
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
        tags=["research", "jaxarc"],
    )


def create_development_wandb_config(project_name: str = "jaxarc-dev") -> WandbConfig:
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
        tags=["development", "jaxarc"],
    )
