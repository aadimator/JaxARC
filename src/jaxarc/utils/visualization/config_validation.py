"""Configuration validation for enhanced visualization system."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chex
from omegaconf import DictConfig, OmegaConf


@chex.dataclass
class ValidationError:
    """Represents a configuration validation error."""
    field: str
    value: Any
    message: str
    suggestion: Optional[str] = None


class ConfigValidator:
    """Validates visualization configuration settings."""
    
    VALID_DEBUG_LEVELS = {"off", "minimal", "standard", "verbose", "full"}
    VALID_OUTPUT_FORMATS = {"svg", "png", "html"}
    VALID_IMAGE_QUALITIES = {"low", "medium", "high"}
    VALID_COLOR_SCHEMES = {"default", "colorblind", "high_contrast"}
    VALID_CLEANUP_POLICIES = {"oldest_first", "size_based", "manual"}
    VALID_LOG_FORMATS = {"json", "hdf5", "pickle"}
    
    def __init__(self):
        self.errors: List[ValidationError] = []
    
    def validate_visualization_config(self, config: DictConfig) -> List[ValidationError]:
        """Validate visualization configuration."""
        self.errors = []
        
        if not hasattr(config, 'visualization'):
            return self.errors
            
        vis_config = config.visualization
        
        # Validate debug level
        if hasattr(vis_config, 'debug_level'):
            self._validate_debug_level(vis_config.debug_level)
        
        # Validate output formats
        if hasattr(vis_config, 'output_formats'):
            self._validate_output_formats(vis_config.output_formats)
        
        # Validate image quality
        if hasattr(vis_config, 'image_quality'):
            self._validate_image_quality(vis_config.image_quality)
        
        # Validate color scheme
        if hasattr(vis_config, 'color_scheme'):
            self._validate_color_scheme(vis_config.color_scheme)
        
        # Validate output directory
        if hasattr(vis_config, 'output_dir'):
            self._validate_output_directory(vis_config.output_dir)
        
        # Validate memory limits
        if hasattr(vis_config, 'memory_limit_mb'):
            self._validate_memory_limit(vis_config.memory_limit_mb)
        
        return self.errors
    
    def validate_storage_config(self, config: DictConfig) -> List[ValidationError]:
        """Validate storage configuration."""
        self.errors = []
        
        if not hasattr(config, 'storage'):
            return self.errors
            
        storage_config = config.storage
        
        # Validate cleanup policy
        if hasattr(storage_config, 'cleanup_policy'):
            self._validate_cleanup_policy(storage_config.cleanup_policy)
        
        # Validate storage limits
        if hasattr(storage_config, 'max_storage_gb'):
            self._validate_storage_limit(storage_config.max_storage_gb)
        
        # Validate episode limits
        if hasattr(storage_config, 'max_episodes_per_run'):
            self._validate_episode_limit(storage_config.max_episodes_per_run)
        
        # Validate base output directory
        if hasattr(storage_config, 'base_output_dir'):
            self._validate_base_output_dir(storage_config.base_output_dir)
        
        return self.errors
    
    def validate_logging_config(self, config: DictConfig) -> List[ValidationError]:
        """Validate logging configuration."""
        self.errors = []
        
        if not hasattr(config, 'logging'):
            return self.errors
            
        logging_config = config.logging
        
        # Validate log format
        if hasattr(logging_config, 'log_format'):
            self._validate_log_format(logging_config.log_format)
        
        # Validate async logger settings
        if hasattr(config, 'async_logger'):
            self._validate_async_logger_config(config.async_logger)
        
        return self.errors
    
    def validate_wandb_config(self, config: DictConfig) -> List[ValidationError]:
        """Validate wandb configuration."""
        self.errors = []
        
        if not hasattr(config, 'wandb'):
            return self.errors
            
        wandb_config = config.wandb
        
        # Validate project name
        if hasattr(wandb_config, 'project_name'):
            self._validate_project_name(wandb_config.project_name)
        
        # Validate image format
        if hasattr(wandb_config, 'image_format'):
            self._validate_wandb_image_format(wandb_config.image_format)
        
        # Validate log frequency
        if hasattr(wandb_config, 'log_frequency'):
            self._validate_log_frequency(wandb_config.log_frequency)
        
        # Validate retry settings
        if hasattr(wandb_config, 'retry_attempts'):
            self._validate_retry_attempts(wandb_config.retry_attempts)
        
        return self.errors
    
    def validate_complete_config(self, config: DictConfig) -> List[ValidationError]:
        """Validate complete configuration with cross-validation."""
        all_errors = []
        
        # Validate individual sections
        all_errors.extend(self.validate_visualization_config(config))
        all_errors.extend(self.validate_storage_config(config))
        all_errors.extend(self.validate_logging_config(config))
        all_errors.extend(self.validate_wandb_config(config))
        
        # Cross-validation checks
        all_errors.extend(self._cross_validate_config(config))
        
        return all_errors
    
    def _validate_debug_level(self, debug_level: str) -> None:
        """Validate debug level."""
        if debug_level not in self.VALID_DEBUG_LEVELS:
            self.errors.append(ValidationError(
                field="visualization.debug_level",
                value=debug_level,
                message=f"Invalid debug level: {debug_level}",
                suggestion=f"Valid options: {', '.join(self.VALID_DEBUG_LEVELS)}"
            ))
    
    def _validate_output_formats(self, formats: List[str]) -> None:
        """Validate output formats."""
        # Handle OmegaConf ListConfig
        if hasattr(formats, '_content'):
            formats = list(formats)
        
        if not isinstance(formats, (list, tuple)):
            self.errors.append(ValidationError(
                field="visualization.output_formats",
                value=formats,
                message="Output formats must be a list",
                suggestion="Use a list like ['svg', 'png']"
            ))
            return
        
        invalid_formats = set(formats) - self.VALID_OUTPUT_FORMATS
        if invalid_formats:
            self.errors.append(ValidationError(
                field="visualization.output_formats",
                value=list(invalid_formats),
                message=f"Invalid output formats: {invalid_formats}",
                suggestion=f"Valid options: {', '.join(self.VALID_OUTPUT_FORMATS)}"
            ))
    
    def _validate_image_quality(self, quality: str) -> None:
        """Validate image quality."""
        if quality not in self.VALID_IMAGE_QUALITIES:
            self.errors.append(ValidationError(
                field="visualization.image_quality",
                value=quality,
                message=f"Invalid image quality: {quality}",
                suggestion=f"Valid options: {', '.join(self.VALID_IMAGE_QUALITIES)}"
            ))
    
    def _validate_color_scheme(self, scheme: str) -> None:
        """Validate color scheme."""
        if scheme not in self.VALID_COLOR_SCHEMES:
            self.errors.append(ValidationError(
                field="visualization.color_scheme",
                value=scheme,
                message=f"Invalid color scheme: {scheme}",
                suggestion=f"Valid options: {', '.join(self.VALID_COLOR_SCHEMES)}"
            ))
    
    def _validate_output_directory(self, output_dir: str) -> None:
        """Validate output directory."""
        try:
            path = Path(output_dir)
            # Check if parent directory exists or can be created
            if not path.parent.exists():
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                except (OSError, PermissionError):
                    self.errors.append(ValidationError(
                        field="visualization.output_dir",
                        value=output_dir,
                        message=f"Cannot create output directory: {output_dir}",
                        suggestion="Check permissions or use a different directory"
                    ))
        except Exception as e:
            self.errors.append(ValidationError(
                field="visualization.output_dir",
                value=output_dir,
                message=f"Invalid output directory path: {e}",
                suggestion="Use a valid file system path"
            ))
    
    def _validate_memory_limit(self, limit: Union[int, float]) -> None:
        """Validate memory limit."""
        if not isinstance(limit, (int, float)) or limit <= 0:
            self.errors.append(ValidationError(
                field="visualization.memory_limit_mb",
                value=limit,
                message="Memory limit must be a positive number",
                suggestion="Use a positive number in MB (e.g., 500)"
            ))
        elif limit > 10000:  # 10GB seems excessive for visualization
            self.errors.append(ValidationError(
                field="visualization.memory_limit_mb",
                value=limit,
                message="Memory limit seems excessive",
                suggestion="Consider using a smaller limit (< 10000 MB)"
            ))
    
    def _validate_cleanup_policy(self, policy: str) -> None:
        """Validate cleanup policy."""
        if policy not in self.VALID_CLEANUP_POLICIES:
            self.errors.append(ValidationError(
                field="storage.cleanup_policy",
                value=policy,
                message=f"Invalid cleanup policy: {policy}",
                suggestion=f"Valid options: {', '.join(self.VALID_CLEANUP_POLICIES)}"
            ))
    
    def _validate_storage_limit(self, limit: Union[int, float]) -> None:
        """Validate storage limit."""
        if not isinstance(limit, (int, float)) or limit <= 0:
            self.errors.append(ValidationError(
                field="storage.max_storage_gb",
                value=limit,
                message="Storage limit must be a positive number",
                suggestion="Use a positive number in GB (e.g., 5.0)"
            ))
    
    def _validate_episode_limit(self, limit: int) -> None:
        """Validate episode limit."""
        if not isinstance(limit, int) or limit <= 0:
            self.errors.append(ValidationError(
                field="storage.max_episodes_per_run",
                value=limit,
                message="Episode limit must be a positive integer",
                suggestion="Use a positive integer (e.g., 100)"
            ))
    
    def _validate_base_output_dir(self, output_dir: str) -> None:
        """Validate base output directory."""
        self._validate_output_directory(output_dir)  # Reuse the same validation
    
    def _validate_log_format(self, format_type: str) -> None:
        """Validate log format."""
        if format_type not in self.VALID_LOG_FORMATS:
            self.errors.append(ValidationError(
                field="logging.log_format",
                value=format_type,
                message=f"Invalid log format: {format_type}",
                suggestion=f"Valid options: {', '.join(self.VALID_LOG_FORMATS)}"
            ))
    
    def _validate_async_logger_config(self, async_config: DictConfig) -> None:
        """Validate async logger configuration."""
        if hasattr(async_config, 'queue_size'):
            if not isinstance(async_config.queue_size, int) or async_config.queue_size <= 0:
                self.errors.append(ValidationError(
                    field="async_logger.queue_size",
                    value=async_config.queue_size,
                    message="Queue size must be a positive integer",
                    suggestion="Use a positive integer (e.g., 1000)"
                ))
        
        if hasattr(async_config, 'worker_threads'):
            if not isinstance(async_config.worker_threads, int) or async_config.worker_threads <= 0:
                self.errors.append(ValidationError(
                    field="async_logger.worker_threads",
                    value=async_config.worker_threads,
                    message="Worker threads must be a positive integer",
                    suggestion="Use 1-4 threads for optimal performance"
                ))
    
    def _validate_project_name(self, project_name: str) -> None:
        """Validate wandb project name."""
        if not isinstance(project_name, str) or not project_name.strip():
            self.errors.append(ValidationError(
                field="wandb.project_name",
                value=project_name,
                message="Project name must be a non-empty string",
                suggestion="Use a descriptive project name"
            ))
    
    def _validate_wandb_image_format(self, image_format: str) -> None:
        """Validate wandb image format."""
        valid_formats = {"png", "svg", "both"}
        if image_format not in valid_formats:
            self.errors.append(ValidationError(
                field="wandb.image_format",
                value=image_format,
                message=f"Invalid wandb image format: {image_format}",
                suggestion=f"Valid options: {', '.join(valid_formats)}"
            ))
    
    def _validate_log_frequency(self, frequency: int) -> None:
        """Validate log frequency."""
        if not isinstance(frequency, int) or frequency < 0:
            self.errors.append(ValidationError(
                field="wandb.log_frequency",
                value=frequency,
                message="Log frequency must be a non-negative integer",
                suggestion="Use 0 for no logging, or positive integer for frequency"
            ))
    
    def _validate_retry_attempts(self, attempts: int) -> None:
        """Validate retry attempts."""
        if not isinstance(attempts, int) or attempts < 0:
            self.errors.append(ValidationError(
                field="wandb.retry_attempts",
                value=attempts,
                message="Retry attempts must be a non-negative integer",
                suggestion="Use 0-10 retry attempts"
            ))
        elif attempts > 10:
            self.errors.append(ValidationError(
                field="wandb.retry_attempts",
                value=attempts,
                message="Too many retry attempts",
                suggestion="Use 0-10 retry attempts for reasonable behavior"
            ))
    
    def _cross_validate_config(self, config: DictConfig) -> List[ValidationError]:
        """Perform cross-validation between different config sections."""
        errors = []
        
        # Check if debug level matches expected storage usage
        if (hasattr(config, 'visualization') and hasattr(config, 'storage') and
            hasattr(config.visualization, 'debug_level') and 
            hasattr(config.storage, 'max_storage_gb')):
            
            debug_level = config.visualization.debug_level
            storage_limit = config.storage.max_storage_gb
            
            # Warn about potential storage issues
            if debug_level in ["verbose", "full"] and storage_limit < 5.0:
                errors.append(ValidationError(
                    field="storage.max_storage_gb",
                    value=storage_limit,
                    message=f"Storage limit may be too low for debug level '{debug_level}'",
                    suggestion="Consider increasing storage limit to 5GB+ for verbose/full debug modes"
                ))
        
        # Check wandb and visualization compatibility
        if (hasattr(config, 'wandb') and hasattr(config, 'visualization') and
            hasattr(config.wandb, 'enabled') and config.wandb.enabled and
            hasattr(config.visualization, 'debug_level') and 
            config.visualization.debug_level == "off"):
            
            errors.append(ValidationError(
                field="visualization.debug_level",
                value="off",
                message="Wandb enabled but visualization is off",
                suggestion="Enable visualization (minimal or higher) when using wandb"
            ))
        
        return errors


def validate_config(config: DictConfig) -> List[ValidationError]:
    """Convenience function to validate a complete configuration."""
    validator = ConfigValidator()
    return validator.validate_complete_config(config)


def format_validation_errors(errors: List[ValidationError]) -> str:
    """Format validation errors into a readable string."""
    if not errors:
        return "Configuration is valid."
    
    lines = ["Configuration validation errors:"]
    for i, error in enumerate(errors, 1):
        lines.append(f"\n{i}. {error.field}: {error.message}")
        if error.suggestion:
            lines.append(f"   Suggestion: {error.suggestion}")
    
    return "\n".join(lines)


def validate_and_raise(config: DictConfig) -> None:
    """Validate configuration and raise exception if invalid."""
    errors = validate_config(config)
    if errors:
        error_message = format_validation_errors(errors)
        raise ValueError(f"Invalid configuration:\n{error_message}")