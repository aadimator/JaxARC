# Design Document

## Overview

This design enhances the JaxARC visualization and logging system with episode-based storage, performance optimization, wandb integration, and configurable debug modes. The solution maintains JAX performance while providing rich visualization capabilities for research and debugging.

## Architecture

### Core Components

```
src/jaxarc/utils/
├── visualization/
│   ├── __init__.py              # Public API exports
│   ├── core.py                  # Core visualization functions (existing)
│   ├── episode_manager.py       # Episode-based storage management
│   ├── async_logger.py          # Asynchronous logging system
│   ├── wandb_integration.py     # Weights & Biases integration
│   ├── replay_system.py         # Episode replay and analysis
│   └── config_validation.py     # Visualization config validation
├── logging/
│   ├── __init__.py              # Logging utilities
│   ├── structured_logger.py     # Structured episode logging
│   ├── performance_monitor.py   # Performance impact monitoring
│   └── storage_manager.py       # Storage cleanup and management
```

### Configuration Structure

```
conf/
├── visualization/
│   ├── debug_off.yaml          # No visualization
│   ├── debug_minimal.yaml      # Episode summaries only
│   ├── debug_standard.yaml     # Key steps and changes
│   ├── debug_verbose.yaml      # All steps and actions
│   └── debug_full.yaml         # Complete state dumps
├── logging/
│   ├── local_only.yaml         # Local file logging
│   ├── wandb_basic.yaml        # Basic wandb integration
│   └── wandb_full.yaml         # Full wandb logging
└── storage/
    ├── development.yaml        # Dev-friendly storage settings
    ├── research.yaml          # Research-optimized settings
    └── production.yaml        # Production-safe settings
```

## Components and Interfaces

### 1. Episode Manager

**Purpose:** Manages episode-based directory structure and file organization.

```python
@chex.dataclass
class EpisodeConfig:
    """Configuration for episode management."""
    base_output_dir: str = "outputs/episodes"
    run_name: str | None = None  # Auto-generated if None
    episode_dir_format: str = "episode_{episode:04d}"
    step_file_format: str = "step_{step:03d}"
    max_episodes_per_run: int = 1000
    cleanup_policy: str = "size_based"  # "oldest_first", "size_based", "manual"
    max_storage_gb: float = 10.0

class EpisodeManager:
    """Manages episode-based storage and organization."""
    
    def __init__(self, config: EpisodeConfig):
        self.config = config
        self.current_run_dir: Path | None = None
        self.current_episode_dir: Path | None = None
        
    def start_new_run(self, run_name: str | None = None) -> Path:
        """Start a new training run with timestamped directory."""
        
    def start_new_episode(self, episode_num: int) -> Path:
        """Start a new episode within the current run."""
        
    def get_step_path(self, step_num: int, file_type: str = "svg") -> Path:
        """Get file path for a specific step visualization."""
        
    def cleanup_old_data(self) -> None:
        """Clean up old data based on configured policy."""
```

### 2. Async Logger

**Purpose:** Provides asynchronous logging to minimize JAX performance impact.

```python
@chex.dataclass
class AsyncLoggerConfig:
    """Configuration for asynchronous logging."""
    queue_size: int = 1000
    worker_threads: int = 2
    batch_size: int = 10
    flush_interval: float = 5.0  # seconds
    enable_compression: bool = True

class AsyncLogger:
    """Asynchronous logger for visualization data."""
    
    def __init__(self, config: AsyncLoggerConfig):
        self.config = config
        self.queue: Queue = Queue(maxsize=config.queue_size)
        self.workers: list[Thread] = []
        
    def log_step_visualization(
        self, 
        step_data: StepVisualizationData,
        priority: int = 0
    ) -> None:
        """Queue step visualization for async processing."""
        
    def log_episode_summary(
        self, 
        episode_data: EpisodeSummaryData
    ) -> None:
        """Queue episode summary for async processing."""
        
    def flush(self) -> None:
        """Force flush all pending logs."""
```

### 3. Wandb Integration

**Purpose:** Seamless integration with Weights & Biases for experiment tracking.

```python
@chex.dataclass
class WandbConfig:
    """Configuration for Weights & Biases integration."""
    enabled: bool = False
    project_name: str = "jaxarc-experiments"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    log_frequency: int = 10  # Log every N steps
    image_format: str = "png"  # "png", "svg", "both"
    max_image_size: tuple[int, int] = (800, 600)
    log_gradients: bool = False
    log_model_topology: bool = False

class WandbIntegration:
    """Weights & Biases integration for JaxARC."""
    
    def __init__(self, config: WandbConfig):
        self.config = config
        self.run = None
        
    def initialize_run(
        self, 
        experiment_config: dict,
        run_name: str | None = None
    ) -> None:
        """Initialize wandb run with experiment configuration."""
        
    def log_step(
        self,
        step_num: int,
        metrics: dict,
        images: dict[str, Any] | None = None
    ) -> None:
        """Log step metrics and visualizations."""
        
    def log_episode_summary(
        self,
        episode_num: int,
        summary_data: dict,
        summary_image: Any | None = None
    ) -> None:
        """Log episode summary with key metrics."""
        
    def finish_run(self) -> None:
        """Properly close wandb run."""
```

### 4. Enhanced Visualization Functions

**Purpose:** Improved visualization functions with better performance and information density.

```python
@chex.dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    debug_level: str = "standard"  # "off", "minimal", "standard", "verbose", "full"
    output_formats: list[str] = field(default_factory=lambda: ["svg"])
    image_quality: str = "high"  # "low", "medium", "high"
    show_coordinates: bool = False
    show_operation_names: bool = True
    highlight_changes: bool = True
    include_metrics: bool = True
    color_scheme: str = "default"  # "default", "colorblind", "high_contrast"

class EnhancedVisualizer:
    """Enhanced visualization system with performance optimization."""
    
    def __init__(
        self, 
        vis_config: VisualizationConfig,
        episode_manager: EpisodeManager,
        async_logger: AsyncLogger,
        wandb_integration: WandbIntegration | None = None
    ):
        self.vis_config = vis_config
        self.episode_manager = episode_manager
        self.async_logger = async_logger
        self.wandb = wandb_integration
        
    def visualize_step(
        self,
        before_state: ArcEnvState,
        action: dict,
        after_state: ArcEnvState,
        reward: float,
        info: dict,
        step_num: int
    ) -> None:
        """Create and save step visualization with enhanced information."""
        
    def visualize_episode_summary(
        self,
        episode_data: EpisodeData,
        episode_num: int
    ) -> None:
        """Create comprehensive episode summary visualization."""
        
    def create_comparison_visualization(
        self,
        episodes: list[EpisodeData],
        comparison_type: str = "reward_progression"
    ) -> str:
        """Create comparison visualization across multiple episodes."""
```

### 5. Structured Logging System

**Purpose:** Structured logging for episode replay and analysis.

```python
@chex.dataclass
class StepLogEntry:
    """Structured log entry for a single step."""
    step_num: int
    timestamp: float
    before_state: dict  # Serialized state
    action: dict
    after_state: dict  # Serialized state
    reward: float
    info: dict
    visualization_path: str | None = None

@chex.dataclass
class EpisodeLogEntry:
    """Structured log entry for a complete episode."""
    episode_num: int
    start_timestamp: float
    end_timestamp: float
    total_steps: int
    total_reward: float
    final_similarity: float
    task_id: str
    config_hash: str
    steps: list[StepLogEntry]
    summary_visualization_path: str | None = None

class StructuredLogger:
    """Structured logging system for episode data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.current_episode: EpisodeLogEntry | None = None
        
    def start_episode(
        self, 
        episode_num: int, 
        task_id: str, 
        config_hash: str
    ) -> None:
        """Start logging a new episode."""
        
    def log_step(
        self,
        step_num: int,
        before_state: ArcEnvState,
        action: dict,
        after_state: ArcEnvState,
        reward: float,
        info: dict,
        visualization_path: str | None = None
    ) -> None:
        """Log a single step."""
        
    def end_episode(self, summary_visualization_path: str | None = None) -> None:
        """End current episode and save log."""
        
    def load_episode(self, episode_num: int) -> EpisodeLogEntry:
        """Load episode data for replay/analysis."""
```

## Data Models

### Configuration Data Models

```python
@chex.dataclass
class DebugConfig:
    """Enhanced debug configuration."""
    level: str = "standard"  # "off", "minimal", "standard", "verbose", "full"
    
    # Output settings
    output_dir: str = "outputs/debug"
    formats: list[str] = field(default_factory=lambda: ["svg"])
    
    # Episode management
    max_episodes: int = 100
    cleanup_policy: str = "size_based"
    max_storage_gb: float = 5.0
    
    # Performance settings
    async_logging: bool = True
    queue_size: int = 1000
    
    # Visualization settings
    show_coordinates: bool = False
    show_operation_names: bool = True
    highlight_changes: bool = True
    color_scheme: str = "default"
    
    # Integration settings
    wandb: WandbConfig = field(default_factory=WandbConfig)

@chex.dataclass
class LoggingConfig:
    """Logging configuration."""
    structured_logging: bool = True
    log_format: str = "json"  # "json", "hdf5", "pickle"
    compression: bool = True
    include_full_states: bool = False  # For performance
    log_level: str = "INFO"
```

### Visualization Data Models

```python
@chex.dataclass
class StepVisualizationData:
    """Data for step visualization."""
    step_num: int
    before_grid: Grid
    after_grid: Grid
    action: dict
    reward: float
    info: dict
    selection_mask: jnp.ndarray
    changed_cells: jnp.ndarray
    operation_name: str

@chex.dataclass
class EpisodeSummaryData:
    """Data for episode summary visualization."""
    episode_num: int
    total_steps: int
    total_reward: float
    reward_progression: list[float]
    similarity_progression: list[float]
    final_similarity: float
    task_id: str
    success: bool
    key_moments: list[int]  # Important step numbers
```

## Error Handling

### Storage Error Handling

```python
class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass

class DiskSpaceError(StorageError):
    """Raised when disk space is insufficient."""
    pass

class PermissionError(StorageError):
    """Raised when file permissions are insufficient."""
    pass

def handle_storage_error(error: StorageError, fallback_dir: Path) -> Path:
    """Handle storage errors with graceful fallback."""
    if isinstance(error, DiskSpaceError):
        # Trigger cleanup and retry
        cleanup_old_files(fallback_dir)
        return fallback_dir
    elif isinstance(error, PermissionError):
        # Try alternative directory
        return get_temp_directory()
    else:
        # Log error and disable visualization
        logger.error(f"Storage error: {error}")
        return None
```

### Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor visualization performance impact."""
    
    def __init__(self):
        self.step_times: list[float] = []
        self.visualization_times: list[float] = []
        
    def measure_step_impact(self, step_func: Callable) -> Callable:
        """Decorator to measure visualization impact on step performance."""
        
    def get_performance_report(self) -> dict:
        """Get performance impact report."""
        
    def should_reduce_logging(self) -> bool:
        """Determine if logging should be reduced due to performance impact."""
```

## Testing Strategy

### Unit Tests

1. **Episode Manager Tests**
   - Directory creation and organization
   - Cleanup policies and storage limits
   - File naming and path generation

2. **Async Logger Tests**
   - Queue management and threading
   - Batch processing and flush operations
   - Error handling and recovery

3. **Wandb Integration Tests**
   - Mock wandb API interactions
   - Image format conversion and upload
   - Configuration validation

4. **Visualization Function Tests**
   - SVG generation and quality
   - Performance benchmarks
   - Memory usage monitoring

### Integration Tests

1. **End-to-End Visualization Pipeline**
   - Complete episode logging workflow
   - Multi-format output generation
   - Storage cleanup and management

2. **JAX Performance Impact Tests**
   - Measure visualization overhead
   - Async logging performance
   - Memory leak detection

3. **Configuration Integration Tests**
   - Hydra config composition
   - Override validation
   - Error message clarity

### Performance Tests

1. **Scalability Tests**
   - Large episode handling (1000+ steps)
   - Multiple concurrent episodes
   - Storage cleanup efficiency

2. **Memory Usage Tests**
   - Visualization memory footprint
   - Async queue memory management
   - Garbage collection effectiveness

## Implementation Notes

### JAX Compatibility

- All visualization callbacks must be JAX-compatible using `jax.debug.callback`
- State serialization must handle JAX arrays properly
- Async processing must not interfere with JAX transformations
- Use `jaxtyping` and `equinox` where feasible

### Performance Optimization

- Use lazy loading for large visualization datasets
- Implement efficient image compression for storage
- Batch file I/O operations to reduce overhead
- Monitor and limit memory usage during visualization

### Wandb Integration Best Practices

- Optimize image sizes for upload speed
- Use wandb's built-in image logging capabilities
- Implement proper error handling for network issues
- Support offline mode for environments without internet

### Configuration Management

- Extend existing Hydra configuration structure
- Provide sensible defaults for all settings
- Validate configuration combinations
- Support environment variable overrides for deployment