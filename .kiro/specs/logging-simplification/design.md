# Design Document

## Overview

This design simplifies the JaxARC logging and visualization system by replacing the complex orchestration architecture with a clean, handler-based approach centered around a single ExperimentLogger class. The solution removes overengineered components while preserving the valuable SVG debugging capabilities that are essential for research.

## Architecture

### Core Design Principles

1. **Simplicity First**: Research project priorities - clear, maintainable code over premature optimization
2. **Single Responsibility**: Each handler has one clear purpose
3. **JAX Compatibility**: Maintain pure functions and JAX transformation support
4. **Graceful Degradation**: System continues working if individual handlers fail

### New Architecture Overview

```
src/jaxarc/utils/
├── logging/
│   ├── __init__.py              # Public API exports
│   ├── experiment_logger.py     # Central ExperimentLogger class (NEW)
│   ├── file_handler.py          # Synchronous file logging (REFACTORED from structured_logger.py)
│   ├── wandb_handler.py         # Simplified wandb integration (MOVED from integrations/)
│   └── structured_logger.py     # SIMPLIFIED - becomes FileHandler
├── visualization/
│   ├── __init__.py              # Existing exports
│   ├── svg_handler.py           # SVG generation handler (NEW)
│   ├── rich_display.py          # Console output (EXISTING - unchanged)
│   └── episode_manager.py       # File path management (EXISTING - unchanged)
```

### Components Removed

```
# These files will be DELETED:
src/jaxarc/utils/visualization/
├── visualizer.py                # Complex orchestrator - replaced by ExperimentLogger
├── async_logger.py              # Async complexity - replaced by synchronous logging
├── memory_manager.py            # Premature optimization - removed
├── performance_monitor.py       # Custom monitoring - use standard profiling tools
└── wandb_sync.py               # Custom sync - use official wandb sync command
```

## Components and Interfaces

### 1. ExperimentLogger (Central Coordinator)

**Purpose:** Single entry point for all logging operations, manages handler lifecycle.

**Location:** `src/jaxarc/utils/logging/experiment_logger.py`

```python
from typing import Dict, List, Optional, Any
import equinox as eqx
from jaxarc.envs.config import JaxArcConfig

class ExperimentLogger(eqx.Module):
    """Central logging coordinator with handler-based architecture."""
    
    config: JaxArcConfig
    handlers: Dict[str, Any]  # Handler instances
    
    def __init__(self, config: JaxArcConfig):
        """Initialize logger with handlers based on configuration."""
        self.config = config
        self.handlers = self._initialize_handlers()
    
    def _initialize_handlers(self) -> Dict[str, Any]:
        """Initialize handlers based on configuration settings."""
        handlers = {}
        
        # File logging handler
        if self.config.debug.level != "off":
            from .file_handler import FileHandler
            handlers['file'] = FileHandler(self.config)
        
        # SVG visualization handler
        if self.config.debug.level in ["standard", "verbose", "full"]:
            from ..visualization.svg_handler import SVGHandler
            handlers['svg'] = SVGHandler(self.config)
        
        # Console output handler
        if self.config.debug.level != "off":
            from ..visualization.rich_display import RichHandler
            handlers['rich'] = RichHandler(self.config)
        
        # Wandb integration handler
        if hasattr(self.config, 'wandb') and self.config.wandb.enabled:
            from .wandb_handler import WandbHandler
            handlers['wandb'] = WandbHandler(self.config.wandb)
        
        return handlers
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log step data through all active handlers."""
        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, 'log_step'):
                    handler.log_step(step_data)
            except Exception as e:
                # Log error but continue with other handlers
                print(f"Handler {handler_name} failed: {e}")
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Log episode summary through all active handlers."""
        for handler_name, handler in self.handlers.items():
            try:
                if hasattr(handler, 'log_episode_summary'):
                    handler.log_episode_summary(summary_data)
            except Exception as e:
                print(f"Handler {handler_name} failed: {e}")
    
    def close(self) -> None:
        """Clean shutdown of all handlers."""
        for handler in self.handlers.values():
            if hasattr(handler, 'close'):
                handler.close()
```

### 2. FileHandler (Synchronous File Logging)

**Purpose:** Simple, synchronous file writing for episode data.

**Location:** `src/jaxarc/utils/logging/file_handler.py` (refactored from structured_logger.py)

```python
import json
import pickle
from pathlib import Path
from typing import Dict, Any
import equinox as eqx
from jaxarc.envs.config import JaxArcConfig

class FileHandler(eqx.Module):
    """Synchronous file logging handler."""
    
    config: JaxArcConfig
    output_dir: Path
    current_episode_data: Dict[str, Any]
    
    def __init__(self, config: JaxArcConfig):
        self.config = config
        self.output_dir = Path(config.debug.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_episode_data = {}
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log step data to current episode."""
        if 'steps' not in self.current_episode_data:
            self.current_episode_data['steps'] = []
        
        # Serialize JAX arrays and other complex data
        serialized_step = self._serialize_step_data(step_data)
        self.current_episode_data['steps'].append(serialized_step)
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Save complete episode data to file."""
        episode_num = summary_data.get('episode_num', 0)
        
        # Combine step data with summary
        complete_episode = {
            **summary_data,
            **self.current_episode_data
        }
        
        # Save as JSON (human readable)
        json_path = self.output_dir / f"episode_{episode_num:04d}.json"
        with open(json_path, 'w') as f:
            json.dump(complete_episode, f, indent=2, default=str)
        
        # Save as pickle (preserves data types)
        pickle_path = self.output_dir / f"episode_{episode_num:04d}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(complete_episode, f)
        
        # Reset for next episode
        self.current_episode_data = {}
    
    def _serialize_step_data(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JAX arrays and complex data to serializable format."""
        # Reuse existing serialization utilities instead of reimplementing
        from jaxarc.utils.serialization_utils import serialize_pytree
        from jaxarc.utils.pytree_utils import tree_map_with_path
        
        # Use existing utilities for JAX array and pytree serialization
        return serialize_pytree(step_data)
    
    def close(self) -> None:
        """Clean shutdown - save any pending data."""
        if self.current_episode_data:
            # Save incomplete episode data
            incomplete_path = self.output_dir / "incomplete_episode.json"
            with open(incomplete_path, 'w') as f:
                json.dump(self.current_episode_data, f, indent=2, default=str)
```

### 3. SVGHandler (Visualization Generation)

**Purpose:** Consolidate SVG generation logic from rl_visualization.py and episode_visualization.py.

**Location:** `src/jaxarc/utils/visualization/svg_handler.py`

```python
from typing import Dict, Any, Optional
from pathlib import Path
import equinox as eqx
from jaxarc.envs.config import JaxArcConfig
from .episode_manager import EpisodeManager

class SVGHandler(eqx.Module):
    """SVG visualization generation handler."""
    
    config: JaxArcConfig
    episode_manager: EpisodeManager
    
    def __init__(self, config: JaxArcConfig):
        self.config = config
        self.episode_manager = EpisodeManager(config)
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Generate and save step visualization."""
        if self.config.debug.level in ["verbose", "full"]:
            step_num = step_data.get('step_num', 0)
            svg_content = self._draw_rl_step_svg_enhanced(step_data)
            
            # Save SVG file
            svg_path = self.episode_manager.get_step_path(step_num, "svg")
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(svg_path, 'w') as f:
                f.write(svg_content)
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Generate and save episode summary visualization."""
        episode_num = summary_data.get('episode_num', 0)
        svg_content = self._draw_enhanced_episode_summary_svg(summary_data)
        
        # Save summary SVG
        summary_path = self.episode_manager.get_episode_summary_path(episode_num, "svg")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(svg_content)
    
    def _draw_rl_step_svg_enhanced(self, step_data: Dict[str, Any]) -> str:
        """
        Core SVG generation logic moved from rl_visualization.py.
        
        This method contains the existing draw_rl_step_svg_enhanced logic
        with all the grid rendering, action highlighting, and operation display.
        """
        # Import and use existing SVG generation functions
        from .rl_visualization import draw_rl_step_svg_enhanced
        
        # Extract required data from step_data
        before_state = step_data.get('before_state')
        after_state = step_data.get('after_state')
        action = step_data.get('action')
        reward = step_data.get('reward', 0.0)
        info = step_data.get('info', {})
        
        return draw_rl_step_svg_enhanced(
            before_state=before_state,
            after_state=after_state,
            action=action,
            reward=reward,
            info=info
        )
    
    def _draw_enhanced_episode_summary_svg(self, summary_data: Dict[str, Any]) -> str:
        """
        Core episode summary SVG generation moved from episode_visualization.py.
        """
        from .episode_visualization import draw_enhanced_episode_summary_svg
        
        return draw_enhanced_episode_summary_svg(summary_data)
    
    def close(self) -> None:
        """Clean shutdown."""
        pass
```

### 4. WandbHandler (Simplified Integration)

**Purpose:** Thin wrapper around official wandb library, removing custom sync and retry logic.

**Location:** `src/jaxarc/utils/logging/wandb_handler.py` (moved from integrations/)

```python
from typing import Dict, Any, Optional
import equinox as eqx

class WandbHandler(eqx.Module):
    """Simplified Weights & Biases integration handler."""
    
    config: Any  # WandbConfig
    run: Optional[Any]  # wandb.Run
    
    def __init__(self, wandb_config):
        self.config = wandb_config
        self.run = None
        self._initialize_wandb()
    
    def _initialize_wandb(self) -> None:
        """Initialize wandb run with simple configuration."""
        try:
            import wandb
            
            # Simple wandb.init() call - let wandb handle offline mode via WANDB_MODE env var
            self.run = wandb.init(
                project=self.config.project_name,
                entity=self.config.entity,
                tags=self.config.tags,
                config=self.config.experiment_config if hasattr(self.config, 'experiment_config') else {}
            )
        except ImportError:
            print("wandb not available - skipping wandb logging")
            self.run = None
        except Exception as e:
            print(f"wandb initialization failed: {e}")
            self.run = None
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log step metrics to wandb."""
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
            
            # Simple wandb.log() call - let wandb handle retries and errors
            if metrics:
                self.run.log(metrics)
                
        except Exception as e:
            # Simple error handling - just print and continue
            print(f"wandb logging failed: {e}")
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Log episode summary to wandb."""
        if self.run is None:
            return
        
        try:
            # Log episode-level metrics
            episode_metrics = {
                'episode_num': summary_data.get('episode_num', 0),
                'total_reward': summary_data.get('total_reward', 0),
                'total_steps': summary_data.get('total_steps', 0),
                'final_similarity': summary_data.get('final_similarity', 0),
                'success': summary_data.get('success', False)
            }
            
            self.run.log(episode_metrics)
            
            # Log summary visualization if available
            if 'summary_svg_path' in summary_data:
                import wandb
                self.run.log({
                    "episode_summary": wandb.Image(summary_data['summary_svg_path'])
                })
                
        except Exception as e:
            print(f"wandb episode logging failed: {e}")
    
    def close(self) -> None:
        """Clean shutdown of wandb run."""
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:
                print(f"wandb finish failed: {e}")
```

## Data Models

### Handler Interface Convention

```python
from typing import Protocol, Dict, Any

class LoggingHandler(Protocol):
    """Protocol defining the interface for logging handlers."""
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log data for a single step."""
        ...
    
    def log_episode_summary(self, summary_data: Dict[str, Any]) -> None:
        """Log data for episode completion."""
        ...
    
    def close(self) -> None:
        """Clean shutdown of the handler."""
        ...
```

### Info Dictionary Convention

```python
# Standard structure for step_data and summary_data
step_data = {
    # Core step information
    'step_num': int,
    'before_state': ArcEnvState,
    'after_state': ArcEnvState,
    'action': dict,
    'reward': float,
    
    # Custom metrics for time-series logging (wandb, etc.)
    'info': {
        'metrics': {
            'similarity': float,
            'ppo_policy_value': float,  # Example from PPO training
            'learning_rate': float,     # Example from training loop
            # ... any other scalar metrics
        },
        # Other complex data for visualization/analysis
        'attention_maps': jnp.ndarray,
        'debug_info': dict,
        # ... other non-metric data
    }
}

summary_data = {
    'episode_num': int,
    'total_steps': int,
    'total_reward': float,
    'final_similarity': float,
    'success': bool,
    'task_id': str,
    # ... other episode-level data
}
```

## Error Handling

### Graceful Handler Failure

```python
def _safe_handler_call(handler, method_name: str, data: Dict[str, Any]) -> None:
    """Safely call handler method with error isolation."""
    try:
        method = getattr(handler, method_name, None)
        if method is not None:
            method(data)
    except Exception as e:
        # Log error but don't crash the system
        print(f"Handler {handler.__class__.__name__}.{method_name} failed: {e}")
        # In production, this would use proper logging
```

### JAX Compatibility Preservation

```python
# JAX callback integration remains unchanged
def jax_save_step_visualization(step_data_serialized: Dict[str, Any]) -> None:
    """JAX-compatible callback for step logging."""
    # This function is called from within JAX transformations
    # It receives serialized data and passes it to the logger
    if hasattr(environment, 'logger'):
        environment.logger.log_step(step_data_serialized)

# Usage in functional.py remains the same
jax.debug.callback(jax_save_step_visualization, serialized_step_data)
```

## Testing Strategy

### Unit Tests

1. **ExperimentLogger Tests**
   - Handler initialization based on configuration
   - Error isolation between handlers
   - Proper shutdown sequence

2. **FileHandler Tests**
   - JSON and pickle serialization
   - JAX array handling
   - File path management

3. **SVGHandler Tests**
   - SVG generation functionality preservation
   - File saving and path management
   - Integration with EpisodeManager

4. **WandbHandler Tests**
   - Mock wandb API interactions
   - Error handling and graceful degradation
   - Metrics extraction from info dictionary

### Integration Tests

1. **End-to-End Logging Pipeline**
   - Complete episode logging workflow
   - Handler coordination and error isolation
   - Configuration-driven handler selection

2. **JAX Compatibility Tests**
   - Callback functionality preservation
   - JIT compilation compatibility
   - Performance impact measurement

3. **Migration Tests**
   - Verify existing functionality is preserved
   - Configuration compatibility
   - Output format consistency

## Implementation Notes

### Reusing Existing Utilities

**Priority**: Always search for existing functionality before implementing new code.

**Key Utilities to Leverage:**
- `utils/serialization_utils.py` - JAX array and pytree serialization
- `utils/pytree_utils.py` - Pytree manipulation and transformation
- `utils/equinox_utils.py` - Equinox patterns and utilities
- `utils/grid_utils.py` - Grid operations and transformations
- `utils/config.py` - Configuration handling and validation
- `utils/jax_types.py` - JAX type definitions and utilities

**Implementation Approach:**
1. **Audit First**: Search existing utils/ for required functionality
2. **Reuse Over Rewrite**: Use existing utilities instead of reimplementing
3. **Extend Carefully**: Only add new utilities if functionality doesn't exist
4. **Document Dependencies**: Clearly document which utilities are used where

### Migration Strategy

1. **Phase 1**: Audit existing utilities and create new components alongside existing ones
2. **Phase 2**: Update ArcEnvironment to use ExperimentLogger
3. **Phase 3**: Remove obsolete components
4. **Phase 4**: Clean up imports and documentation

### Configuration Compatibility

```python
# Existing debug configuration continues to work
debug:
  level: "standard"  # off, minimal, standard, verbose, full
  output_dir: "outputs/debug"
  
# Existing wandb configuration continues to work  
wandb:
  enabled: true
  project_name: "jaxarc-experiments"
  entity: "research-team"
```

### Performance Considerations

- **Synchronous Logging**: Acceptable for research use case
- **Handler Isolation**: Failed handlers don't affect others
- **JAX Compatibility**: Maintained through existing callback mechanism
- **Memory Usage**: Simplified architecture reduces memory overhead

### Backward Compatibility

- Existing configuration files continue to work
- JAX callback interface unchanged
- SVG output format and quality preserved
- File logging format maintained for analysis tools