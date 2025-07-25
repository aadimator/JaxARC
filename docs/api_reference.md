# API Reference

Complete reference for JaxARC parsers, configuration, and core functionality
with practical examples.

## Parser Classes

### ArcAgiParser

Parser for ARC-AGI datasets (2024/2025) from GitHub repositories.

```python
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig
import jax

# Basic usage
config = DictConfig(
    {
        "training": {"path": "data/raw/ARC-AGI-1/data/training"},
        "evaluation": {"path": "data/raw/ARC-AGI-1/data/evaluation"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 10,
        "max_test_pairs": 3,
    }
)

parser = ArcAgiParser(config)

# Get tasks
task_ids = parser.get_available_task_ids()  # List all task IDs
task = parser.get_task_by_id("007bbfb7")  # Get specific task
random_task = parser.get_random_task(jax.random.PRNGKey(42))  # Random task
```

**Key Methods:**

- `get_available_task_ids() -> list[str]`: List all available task IDs
- `get_task_by_id(task_id: str) -> JaxArcTask`: Get specific task
- `get_random_task(key: chex.PRNGKey) -> JaxArcTask`: Get random task

**Supported Datasets:**

- **ARC-AGI-1**: 400 train + 400 eval tasks (`fchollet/ARC-AGI`)
- **ARC-AGI-2**: 1000 train + 120 eval tasks (`arcprize/ARC-AGI-2`)

### ConceptArcParser

Parser for ConceptARC dataset with concept-based task organization.

```python
from jaxarc.parsers import ConceptArcParser
from omegaconf import DictConfig
import jax

# Basic usage
config = DictConfig(
    {
        "corpus": {"path": "data/raw/ConceptARC/corpus"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 4,
        "max_test_pairs": 3,
    }
)

parser = ConceptArcParser(config)

# Explore concepts
concepts = parser.get_concept_groups()  # List all concept groups
tasks_in_center = parser.get_tasks_in_concept("Center")  # Tasks in concept
task = parser.get_random_task_from_concept("Center", jax.random.PRNGKey(42))
```

**Key Methods:**

- `get_concept_groups() -> list[str]`: List available concept groups
- `get_random_task_from_concept(concept, key) -> JaxArcTask`: Random task from
  concept
- `get_tasks_in_concept(concept: str) -> list[str]`: All tasks in concept group

**Concept Groups:** Spatial (Center, AboveBelow), Pattern (Copy, Order), Object
(ExtractObjects), Property (Count, CleanUp)

### MiniArcParser

Parser for MiniARC dataset optimized for 5x5 grids and rapid prototyping.

```python
from jaxarc.parsers import MiniArcParser
from omegaconf import DictConfig
import jax

# Basic usage
config = DictConfig(
    {
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
        "grid": {"max_grid_height": 5, "max_grid_width": 5},
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

parser = MiniArcParser(config)

# Get tasks (5x5 optimized)
task_ids = parser.get_available_task_ids()  # List all task IDs
task = parser.get_random_task(jax.random.PRNGKey(42))  # Random 5x5 task
stats = parser.get_dataset_statistics()  # Performance metrics
```

**Key Methods:**

- `get_available_task_ids() -> list[str]`: List all available task IDs
- `get_random_task(key: chex.PRNGKey) -> JaxArcTask`: Get random 5x5 task
- `get_dataset_statistics() -> dict`: Performance and optimization metrics

**Performance Benefits:** 36x less memory, 10-50x faster processing, larger
batch sizes

## Unified Configuration System

### ConfigFactory

Centralized factory for creating configurations with consistent patterns.

```python
from jaxarc.envs.factory import ConfigFactory
from jaxarc.envs.config import JaxArcConfig

# Preset-based configuration creation
dev_config = ConfigFactory.create_development_config()
research_config = ConfigFactory.create_research_config()
production_config = ConfigFactory.create_production_config()

# Flexible preset system with overrides
config = ConfigFactory.from_preset(
    "development",
    {
        "environment.max_episode_steps": 200,
        "debug.level": "verbose",
        "visualization.show_coordinates": True,
    },
)


# Convert from Hydra configuration
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    config = ConfigFactory.from_hydra(cfg)
    env = ArcEnvironment(config)


# Load from YAML file
config = ConfigFactory.from_yaml("my_config.yaml")
```

**Factory Methods:**

- `create_development_config(**overrides) -> JaxArcConfig`: Development preset
  with debugging enabled
- `create_research_config(**overrides) -> JaxArcConfig`: Research preset with
  full logging
- `create_production_config(**overrides) -> JaxArcConfig`: Production preset
  with minimal overhead
- `from_preset(name: str, overrides: dict) -> JaxArcConfig`: Load named preset
  with overrides
- `from_hydra(hydra_config: DictConfig) -> JaxArcConfig`: Convert Hydra config
  to unified config
- `from_yaml(yaml_path: str) -> JaxArcConfig`: Load configuration from YAML file

### JaxArcConfig

Unified configuration object containing all system parameters in logical groups, now enhanced with episode management, action history, and action space control:

```python
from jaxarc.envs.config import (
    JaxArcConfig,
    EnvironmentConfig,
    DebugConfig,
    VisualizationConfig,
    StorageConfig,
    LoggingConfig,
    WandbConfig,
    ArcEpisodeConfig,
    HistoryConfig,
    ActionConfig,
    ObservationConfig,
)

# Complete configuration with all enhanced groups
config = JaxArcConfig(
    environment=EnvironmentConfig(
        max_episode_steps=100, grid_size=(30, 30), reward_on_submit_only=False
    ),
    debug=DebugConfig(
        level="standard",  # off, minimal, standard, verbose, research
        log_steps=True,
        log_grids=False,
        save_episodes=False,
    ),
    visualization=VisualizationConfig(
        enabled=True,
        level="standard",
        show_grids=True,
        show_actions=True,
        color_scheme="default",
    ),
    storage=StorageConfig(
        policy="standard",  # none, minimal, standard, research
        max_size_gb=5.0,
        cleanup_on_exit=True,
        compression=True,
    ),
    logging=LoggingConfig(level="INFO", console_logging=True, file_logging=False),
    wandb=WandbConfig(enabled=False, project="jaxarc", entity=None),
    
    # Enhanced configuration groups
    episode=ArcEpisodeConfig(
        episode_mode="train",
        demo_selection_strategy="random",
        allow_demo_switching=True,
        require_all_demos_solved=False,
        terminate_on_first_success=False,
        max_pairs_per_episode=4,
        training_reward_frequency="step",
        evaluation_reward_frequency="submit"
    ),
    history=HistoryConfig(
        enabled=True,
        max_history_length=1000,
        store_selection_data=True,
        store_intermediate_grids=False,
        compress_repeated_actions=True
    ),
    action=ActionConfig(
        selection_format="mask",
        selection_threshold=0.5,
        allow_partial_selection=True,
        max_operations=42,  # Updated for enhanced operations
        allowed_operations=None,
        validate_actions=True,
        allow_invalid_actions=False,
        dynamic_action_filtering=True,
        context_dependent_operations=True
    ),
    observation=ObservationConfig(
        include_target_grid=True,
        include_completion_status=True,
        include_action_space_info=True,
        include_recent_actions=False,
        recent_action_count=10,
        include_step_count=True,
        observation_format="standard",
        mask_internal_state=True
    )
)

# Configuration validation
validation_result = config.validate()
if validation_result.errors:
    for error in validation_result.errors:
        print(f"Error: {error}")

# YAML export/import
yaml_content = config.to_yaml()
loaded_config = JaxArcConfig.from_yaml("config.yaml")
```

**Configuration Groups:**

- `EnvironmentConfig`: Core environment parameters (steps, grid size, rewards)
- `DebugConfig`: Debug levels and logging options
- `VisualizationConfig`: Visualization settings and output formats
- `StorageConfig`: Storage policies and resource limits
- `LoggingConfig`: Logging levels and output destinations
- `WandbConfig`: Weights & Biases integration settings
- `ArcEpisodeConfig`: Episode management and multi-demonstration settings
- `HistoryConfig`: Action history tracking configuration
- `ActionConfig`: Enhanced action space control and validation
- `ObservationConfig`: Agent observation space configuration

## Core Data Types

### ArcEnvState (Equinox Module)

The centralized environment state using Equinox for better JAX integration, now enhanced with multi-demonstration support, action history tracking, and dynamic action space control:

```python
from jaxarc.state import ArcEnvState
from jaxarc.utils.jax_types import (
    GridArray, MaskArray, SimilarityScore, EpisodeMode, 
    AvailablePairs, CompletionStatus, ActionHistory, 
    HistoryLength, OperationMask
)
import equinox as eqx


class ArcEnvState(eqx.Module):
    """Enhanced ARC environment state with multi-demonstration and action tracking support."""

    # Core ARC state with JAXTyping annotations
    task_data: JaxArcTask
    working_grid: GridArray  # Int[Array, "height width"]
    working_grid_mask: MaskArray  # Bool[Array, "height width"]
    target_grid: GridArray

    # Episode management
    step_count: StepCount  # Int[Array, ""]
    episode_done: EpisodeDone  # Bool[Array, ""]
    current_example_idx: EpisodeIndex

    # Grid operations
    selected: SelectionArray  # Bool[Array, "height width"]
    clipboard: GridArray
    similarity_score: SimilarityScore  # Float[Array, ""]

    # Enhanced multi-demonstration support
    episode_mode: EpisodeMode  # Int[Array, ""] - 0=train, 1=test
    available_demo_pairs: AvailablePairs  # Bool[Array, "max_pairs"]
    available_test_pairs: AvailablePairs  # Bool[Array, "max_pairs"]
    demo_completion_status: CompletionStatus  # Bool[Array, "max_pairs"]
    test_completion_status: CompletionStatus  # Bool[Array, "max_pairs"]

    # Action history tracking
    action_history: ActionHistory  # Structured action records
    action_history_length: HistoryLength  # Int[Array, ""]

    # Dynamic action space control
    allowed_operations_mask: OperationMask  # Bool[Array, "num_operations"]


# Usage examples
import jax.numpy as jnp

# Create state
state = ArcEnvState(
    task_data=task,
    working_grid=jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
    working_grid_mask=jnp.ones((2, 2), dtype=bool),
    target_grid=jnp.array([[4, 3], [2, 1]], dtype=jnp.int32),
    step_count=jnp.array(0, dtype=jnp.int32),
    episode_done=jnp.array(False, dtype=bool),
    current_example_idx=jnp.array(0, dtype=jnp.int32),
    selected=jnp.zeros((2, 2), dtype=bool),
    clipboard=jnp.zeros((2, 2), dtype=jnp.int32),
    similarity_score=jnp.array(0.0, dtype=jnp.float32),
)

# Update state using Equinox patterns
import equinox as eqx

# Single field update
new_state = eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)

# Multiple field update
new_state = eqx.tree_at(
    lambda s: (s.step_count, s.episode_done),
    state,
    (state.step_count + 1, jnp.array(True)),
)

# Convenience replace method
new_state = state.replace(step_count=state.step_count + 1, episode_done=True)
```

### JAXTyping Type Definitions

Precise array type annotations for better type safety, including enhanced types for multi-demonstration support:

```python
from jaxarc.utils.jax_types import (
    # Grid types (support both single and batched operations)
    GridArray,  # Int[Array, "*batch height width"]
    MaskArray,  # Bool[Array, "*batch height width"]
    SelectionArray,  # Bool[Array, "*batch height width"]
    # Action types
    PointCoords,  # Int[Array, "2"] - [row, col]
    BboxCoords,  # Int[Array, "4"] - [r1, c1, r2, c2]
    OperationId,  # Int[Array, ""] - scalar operation ID
    # Scoring and state types
    SimilarityScore,  # Float[Array, "*batch"]
    StepCount,  # Int[Array, ""]
    EpisodeIndex,  # Int[Array, ""]
    EpisodeDone,  # Bool[Array, ""]
    
    # Enhanced multi-demonstration types
    EpisodeMode,  # Int[Array, ""] - 0=train, 1=test
    AvailablePairs,  # Bool[Array, "max_pairs"] - mask of available pairs
    CompletionStatus,  # Bool[Array, "max_pairs"] - per-pair completion
    
    # Action history types
    ActionHistory,  # Structured action records with fixed size
    HistoryLength,  # Int[Array, ""] - current history length
    ActionRecord,  # Single action record with metadata
    
    # Dynamic action space types
    OperationMask,  # Bool[Array, "num_operations"] - allowed operations
)


# Usage with precise type annotations
def compute_similarity(grid1: GridArray, grid2: GridArray) -> SimilarityScore:
    """Compute similarity with automatic shape validation."""
    diff = jnp.abs(grid1 - grid2)
    return 1.0 - jnp.mean(diff) / 9.0


def process_point_action(
    point: PointCoords, operation: OperationId, grid_shape: tuple[int, int]
) -> SelectionArray:
    """Convert point action to selection mask."""
    row, col = point
    selection = jnp.zeros(grid_shape, dtype=bool)
    return selection.at[row, col].set(True)


# Batch operations work with the same types
batch_grids: GridArray = jnp.stack([grid1, grid2])  # Shape: (2, height, width)
batch_similarities: SimilarityScore = jax.vmap(compute_similarity)(
    batch_grids, batch_grids
)
```

### JaxArcTask

Main data structure for ARC tasks with static shapes for JAX compatibility:

```python
from jaxarc.types import JaxArcTask
import chex


@chex.dataclass
class JaxArcTask:
    input_grids_examples: chex.Array  # (max_train_pairs, H, W)
    output_grids_examples: chex.Array  # (max_train_pairs, H, W)
    num_train_pairs: int

    test_input_grids: chex.Array  # (max_test_pairs, H, W)
    true_test_output_grids: chex.Array  # (max_test_pairs, H, W)
    num_test_pairs: int

    task_index: chex.Array  # Unique identifier
```

## Equinox Utilities

Enhanced utilities for working with Equinox modules and state debugging:

```python
from jaxarc.utils.equinox_utils import (
    tree_map_with_path,
    validate_state_shapes,
    create_state_diff,
    print_state_summary,
    module_memory_usage,
)

# State validation
if validate_state_shapes(state):
    print("✅ State validation passed")
else:
    print("❌ State validation failed")

# State debugging
print_state_summary(state, "Current State")


# Tree traversal with path information
def debug_arrays(path: str, value: Any) -> Any:
    if hasattr(value, "shape"):
        print(f"{path}: shape={value.shape}, dtype={value.dtype}")
    return value


tree_map_with_path(debug_arrays, state)

# State comparison
diff = create_state_diff(old_state, new_state)
for path, change_info in diff.items():
    print(f"Changed: {path}")
    print(f"  Type: {change_info['type']}")
    if "old" in change_info:
        print(f"  Old: {change_info['old']}")
    if "new" in change_info:
        print(f"  New: {change_info['new']}")

# Memory analysis
memory_info = module_memory_usage(state)
print(f"Total memory: {memory_info['total_bytes']:,} bytes")
print(f"Total elements: {memory_info['total_elements']:,}")
```

**Key Functions:**

- `validate_state_shapes(state) -> bool`: Validate Equinox module structure
- `print_state_summary(state, name)`: Print comprehensive state summary
- `tree_map_with_path(fn, tree)`: Map function over tree with path information
- `create_state_diff(old, new) -> dict`: Create diff between two states
- `module_memory_usage(module) -> dict`: Analyze memory usage of Equinox module

## Configuration Validation and Error Handling

Comprehensive validation system with clear error messages and suggestions.

```python
from jaxarc.envs.config import JaxArcConfig, ConfigValidationError

# Configuration validation
try:
    config = JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=-10),  # Invalid
        debug=DebugConfig(level="invalid_level"),  # Invalid
        storage=StorageConfig(max_size_gb=-1.0),  # Invalid
    )
except ConfigValidationError as e:
    print("Configuration validation failed:")
    for field, error in e.field_errors.items():
        print(f"  {field}: {error}")

    # Example output:
    # environment.max_episode_steps: Must be positive, got -10
    # debug.level: Must be one of ['off', 'minimal', 'standard', 'verbose', 'research'], got 'invalid_level'
    # storage.max_size_gb: Must be non-negative, got -1.0

# Validation with warnings
validation_result = config.validate()
for warning in validation_result.warnings:
    print(f"Warning: {warning}")
for error in validation_result.errors:
    print(f"Error: {error}")

# Cross-configuration validation
if config.debug.level == "off" and config.visualization.enabled:
    print("Warning: Visualization enabled but debug level is 'off'")
```

**Validation Features:**

- **Field-level validation**: Range checking, type validation, choice validation
- **Cross-field consistency**: Detect conflicting parameter combinations
- **Clear error messages**: Specific field names and expected values
- **Helpful warnings**: Potentially problematic but valid configurations
- **Validation reports**: Structured validation results with errors and warnings

**Configuration Migration:**

```python
from jaxarc.envs.migration import ConfigMigrator

# Migrate legacy configurations
migrator = ConfigMigrator()

# Migrate dual config pattern
legacy_env_config = {"max_episode_steps": 100}
legacy_hydra_config = {"debug": {"enabled": True}}

new_config = migrator.migrate_dual_config(legacy_env_config, legacy_hydra_config)

# Migration report
report = migrator.create_migration_report(legacy_config)
for warning in report.warnings:
    print(f"Migration warning: {warning}")
```

## Enhanced Environment Features

### Episode Management System

The enhanced environment supports sophisticated episode management with multi-demonstration training and test pair evaluation:

```python
from jaxarc.envs.episode_manager import ArcEpisodeManager, ArcEpisodeConfig
from jaxarc.envs import arc_reset, arc_step
import jax

# Episode configuration for multi-demonstration training
episode_config = ArcEpisodeConfig(
    episode_mode="train",  # "train" or "test"
    demo_selection_strategy="random",  # "sequential" or "random"
    allow_demo_switching=True,
    require_all_demos_solved=False,
    terminate_on_first_success=False,
    max_pairs_per_episode=4,
    training_reward_frequency="step",  # "step" or "submit"
    evaluation_reward_frequency="submit"
)

# Initialize episode manager
episode_manager = ArcEpisodeManager()

# Reset with specific episode mode and pair selection
key = jax.random.PRNGKey(42)
state, observation = arc_reset(
    key, 
    config, 
    task_data=task,
    episode_mode="train",  # Start in training mode
    initial_pair_idx=None  # Let manager select initial pair
)

# Enhanced step function with episode management
action = {"selection": selection_mask, "operation": 35}  # SWITCH_TO_NEXT_DEMO_PAIR
state, observation, reward, done, info = arc_step(state, action, config)

# Check episode continuation
should_continue = episode_manager.should_continue_episode(state, episode_config)
```

**Episode Management Configuration Options:**

- `episode_mode`: "train" (access to targets) or "test" (no target access)
- `demo_selection_strategy`: How to select initial demonstration pairs
- `allow_demo_switching`: Enable switching between demonstration pairs
- `require_all_demos_solved`: Episode termination criteria
- `terminate_on_first_success`: Stop after first successful solution
- `max_pairs_per_episode`: Limit pairs processed per episode
- `training_reward_frequency`: When to calculate rewards in training
- `evaluation_reward_frequency`: When to calculate rewards in evaluation

**New Control Operations (35-41):**

```python
# Non-parametric pair control operations
ENHANCED_OPERATIONS = {
    35: "SWITCH_TO_NEXT_DEMO_PAIR",     # Move to next available demo
    36: "SWITCH_TO_PREV_DEMO_PAIR",     # Move to previous demo
    37: "SWITCH_TO_NEXT_TEST_PAIR",     # Move to next test pair
    38: "SWITCH_TO_PREV_TEST_PAIR",     # Move to previous test pair
    39: "RESET_CURRENT_PAIR",           # Reset current pair to initial state
    40: "SWITCH_TO_FIRST_UNSOLVED_DEMO", # Jump to first unsolved demo
    41: "SWITCH_TO_FIRST_UNSOLVED_TEST", # Jump to first unsolved test
}

# Usage example
action = {
    "selection": jnp.zeros((30, 30), dtype=bool),  # Selection ignored for control ops
    "operation": 35  # SWITCH_TO_NEXT_DEMO_PAIR
}
```

### Action History Tracking

Comprehensive action history tracking with configurable storage and JAX compatibility:

```python
from jaxarc.envs.action_history import ActionHistoryTracker, HistoryConfig, ActionRecord

# History configuration
history_config = HistoryConfig(
    enabled=True,
    max_history_length=1000,
    store_selection_data=True,  # Store full selection masks
    store_intermediate_grids=False,  # Memory-intensive option
    compress_repeated_actions=True
)

# Initialize history tracker
history_tracker = ActionHistoryTracker()

# Action history is automatically tracked during stepping
state, observation, reward, done, info = arc_step(state, action, config)

# Access action history
action_sequence = history_tracker.get_action_sequence(
    state, 
    start_idx=0, 
    end_idx=10  # Get last 10 actions
)

# Action record structure
action_record = ActionRecord(
    selection_data=selection_mask,  # Full selection information
    operation_id=operation_id,      # Operation that was executed
    timestamp=step_count,           # When action was taken
    pair_index=current_pair_idx,    # Which demo/test pair
    valid=True                      # Whether record contains valid data
)

# Clear history for new episode
state = history_tracker.clear_history(state)
```

**History Configuration Options:**

- `enabled`: Enable/disable action history tracking
- `max_history_length`: Maximum number of actions to store
- `store_selection_data`: Whether to store full selection masks (memory-intensive)
- `store_intermediate_grids`: Store grid states after each action (very memory-intensive)
- `compress_repeated_actions`: Reduce storage for repeated operations

**Memory Optimization:**

```python
# Memory-efficient configuration for production
efficient_history = HistoryConfig(
    max_history_length=100,
    store_selection_data=False,  # Only store operation IDs
    store_intermediate_grids=False,
    compress_repeated_actions=True
)

# Research configuration with full tracking
research_history = HistoryConfig(
    max_history_length=2000,
    store_selection_data=True,
    store_intermediate_grids=True,  # Enable for detailed analysis
    compress_repeated_actions=False
)
```

### Action Space Control

Dynamic action space control with context-aware operation filtering:

```python
from jaxarc.envs.action_space import ActionSpaceController, ActionConfig

# Enhanced action configuration
action_config = ActionConfig(
    # Existing configuration options
    selection_format="mask",
    selection_threshold=0.5,
    allow_partial_selection=True,
    max_operations=42,  # Updated to include new control operations
    allowed_operations=None,  # None = all operations allowed
    validate_actions=True,
    allow_invalid_actions=False,
    
    # New dynamic control options
    dynamic_action_filtering=True,  # Enable runtime filtering
    context_dependent_operations=True  # Context-aware availability
)

# Initialize action space controller
action_controller = ActionSpaceController()

# Get currently allowed operations
allowed_mask = action_controller.get_allowed_operations(state, action_config)

# Validate specific operation
is_valid, error_msg = action_controller.validate_operation(
    operation_id=35,  # SWITCH_TO_NEXT_DEMO_PAIR
    state=state,
    config=action_config
)

# Filter invalid operations according to policy
filtered_operation = action_controller.filter_invalid_operation(
    operation_id=invalid_op,
    state=state,
    config=action_config
)
```

**Context-Aware Operation Filtering:**

- Demo pair switching only available in train mode with multiple demos
- Test pair switching only available in test mode with multiple tests
- Pair reset only available if current pair has been modified
- Operation availability updates dynamically based on state

**Action Validation Policies:**

```python
# Different handling of invalid operations
action_config_reject = ActionConfig(
    validate_actions=True,
    allow_invalid_actions=False,  # Raise error on invalid operations
    dynamic_action_filtering=True
)

action_config_clip = ActionConfig(
    validate_actions=True,
    allow_invalid_actions=True,  # Clip to nearest valid operation
    dynamic_action_filtering=True
)

action_config_penalize = ActionConfig(
    validate_actions=True,
    allow_invalid_actions=True,  # Allow but apply penalty
    dynamic_action_filtering=False  # Let agent learn from mistakes
)
```

### Enhanced Observation Space

The environment now provides a focused `ArcObservation` structure separate from the full internal state:

```python
from jaxarc.envs.observation import ArcObservation, ObservationConfig

# Observation configuration
obs_config = ObservationConfig(
    include_target_grid=True,  # Include target in train mode
    include_completion_status=True,  # Show progress tracking
    include_action_space_info=True,  # Show allowed operations
    include_recent_actions=False,  # Include recent action history
    recent_action_count=10,
    include_step_count=True,
    observation_format="standard",  # "minimal", "standard", "rich"
    mask_internal_state=True  # Hide implementation details
)

# ArcObservation structure
@chex.dataclass
class ArcObservation:
    """Focused agent observation space."""
    
    # Core grid information
    working_grid: GridArray
    working_grid_mask: MaskArray
    
    # Episode context
    episode_mode: EpisodeMode  # 0=train, 1=test
    current_pair_idx: EpisodeIndex
    step_count: StepCount
    
    # Progress tracking
    demo_completion_status: CompletionStatus
    test_completion_status: CompletionStatus
    
    # Action space information
    allowed_operations_mask: OperationMask
    
    # Optional components (configurable)
    target_grid: Optional[GridArray] = None  # Only in train mode
    recent_actions: Optional[ActionHistory] = None
```

**Observation vs State Separation:**

- **ArcObservation**: Clean, focused view for agents with configurable information
- **ArcEnvState**: Complete internal state with all implementation details
- **Information Hiding**: Agents only see relevant information, not internal bookkeeping
- **Research Flexibility**: Different observation configurations for different experiments

## Environment Integration

### Unified Configuration Pattern

```python
from jaxarc.envs import ArcEnvironment
from jaxarc.envs.factory import ConfigFactory

# Single configuration object (replaces dual config pattern)
config = ConfigFactory.create_development_config(
    max_episode_steps=100, debug_level="standard", visualization_enabled=True
)

# Environment uses single config
env = ArcEnvironment(config)  # No hydra_config parameter needed

# Access configuration parameters through logical groups
max_steps = config.environment.max_episode_steps
debug_level = config.debug.level
viz_enabled = config.visualization.enabled
```

### Enhanced Functional API

```python
from jaxarc.envs import arc_reset, arc_step, create_observation

# Enhanced reset with episode mode and pair selection
state, observation = arc_reset(
    key, 
    config, 
    task_data=task,
    episode_mode="train",  # "train" or "test"
    initial_pair_idx=None  # Let episode manager select, or specify index
)

# Enhanced step with separate state and observation
state, observation, reward, done, info = arc_step(state, action, config)

# Manual observation creation from state
observation = create_observation(state, config.observation)

# Access enhanced state information
print(f"Episode mode: {state.episode_mode}")  # 0=train, 1=test
print(f"Current pair: {state.current_example_idx}")
print(f"Demo completion: {state.demo_completion_status}")
print(f"Action history length: {state.action_history_length}")
print(f"Allowed operations: {state.allowed_operations_mask.sum()} of 42")

# Access focused observation information
print(f"Working grid shape: {observation.working_grid.shape}")
print(f"Target available: {observation.target_grid is not None}")
print(f"Recent actions: {observation.recent_actions is not None}")
```

### Class-Based API

```python
from jaxarc.envs import ArcEnvironment

# Single configuration parameter
env = ArcEnvironment(config)
state, obs = env.reset(key, task_data=task)
state, obs, reward, info = env.step(action)

# Configuration access within environment
assert env.config.environment.max_episode_steps == config.environment.max_episode_steps
assert env.config.debug.level == config.debug.level
```

## Usage Examples

### ConceptARC Example

```python
import jax
from jaxarc.parsers import ConceptArcParser
from jaxarc.envs import ArcEnvironment
from jaxarc.envs.factory import ConfigFactory
from omegaconf import DictConfig

# Create parser configuration
parser_config = DictConfig(
    {
        "corpus": {"path": "data/raw/ConceptARC/corpus"},
        "grid": {"max_grid_height": 30, "max_grid_width": 30},
        "max_train_pairs": 4,
        "max_test_pairs": 3,
    }
)

# Initialize parser
parser = ConceptArcParser(parser_config)

# Explore concept groups
concepts = parser.get_concept_groups()
print(f"Available concepts: {concepts}")

# Get task from specific concept
key = jax.random.PRNGKey(42)
task = parser.get_random_task_from_concept("Center", key)

# Create unified environment configuration
config = ConfigFactory.create_research_config(
    max_episode_steps=150, debug_level="verbose", visualization_enabled=True
)

# Run environment with single config
env = ArcEnvironment(config)
state, obs = env.reset(key, task_data=task)

# Access configuration parameters through groups
print(f"Max steps: {config.environment.max_episode_steps}")
print(f"Debug level: {config.debug.level}")
print(f"Visualization: {config.visualization.enabled}")
```

### MiniARC Example

```python
import jax
from jaxarc.parsers import MiniArcParser
from jaxarc.envs import ArcEnvironment
from jaxarc.envs.factory import ConfigFactory
from omegaconf import DictConfig

# Create parser configuration
parser_config = DictConfig(
    {
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
        "grid": {"max_grid_height": 5, "max_grid_width": 5},
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

# Initialize parser
parser = MiniArcParser(parser_config)

# Get random task (5x5 optimized)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

# Create optimized configuration for rapid iteration
config = ConfigFactory.from_preset(
    "development",
    {
        "environment.max_episode_steps": 50,
        "environment.grid_size": (5, 5),
        "debug.level": "minimal",
        "storage.policy": "minimal",
    },
)

# Run environment (10-50x faster than standard ARC)
env = ArcEnvironment(config)
state, obs = env.reset(key, task_data=task)

# Configuration is optimized for MiniARC
assert config.environment.grid_size == (5, 5)
assert config.debug.level == "minimal"
```

### Enhanced Multi-Demonstration Training Example

```python
import jax
import jax.numpy as jnp
from jaxarc.envs import ArcEnvironment, arc_reset, arc_step
from jaxarc.envs.factory import ConfigFactory
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig

# Create enhanced configuration for multi-demonstration training
config = ConfigFactory.create_research_config(
    # Episode management
    episode_mode="train",
    demo_selection_strategy="random",
    allow_demo_switching=True,
    max_pairs_per_episode=3,
    
    # Action history tracking
    history_enabled=True,
    max_history_length=500,
    store_selection_data=True,
    
    # Dynamic action space
    dynamic_action_filtering=True,
    context_dependent_operations=True,
    
    # Observation configuration
    include_completion_status=True,
    include_action_space_info=True,
    observation_format="rich"
)

# Load task with multiple demonstration pairs
parser_config = DictConfig({
    "training": {"path": "data/raw/ARC-AGI-1/data/training"},
    "grid": {"max_grid_height": 30, "max_grid_width": 30},
    "max_train_pairs": 4,  # Use multiple demonstration pairs
    "max_test_pairs": 2,
})
parser = ArcAgiParser(parser_config)
task = parser.get_random_task(jax.random.PRNGKey(42))

# Initialize environment with enhanced features
env = ArcEnvironment(config)
key = jax.random.PRNGKey(123)

# Reset in training mode - will select initial demo pair
state, observation = arc_reset(key, config, task_data=task, episode_mode="train")

print(f"Started in episode mode: {state.episode_mode}")  # 0 = train
print(f"Current demo pair: {state.current_example_idx}")
print(f"Available demo pairs: {state.available_demo_pairs.sum()}")
print(f"Target grid available: {observation.target_grid is not None}")
print(f"Allowed operations: {observation.allowed_operations_mask.sum()} of 42")

# Training loop with multi-demonstration support
for step in range(100):
    # Regular grid operation
    if step < 50:
        action = {
            "selection": jnp.zeros((30, 30), dtype=bool).at[5:10, 5:10].set(True),
            "operation": 0  # FILL operation
        }
    # Switch to next demonstration pair
    elif step == 50:
        action = {
            "selection": jnp.zeros((30, 30), dtype=bool),  # Ignored for control ops
            "operation": 35  # SWITCH_TO_NEXT_DEMO_PAIR
        }
        print(f"Switching to next demo pair at step {step}")
    # Continue with new pair
    else:
        action = {
            "selection": jnp.zeros((30, 30), dtype=bool).at[0:5, 0:5].set(True),
            "operation": 1  # Different operation on new pair
        }
    
    # Step environment
    state, observation, reward, done, info = arc_step(state, action, config)
    
    # Check if we switched pairs
    if step == 50:
        print(f"Now on demo pair: {state.current_example_idx}")
        print(f"Demo completion status: {state.demo_completion_status}")
    
    # Access action history
    if step % 25 == 0:
        print(f"Step {step}: Action history length: {state.action_history_length}")
    
    if done:
        print(f"Episode completed at step {step}")
        break

# Switch to evaluation mode for test pairs
print("\n--- Switching to Evaluation Mode ---")
state, observation = arc_reset(
    jax.random.split(key)[0], 
    config, 
    task_data=task, 
    episode_mode="test"  # Switch to test mode
)

print(f"Evaluation mode: {state.episode_mode}")  # 1 = test
print(f"Current test pair: {state.current_example_idx}")
print(f"Available test pairs: {state.available_test_pairs.sum()}")
print(f"Target grid available: {observation.target_grid is not None}")  # Should be None

# Evaluation loop (no target access)
for step in range(20):
    action = {
        "selection": jnp.zeros((30, 30), dtype=bool).at[step:step+2, step:step+2].set(True),
        "operation": step % 10  # Vary operations
    }
    
    state, observation, reward, done, info = arc_step(state, action, config)
    
    if done:
        print(f"Evaluation completed at step {step}")
        break

print(f"Final action history length: {state.action_history_length}")
```

### Hydra Integration Example

```python
import hydra
from omegaconf import DictConfig
from jaxarc.envs import ArcEnvironment
from jaxarc.envs.factory import ConfigFactory


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert Hydra config to unified JaxArcConfig
    config = ConfigFactory.from_hydra(cfg)

    # Single configuration object contains all enhanced parameters
    env = ArcEnvironment(config)

    # Access parameters through logical groups
    print(f"Environment: max_steps={config.environment.max_episode_steps}")
    print(f"Debug: level={config.debug.level}")
    print(f"Visualization: enabled={config.visualization.enabled}")
    print(f"Storage: policy={config.storage.policy}")
    
    # Access enhanced configuration groups
    print(f"Episode: mode={config.episode.episode_mode}")
    print(f"History: enabled={config.history.enabled}, length={config.history.max_history_length}")
    print(f"Action: dynamic_filtering={config.action.dynamic_action_filtering}")
    print(f"Observation: format={config.observation.observation_format}")

    # Run environment with enhanced features
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)


if __name__ == "__main__":
    main()
```

## Performance Considerations

### ConceptARC

- **Grid Size**: Up to 30×30 (same as standard ARC)
- **Memory Usage**: Standard ARC memory requirements
- **Use Case**: Systematic evaluation and concept-based analysis
- **Optimization**: Concept-based task sampling and organization

### MiniARC

- **Grid Size**: Maximum 5×5 (36x fewer cells than standard ARC)
- **Memory Usage**: 36x less memory per grid
- **Processing Speed**: 10-50x faster than standard ARC
- **Batch Size**: Support for much larger batch sizes
- **Use Case**: Rapid prototyping, algorithm development, quick experiments

### JAX Compatibility

All parsers and configurations are fully compatible with JAX transformations:

```python
# JIT compilation
@jax.jit
def process_task(task, config):
    return some_processing(task, config)


# Vectorization
batch_process = jax.vmap(process_task, in_axes=(0, None))

# Parallel processing
parallel_process = jax.pmap(process_task, in_axes=(0, None))
```

## Error Handling

### Common Exceptions

- **`FileNotFoundError`**: Dataset files not found
- **`ValueError`**: Invalid configuration or task parameters
- **`KeyError`**: Missing concept group or task ID
- **`ValidationError`**: Configuration validation failures
- **`RuntimeError`**: Legacy Kaggle format detected or GitHub download issues

### Best Practices

1. **Validate configurations** before use
2. **Handle missing datasets** gracefully
3. **Use appropriate grid sizes** for each dataset type
4. **Check task availability** before accessing specific tasks
5. **Monitor memory usage** with large batch sizes

## Testing

### Basic Testing Patterns

```python
import pytest
import jax
from jaxarc.parsers import ArcAgiParser
from omegaconf import DictConfig


def test_parser_functionality():
    """Test basic parser operations."""
    config = DictConfig(
        {
            "training": {"path": "data/raw/ARC-AGI-1/data/training"},
            "grid": {"max_grid_height": 30, "max_grid_width": 30},
            "max_train_pairs": 10,
            "max_test_pairs": 3,
        }
    )

    parser = ArcAgiParser(config)
    task_ids = parser.get_available_task_ids()
    assert len(task_ids) > 0

    # Test task loading
    task = parser.get_task_by_id(task_ids[0])
    assert task.num_train_pairs > 0

    # Test random task
    key = jax.random.PRNGKey(42)
    random_task = parser.get_random_task(key)
    assert random_task.task_index is not None


def test_jax_compatibility():
    """Test JAX transformations work correctly."""

    @jax.jit
    def process_task(task):
        return task.input_grids_examples.sum()

    # Test with actual task data
    task = parser.get_random_task(jax.random.PRNGKey(0))
    result = process_task(task)  # Should not raise errors
    assert result is not None
```

### Common Issues

- **FileNotFoundError**: Download datasets with
  `python scripts/download_dataset.py <dataset-name>`
- **Legacy format error**: Update config to use `path` instead of
  `challenges`/`solutions`
- **Memory issues**: Use MiniARC for development or clear parser cache
  periodically
