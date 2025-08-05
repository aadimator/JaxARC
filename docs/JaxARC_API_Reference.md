# JaxARC API Reference for RL Algorithm Development

This document provides a comprehensive reference for developers implementing reinforcement learning algorithms (like PPO, DQN, A2C, etc.) on top of the JaxARC environment. It covers all essential APIs, data structures, and patterns needed for successful integration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Environment API](#core-environment-api)
3. [Configuration System](#configuration-system)
4. [State Management](#state-management)
5. [Action System](#action-system)
6. [Data Types and Structures](#data-types-and-structures)
7. [Task Loading and Management](#task-loading-and-management)
8. [Logging and Experiment Management](#logging-and-experiment-management)
9. [Visualization and Debugging](#visualization-and-debugging)
10. [JAX Integration Patterns](#jax-integration-patterns)
11. [Performance Optimization](#performance-optimization)
12. [Common Patterns for RL Algorithms](#common-patterns-for-rl-algorithms)
13. [Troubleshooting and Best Practices](#troubleshooting-and-best-practices)

## Quick Start

### Basic Environment Setup

```python
import jax
import jax.numpy as jnp
from jaxarc import ArcEnvironment, JaxArcConfig
from jaxarc.parsers import ArcAgiParser

# 1. Load tasks from dataset
parser = ArcAgiParser(data_dir="data/arc-prize-2024")
tasks = parser.load_tasks(split="training")

# 2. Create environment configuration
config = JaxArcConfig()  # Uses sensible defaults

# 3. Initialize environment
env = ArcEnvironment(config)

# 4. Reset environment with a task
key = jax.random.PRNGKey(42)
task_data = tasks[0]  # Use first task
state, obs = env.reset(key, task_data)

# 5. Take actions using factory functions
from jaxarc.envs.structured_actions import create_point_action
action = create_point_action(operation=0, row=5, col=5)  # Fill operation at (5,5)
next_state, next_obs, reward, info = env.step(action)
```

### Functional API (Recommended for RL)

```python
from jaxarc.envs import arc_reset, arc_step
from jaxarc.envs.config import JaxArcConfig

# Functional API is JAX-compatible and stateless
config = JaxArcConfig()
key = jax.random.PRNGKey(42)

# Reset returns (state, observation)
state, obs = arc_reset(key, config, task_data)

# Step returns (state, observation, reward, done, info)
next_state, next_obs, reward, done, info = arc_step(state, action, config)
```

## Core Environment API

### ArcEnvironment Class

The main environment class provides a stateful interface suitable for interactive use and simple RL loops.

```python
class ArcEnvironment:
    def __init__(self, config: JaxArcConfig)
    
    def reset(self, key: chex.PRNGKey, task_data: Optional[JaxArcTask] = None) -> Tuple[ArcEnvState, jnp.ndarray]
    
    def step(self, action: StructuredAction) -> Tuple[ArcEnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]
    
    def close(self) -> None
    
    # Properties
    @property
    def state(self) -> Optional[ArcEnvState]
    @property
    def is_done(self) -> bool
    
    # Space information
    def get_observation_space_info(self) -> Dict[str, Any]
    def get_action_space_info(self) -> Dict[str, Any]
```

### Functional API (Recommended)

The functional API is pure, stateless, and fully JAX-compatible:

```python
def arc_reset(
    key: PRNGKey, 
    config: JaxArcConfig, 
    task_data: Optional[JaxArcTask] = None
) -> Tuple[ArcEnvState, ObservationArray]

def arc_step(
    state: ArcEnvState, 
    action: StructuredAction, 
    config: JaxArcConfig
) -> Tuple[ArcEnvState, ObservationArray, RewardValue, EpisodeDone, Dict[str, Any]]
```

**Key Benefits for RL:**
- Pure functions enable `jax.jit` compilation
- Stateless design works with `jax.vmap` for batch processing
- No side effects or hidden state
- Deterministic given same inputs

## Configuration System

### JaxArcConfig - Unified Configuration

```python
@dataclass
class JaxArcConfig(eqx.Module):
    environment: EnvironmentConfig
    dataset: DatasetConfig
    action: ActionConfig
    reward: RewardConfig
    visualization: VisualizationConfig
    storage: StorageConfig
    logging: LoggingConfig
    wandb: WandbConfig
    episode: ArcEpisodeConfig
    history: HistoryConfig
```

### Key Configuration Sections

#### Environment Configuration
```python
class EnvironmentConfig(eqx.Module):
    max_episode_steps: int = 100
    auto_reset: bool = True
    strict_validation: bool = True
    allow_invalid_actions: bool = False
    debug_level: Literal["off", "minimal", "standard", "verbose", "research"] = "standard"
```

#### Dataset Configuration
```python
class DatasetConfig(eqx.Module):
    dataset_name: str = "arc-agi-1"
    max_grid_height: int = 30
    max_grid_width: int = 30
    max_colors: int = 10
    max_train_pairs: int = 10
    max_test_pairs: int = 3
    task_split: str = "train"
```

#### Action Configuration
```python
class ActionConfig(eqx.Module):
    selection_format: Literal["point", "bbox", "mask"] = "mask"
    max_operations: int = 42
    validate_actions: bool = True
    allow_invalid_actions: bool = False
```

#### Reward Configuration
```python
class RewardConfig(eqx.Module):
    similarity_weight: float = 1.0
    step_penalty: float = -0.01
    success_bonus: float = 10.0
    progress_bonus: float = 0.1
    reward_on_submit_only: bool = False
    # Enhanced reward components
    training_similarity_weight: float = 1.0
    demo_completion_bonus: float = 5.0
    test_completion_bonus: float = 15.0
    efficiency_bonus: float = 2.0
    efficiency_bonus_threshold: int = 50
```

### Configuration Creation Patterns

```python
# Use defaults
config = JaxArcConfig()

# Customize specific sections
config = JaxArcConfig(
    environment=EnvironmentConfig(max_episode_steps=200),
    reward=RewardConfig(step_penalty=-0.005, success_bonus=20.0),
    action=ActionConfig(selection_format="point")
)

# Load from Hydra (recommended approach)
from jaxarc.utils.config import get_config

# Load with overrides (as shown in miniarc_rl_loop.py)
config_overrides = [
    "dataset=mini_arc",
    "action=raw",
    "action.selection_format=bbox",
    "visualization=full",
    "logging=full",
    "storage=research",
    "wandb=research",
]

hydra_config = get_config(overrides=config_overrides)
typed_config = JaxArcConfig.from_hydra(hydra_config)
```

## State Management

### ArcEnvState Structure

The environment state contains all information needed for RL algorithms:

```python
class ArcEnvState(eqx.Module):
    # Core ARC state
    task_data: JaxArcTask
    working_grid: GridArray  # Current grid being modified
    working_grid_mask: MaskArray  # Valid cells mask
    target_grid: GridArray  # Goal grid for current example
    target_grid_mask: MaskArray  # Valid cells mask for target
    
    # Episode management
    step_count: StepCount
    episode_done: EpisodeDone
    current_example_idx: EpisodeIndex
    
    # Grid operations
    selected: SelectionArray  # Selection mask for operations
    clipboard: GridArray  # For copy/paste operations
    similarity_score: SimilarityScore  # Grid similarity to target (0.0 to 1.0)
    
    # Enhanced functionality
    episode_mode: EpisodeMode  # 0=train, 1=test
    available_demo_pairs: AvailableTrainPairs
    available_test_pairs: AvailableTestPairs
    demo_completion_status: TrainCompletionStatus
    test_completion_status: TestCompletionStatus
    action_history: ActionHistory
    action_history_length: HistoryLength
    allowed_operations_mask: OperationMask
```

### State Utility Methods

```python
# Check episode mode
state.is_training_mode() -> bool
state.is_test_mode() -> bool

# Get completion information
state.get_available_demo_count() -> Int[Array, ""]
state.get_completed_demo_count() -> Int[Array, ""]
state.is_current_pair_completed() -> Bool[Array, ""]

# Grid utilities
state.get_actual_grid_shape() -> tuple[int, int]
state.get_actual_working_grid() -> GridArray
state.get_actual_target_grid() -> GridArray

# Action history
state.add_action_to_history(operation_id, selection_data) -> ArcEnvState
state.get_recent_actions(count) -> list[dict]
```

### State Updates (Functional Pattern)

```python
from jaxarc.utils.pytree_utils import update_multiple_fields

# Update multiple fields efficiently
new_state = update_multiple_fields(
    state,
    step_count=state.step_count + 1,
    similarity_score=new_similarity,
    episode_done=jnp.array(True)
)
```

## Action System

### Action Formats

JaxARC supports three action formats optimized for different use cases:

#### 1. Point Actions (Discrete)
```python
from jaxarc.envs.structured_actions import create_point_action

# Create point action
action = create_point_action(
    operation=0,  # Fill operation (0-9 for different colors)
    row=5,        # Grid row
    col=7         # Grid column
)

# Action space info for RL algorithms
action_space = {
    "type": "discrete",
    "n": grid_height * grid_width * num_operations,
    # Or as MultiDiscrete: [grid_height, grid_width, num_operations]
}
```

#### 2. Bounding Box Actions (Structured Discrete)
```python
from jaxarc.envs.structured_actions import create_bbox_action

# Create bbox action
action = create_bbox_action(
    operation=10,  # Flood fill operation
    r1=2, c1=3,    # Top-left corner
    r2=8, c2=9     # Bottom-right corner
)

# Action space info
action_space = {
    "type": "dict",
    "spaces": {
        "operation": gym.spaces.Discrete(42),
        "r1": gym.spaces.Discrete(grid_height),
        "c1": gym.spaces.Discrete(grid_width),
        "r2": gym.spaces.Discrete(grid_height),
        "c2": gym.spaces.Discrete(grid_width),
    }
}
```

#### 3. Mask Actions (Continuous/Mixed)
```python
from jaxarc.envs.structured_actions import create_mask_action

# Create mask action
selection_mask = jnp.zeros((grid_height, grid_width), dtype=jnp.bool_)
selection_mask = selection_mask.at[5:10, 5:10].set(True)  # Select region

action = create_mask_action(
    operation=0,
    selection=selection_mask
)

# Action space info
action_space = {
    "type": "dict",
    "spaces": {
        "operation": gym.spaces.Discrete(42),
        "selection": gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(grid_height, grid_width), 
            dtype=np.float32
        )
    }
}
```

### Available Operations

JaxARC provides 42 operations (0-41):

```python
# Fill operations (0-9) - Fill selected area with color
FILL_0 = 0    # Fill with black
FILL_1 = 1    # Fill with blue
# ... up to FILL_9 = 9

# Flood fill operations (10-19) - Flood fill from selection
FLOOD_FILL_0 = 10
# ... up to FLOOD_FILL_9 = 19

# Movement operations (20-23)
MOVE_UP = 20
MOVE_DOWN = 21
MOVE_LEFT = 22
MOVE_RIGHT = 23

# Transformation operations (24-27)
ROTATE_C = 24      # Clockwise rotation
ROTATE_CC = 25     # Counter-clockwise rotation
FLIP_HORIZONTAL = 26
FLIP_VERTICAL = 27

# Editing operations (28-33)
COPY = 28
PASTE = 29
CUT = 30
CLEAR = 31
COPY_INPUT = 32    # Copy from input grid
RESIZE = 33

# Control operations (34-41)
SUBMIT = 34                        # Submit current solution
SWITCH_TO_NEXT_DEMO_PAIR = 35     # Switch to next demo pair
SWITCH_TO_PREV_DEMO_PAIR = 36     # Switch to previous demo pair
SWITCH_TO_NEXT_TEST_PAIR = 37     # Switch to next test pair
SWITCH_TO_PREV_TEST_PAIR = 38     # Switch to previous test pair
RESET_CURRENT_PAIR = 39           # Reset current pair
SWITCH_TO_FIRST_UNSOLVED_DEMO = 40
SWITCH_TO_FIRST_UNSOLVED_TEST = 41
```

### Action Validation

```python
# Actions are automatically validated and clipped
from jaxarc.envs.actions import validate_structured_action

validate_structured_action(action, grid_shape=(30, 30))
```

## Data Types and Structures

### Core Data Types

```python
# Grid representation
class Grid(eqx.Module):
    data: GridArray      # Int[Array, "height width"] - color values 0-9
    mask: MaskArray      # Bool[Array, "height width"] - valid cells
    
    @property
    def shape(self) -> tuple[int, int]  # Actual grid dimensions

# Task representation
class JaxArcTask(eqx.Module):
    # Training examples
    input_grids_examples: TaskInputGrids    # Int[Array, "max_pairs height width"]
    input_masks_examples: TaskInputMasks    # Bool[Array, "max_pairs height width"]
    output_grids_examples: TaskOutputGrids  # Int[Array, "max_pairs height width"]
    output_masks_examples: TaskOutputMasks  # Bool[Array, "max_pairs height width"]
    num_train_pairs: int
    
    # Test examples
    test_input_grids: TaskInputGrids
    test_input_masks: TaskInputMasks
    true_test_output_grids: TaskOutputGrids
    true_test_output_masks: TaskOutputMasks
    num_test_pairs: int
    
    task_index: TaskIndex  # Int[Array, ""] - unique task identifier
```

### JAX Type Annotations

JaxARC uses JAXTyping for precise type safety:

```python
from jaxarc.utils.jax_types import (
    GridArray,           # Int[Array, "height width"]
    MaskArray,           # Bool[Array, "height width"]
    SelectionArray,      # Bool[Array, "height width"]
    ObservationArray,    # Float[Array, "..."]
    RewardValue,         # Float[Array, ""]
    EpisodeDone,         # Bool[Array, ""]
    StepCount,           # Int[Array, ""]
    SimilarityScore,     # Float[Array, ""]
    OperationId,         # Int[Array, ""]
    PRNGKey,             # chex.PRNGKey
)
```

### Task Utility Methods

```python
# Access training pairs
task.get_train_pair(pair_idx) -> TaskPair
task.get_train_input_grid(pair_idx) -> Grid
task.get_train_output_grid(pair_idx) -> Grid

# Access test pairs
task.get_test_pair(pair_idx) -> TaskPair
task.get_test_input_grid(pair_idx) -> Grid

# Check availability
task.is_demo_pair_available(pair_idx) -> Bool[Array, ""]
task.is_test_pair_available(pair_idx) -> Bool[Array, ""]

# Get metadata
task.get_task_summary() -> dict
task.get_grid_shape() -> tuple[int, int]
```

## Task Loading and Management

### Dataset Parsers

```python
from jaxarc.parsers import ArcAgiParser, ConceptArcParser, MiniArcParser

# ARC-AGI dataset (official competition data)
parser = ArcAgiParser(data_dir="data/arc-prize-2024")
training_tasks = parser.load_tasks(split="training")
evaluation_tasks = parser.load_tasks(split="evaluation")

# ConceptARC dataset (additional tasks)
concept_parser = ConceptArcParser(data_dir="data/ConceptARC")
concept_tasks = concept_parser.load_tasks(split="corpus")

# MiniARC dataset (smaller tasks for testing)
mini_parser = MiniArcParser(data_dir="data/MiniARC")
mini_tasks = mini_parser.load_tasks(split="training")
```

### Task Sampling for RL

```python
import jax.random as random

def sample_task(key: chex.PRNGKey, tasks: List[JaxArcTask]) -> JaxArcTask:
    """Sample a random task for training."""
    task_idx = random.randint(key, (), 0, len(tasks))
    return tasks[task_idx]

def create_task_sampler(tasks: List[JaxArcTask]):
    """Create a task sampling function for RL training."""
    def sample_fn(key: chex.PRNGKey) -> JaxArcTask:
        return sample_task(key, tasks)
    return sample_fn

# Usage in RL training loop
key = jax.random.PRNGKey(42)
task_sampler = create_task_sampler(training_tasks)

for episode in range(num_episodes):
    key, task_key = jax.random.split(key)
    task = task_sampler(task_key)
    
    key, reset_key = jax.random.split(key)
    state, obs = arc_reset(reset_key, config, task)
    # ... training loop
```

## Logging and Experiment Management

### ExperimentLogger - Unified Logging System

JaxARC provides a comprehensive logging system through the `ExperimentLogger` class that manages multiple handlers for different output formats:

```python
from jaxarc.utils.logging import ExperimentLogger

# Initialize experiment logger (automatically creates handlers based on config)
experiment_logger = ExperimentLogger(config)

# The logger automatically initializes handlers based on configuration:
# - FileHandler: For structured file logging
# - SVGHandler: For SVG visualization generation
# - ConsoleHandler: For rich console output
# - WandbHandler: For Weights & Biases integration
```

### Logging Configuration

```python
class LoggingConfig(eqx.Module):
    structured_logging: bool = True
    log_format: Literal["json", "text", "structured"] = "json"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_operations: bool = False
    log_grid_changes: bool = False
    log_rewards: bool = False
    log_episode_start: bool = True
    log_episode_end: bool = True
    log_key_moments: bool = True
    log_frequency: int = 10  # Log every N steps

class WandbConfig(eqx.Module):
    enabled: bool = False
    project_name: str = "jaxarc-experiments"
    entity: str | None = None
    tags: tuple[str, ...] = ("jaxarc",)
    log_frequency: int = 10
    log_gradients: bool = False
    log_system_metrics: bool = True
```

### Step-by-Step Logging

```python
def run_rl_loop_with_logging(config, experiment_logger, num_episodes=5):
    """Complete RL loop with proper logging integration."""
    
    # Start a new run
    if 'svg' in experiment_logger.handlers:
        experiment_logger.handlers['svg'].start_run(f"experiment_{int(time.time())}")
    
    for episode_idx in range(num_episodes):
        # Start episode logging
        if 'svg' in experiment_logger.handlers:
            experiment_logger.handlers['svg'].start_episode(episode_idx)
        
        # Reset environment
        state, obs = arc_reset(key, config, task_data)
        
        for step_num in range(max_steps):
            # Take action
            action, agent_state = agent.select_action(agent_state, state, config)
            next_state, next_obs, reward, done, info = arc_step(state, action, config)
            
            # Prepare step data for logging
            step_data = {
                "step_num": step_num,
                "episode_num": episode_idx,
                "before_state": state,
                "after_state": next_state,
                "action": {
                    "operation": int(action.operation),
                    "r1": int(action.r1),
                    "c1": int(action.c1),
                    "r2": int(action.r2),
                    "c2": int(action.c2),
                    "selection": action.to_selection_mask(state.working_grid.shape),
                },
                "reward": float(reward),
                "info": {
                    **info,
                    "metrics": {  # Metrics for wandb handler
                        "reward": float(reward),
                        "total_reward": float(total_reward),
                        "similarity": float(info.get("similarity", 0.0)),
                        "episode": episode_idx,
                        "operation": int(action.operation),
                    }
                },
                "task_id": task_id,
                "task_pair_index": state.current_example_idx,
                "total_task_pairs": task.num_train_pairs,
            }
            
            # Log step through experiment logger
            experiment_logger.log_step(step_data)
            
            state = next_state
            if done:
                break
        
        # Log episode summary
        summary_data = {
            "episode_num": episode_idx,
            "total_steps": int(state.step_count),
            "total_reward": float(total_reward),
            "final_similarity": float(state.similarity_score),
            "task_id": task_id,
            "success": float(state.similarity_score) >= 0.99,
        }
        
        experiment_logger.log_episode_summary(summary_data)
    
    # Cleanup
    experiment_logger.close()
```

### Handler-Specific Features

#### SVG Handler
```python
# Automatically generates SVG visualizations for:
# - Individual steps showing before/after grids
# - Action selections and changes
# - Episode summaries with reward progression

# Access SVG handler directly if needed
if 'svg' in experiment_logger.handlers:
    svg_handler = experiment_logger.handlers['svg']
    svg_handler.start_run("custom_run_name")
    svg_handler.start_episode(episode_num)
```

#### Wandb Handler
```python
# Automatically logs to Weights & Biases:
# - Step metrics (reward, similarity, operation counts)
# - Episode summaries
# - System metrics
# - Grid visualizations as images

# Custom metrics can be added to info['metrics']
info['metrics'] = {
    "custom_metric": value,
    "policy_entropy": entropy,
    "value_loss": v_loss,
}
```

#### Console Handler
```python
# Provides rich console output with:
# - Progress bars for episodes
# - Colored step information
# - Performance metrics
# - Error reporting
```

### Configuration Examples

```python
# Research configuration with full logging
config = JaxArcConfig(
    logging=LoggingConfig(
        structured_logging=True,
        log_format="json",
        log_operations=True,
        log_grid_changes=True,
        log_rewards=True,
    ),
    wandb=WandbConfig(
        enabled=True,
        project_name="arc-rl-experiments",
        tags=("ppo", "arc-agi"),
        log_gradients=True,
    ),
    visualization=VisualizationConfig(
        enabled=True,
        level="full",
        output_formats=("svg", "png"),
    ),
    storage=StorageConfig(
        policy="research",
        base_output_dir="experiments",
    )
)

# Minimal configuration for fast training
config = JaxArcConfig(
    logging=LoggingConfig(
        log_format="text",
        log_frequency=100,  # Log less frequently
        log_operations=False,
    ),
    wandb=WandbConfig(enabled=False),
    visualization=VisualizationConfig(level="minimal"),
)
```

## Visualization and Debugging

### Console Visualization

```python
from jaxarc.utils.visualization import log_grid_to_console

# Log grids to console with Rich formatting
log_grid_to_console(state.working_grid, state.working_grid_mask, title="Current State")
log_grid_to_console(state.target_grid, state.target_grid_mask, title="Target")
```

### SVG Generation

```python
from jaxarc.utils.visualization import draw_grid_svg, draw_rl_step_svg

# Generate SVG for single grid
svg_content = draw_grid_svg(
    grid=state.working_grid,
    mask=state.working_grid_mask,
    title="Working Grid"
)

# Generate SVG for RL step visualization
step_svg = draw_rl_step_svg(
    before_grid=before_state.working_grid,
    after_grid=after_state.working_grid,
    action=action_dict,
    reward=reward,
    info=info,
    step_num=step_count
)
```

### JAX Debug Callbacks

```python
from jax import debug
from jaxarc.utils.visualization import log_grid_callback

# Use JAX debug callbacks for visualization during JIT compilation
def training_step(state, action):
    next_state, obs, reward, done, info = arc_step(state, action, config)
    
    # Log grid during JAX execution
    debug.callback(
        log_grid_callback,
        next_state.working_grid,
        next_state.working_grid_mask
    )
    
    return next_state, obs, reward, done, info

# JIT compile the training step
jit_training_step = jax.jit(training_step)
```

## JAX Integration Patterns

### Vectorization with vmap

```python
# Vectorize environment operations for batch processing
batch_reset = jax.vmap(arc_reset, in_axes=(0, None, 0))
batch_step = jax.vmap(arc_step, in_axes=(0, 0, None))

# Usage
batch_size = 64
keys = jax.random.split(key, batch_size)
tasks = [sample_task(k, training_tasks) for k in keys]

# Batch reset
batch_states, batch_obs = batch_reset(keys, config, tasks)

# Batch step
batch_actions = create_batch_actions(batch_size)  # Your action generation
batch_next_states, batch_next_obs, batch_rewards, batch_dones, batch_infos = batch_step(
    batch_states, batch_actions, config
)
```

### JIT Compilation

```python
# JIT compile environment functions for performance
jit_reset = jax.jit(arc_reset, static_argnums=(1,))  # config is static
jit_step = jax.jit(arc_step, static_argnums=(2,))    # config is static

# Usage
state, obs = jit_reset(key, config, task_data)
next_state, next_obs, reward, done, info = jit_step(state, action, config)
```

### Parallel Processing with pmap

```python
# Distribute across multiple devices
num_devices = jax.device_count()
batch_size_per_device = 16

# Reshape for pmap (devices, batch_per_device, ...)
keys = jax.random.split(key, num_devices * batch_size_per_device)
keys = keys.reshape(num_devices, batch_size_per_device, -1)

# Parallel map across devices
pmap_reset = jax.pmap(jax.vmap(arc_reset, in_axes=(0, None, 0)), in_axes=(0, None, 0))
pmap_step = jax.pmap(jax.vmap(arc_step, in_axes=(0, 0, None)), in_axes=(0, 0, None))
```

## Performance Optimization

### Memory Management

```python
# Use static shapes for optimal performance
config = JaxArcConfig(
    dataset=DatasetConfig(
        max_grid_height=30,  # Fixed size for static shapes
        max_grid_width=30,
        max_train_pairs=10,
        max_test_pairs=3
    )
)

# Pre-allocate arrays when possible
def create_empty_state_template(config: JaxArcConfig) -> ArcEnvState:
    """Create empty state template for efficient copying."""
    grid_shape = (config.dataset.max_grid_height, config.dataset.max_grid_width)
    empty_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
    empty_mask = jnp.zeros(grid_shape, dtype=jnp.bool_)
    
    return ArcEnvState(
        working_grid=empty_grid,
        working_grid_mask=empty_mask,
        # ... other fields
    )
```

### Efficient Action Handling

```python
# Pre-compile action handlers
from jaxarc.envs.actions import get_action_handler

action_handler = get_action_handler(config.action.selection_format)
jit_action_handler = jax.jit(action_handler)

# Use in training loop
selection_mask = jit_action_handler(action, state.working_grid_mask)
```

### Gradient Computation

```python
# Define loss function for RL algorithms
def compute_loss(params, state, action, target_reward):
    """Example loss function for policy gradient methods."""
    # Your neural network forward pass
    logits = policy_network(params, state)
    log_probs = jax.nn.log_softmax(logits)
    
    # Action log probability
    action_log_prob = log_probs[action]
    
    # Policy gradient loss
    loss = -action_log_prob * target_reward
    return loss

# Compute gradients
grad_fn = jax.grad(compute_loss)
gradients = grad_fn(params, state, action, reward)
```

## Common Patterns for RL Algorithms

### PPO Implementation Pattern

```python
import optax
from typing import NamedTuple

class PPOState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    key: chex.PRNGKey

def create_ppo_agent(config: JaxArcConfig, network_config: dict):
    """Create PPO agent for JaxARC."""
    
    # Initialize network parameters
    key = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros((config.dataset.max_grid_height, config.dataset.max_grid_width))
    params = init_network_params(key, dummy_obs, network_config)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=3e-4)
    opt_state = optimizer.init(params)
    
    return PPOState(params=params, opt_state=opt_state, key=key)

def ppo_update(ppo_state: PPOState, batch_data: dict, config: JaxArcConfig):
    """PPO update step."""
    
    def loss_fn(params):
        # Compute policy and value predictions
        policy_logits, values = network_forward(params, batch_data['observations'])
        
        # PPO loss computation
        policy_loss = compute_policy_loss(policy_logits, batch_data)
        value_loss = compute_value_loss(values, batch_data)
        
        return policy_loss + 0.5 * value_loss
    
    # Compute gradients and update
    loss, grads = jax.value_and_grad(loss_fn)(ppo_state.params)
    updates, new_opt_state = optimizer.update(grads, ppo_state.opt_state)
    new_params = optax.apply_updates(ppo_state.params, updates)
    
    return ppo_state._replace(params=new_params, opt_state=new_opt_state)

# Training loop
def train_ppo(config: JaxArcConfig, num_episodes: int):
    ppo_state = create_ppo_agent(config, network_config)
    
    for episode in range(num_episodes):
        # Collect rollout
        rollout_data = collect_rollout(ppo_state, config)
        
        # Update agent
        ppo_state = ppo_update(ppo_state, rollout_data, config)
```

### DQN Implementation Pattern

```python
class DQNState(NamedTuple):
    params: dict
    target_params: dict
    opt_state: optax.OptState
    replay_buffer: dict
    key: chex.PRNGKey

def create_dqn_agent(config: JaxArcConfig):
    """Create DQN agent for JaxARC."""
    key = jax.random.PRNGKey(42)
    
    # Initialize Q-network
    dummy_obs = jnp.zeros((config.dataset.max_grid_height, config.dataset.max_grid_width))
    params = init_q_network(key, dummy_obs)
    target_params = params  # Initialize target network
    
    # Initialize optimizer and replay buffer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    replay_buffer = create_replay_buffer(capacity=100000)
    
    return DQNState(
        params=params,
        target_params=target_params,
        opt_state=opt_state,
        replay_buffer=replay_buffer,
        key=key
    )

def dqn_update(dqn_state: DQNState, config: JaxArcConfig):
    """DQN update step."""
    
    # Sample batch from replay buffer
    batch = sample_replay_buffer(dqn_state.replay_buffer, batch_size=32)
    
    def loss_fn(params):
        # Q-values for current states
        q_values = q_network(params, batch['observations'])
        q_values_selected = q_values[jnp.arange(len(batch['actions'])), batch['actions']]
        
        # Target Q-values
        next_q_values = q_network(dqn_state.target_params, batch['next_observations'])
        targets = batch['rewards'] + 0.99 * jnp.max(next_q_values, axis=1) * (1 - batch['dones'])
        
        # MSE loss
        return jnp.mean((q_values_selected - targets) ** 2)
    
    # Update Q-network
    loss, grads = jax.value_and_grad(loss_fn)(dqn_state.params)
    updates, new_opt_state = optimizer.update(grads, dqn_state.opt_state)
    new_params = optax.apply_updates(dqn_state.params, updates)
    
    return dqn_state._replace(params=new_params, opt_state=new_opt_state)
```

### Action Space Handling with ActionSpaceController

```python
from jaxarc.envs import ActionSpaceController

def create_action_space_handler(config: JaxArcConfig):
    """Create action space handler with proper operation validation."""
    
    # Initialize action space controller for operation validation
    action_controller = ActionSpaceController()
    
    if config.action.selection_format == "point":
        def sample_action(key: chex.PRNGKey, state: ArcEnvState, logits: jnp.ndarray) -> StructuredAction:
            # Get allowed operations for current state
            allowed_mask = action_controller.get_allowed_operations(state, config.action)
            allowed_operations = jnp.where(allowed_mask)[0]
            
            # Sample from allowed operations only
            if len(allowed_operations) > 0:
                op_key, coord_key = jax.random.split(key)
                random_op_idx = jax.random.randint(op_key, (), 0, len(allowed_operations))
                operation = allowed_operations[random_op_idx]
            else:
                operation = jnp.array(0, dtype=jnp.int32)  # Fallback
            
            # Sample coordinates
            row = jax.random.randint(coord_key, (), 0, config.dataset.max_grid_height)
            col = jax.random.randint(coord_key, (), 0, config.dataset.max_grid_width)
            
            return create_point_action(operation, row, col)
            
    elif config.action.selection_format == "bbox":
        def sample_action(key: chex.PRNGKey, state: ArcEnvState, logits: dict) -> StructuredAction:
            # Get allowed operations
            allowed_mask = action_controller.get_allowed_operations(state, config.action)
            allowed_operations = jnp.where(allowed_mask)[0]
            
            op_key, coord_key = jax.random.split(key)
            
            # Sample operation from allowed set
            if len(allowed_operations) > 0:
                random_op_idx = jax.random.randint(op_key, (), 0, len(allowed_operations))
                operation = allowed_operations[random_op_idx]
            else:
                operation = jnp.array(0, dtype=jnp.int32)
            
            # Sample bbox coordinates
            k1, k2, k3, k4 = jax.random.split(coord_key, 4)
            r1 = jax.random.randint(k1, (), 0, config.dataset.max_grid_height)
            c1 = jax.random.randint(k2, (), 0, config.dataset.max_grid_width)
            r2 = jax.random.randint(k3, (), 0, config.dataset.max_grid_height)
            c2 = jax.random.randint(k4, (), 0, config.dataset.max_grid_width)
            
            # Ensure proper ordering
            min_r, max_r = jnp.minimum(r1, r2), jnp.maximum(r1, r2)
            min_c, max_c = jnp.minimum(c1, c2), jnp.maximum(c1, c2)
            
            return create_bbox_action(operation, min_r, min_c, max_r, max_c)
            
    elif config.action.selection_format == "mask":
        def sample_action(key: chex.PRNGKey, state: ArcEnvState, logits: dict) -> StructuredAction:
            # Get allowed operations
            allowed_mask = action_controller.get_allowed_operations(state, config.action)
            allowed_operations = jnp.where(allowed_mask)[0]
            
            op_key, mask_key = jax.random.split(key)
            
            # Sample operation from allowed set
            if len(allowed_operations) > 0:
                random_op_idx = jax.random.randint(op_key, (), 0, len(allowed_operations))
                operation = allowed_operations[random_op_idx]
            else:
                operation = jnp.array(0, dtype=jnp.int32)
            
            # Sample selection mask
            selection_probs = jax.nn.sigmoid(logits['selection'])
            selection = jax.random.bernoulli(mask_key, selection_probs)
            
            return create_mask_action(operation, selection)
    
    return sample_action

# Example usage in RL agent
class RandomAgent:
    def __init__(self, grid_shape, action_controller):
        self.grid_height, self.grid_width = grid_shape
        self.action_controller = action_controller
    
    def select_action(self, agent_state, env_state, config):
        """Select action with proper operation validation."""
        key = agent_state.key
        
        # Get allowed operations for current state
        allowed_mask = self.action_controller.get_allowed_operations(env_state, config.action)
        allowed_operations = jnp.where(allowed_mask)[0]
        
        # Sample from allowed operations
        op_key, coord_key = jax.random.split(key)
        if len(allowed_operations) > 0:
            random_op_idx = jax.random.randint(op_key, (), 0, len(allowed_operations))
            operation = allowed_operations[random_op_idx]
        else:
            operation = jnp.array(0, dtype=jnp.int32)
        
        # Generate bbox coordinates
        k1, k2, k3, k4 = jax.random.split(coord_key, 4)
        r1 = jax.random.randint(k1, (), 0, self.grid_height)
        c1 = jax.random.randint(k2, (), 0, self.grid_width)
        r2 = jax.random.randint(k3, (), 0, self.grid_height)
        c2 = jax.random.randint(k4, (), 0, self.grid_width)
        
        # Ensure proper ordering
        min_r, max_r = jnp.minimum(r1, r2), jnp.maximum(r1, r2)
        min_c, max_c = jnp.minimum(c1, c2), jnp.maximum(c1, c2)
        
        action = create_bbox_action(operation, min_r, min_c, max_r, max_c)
        new_agent_state = agent_state._replace(key=jax.random.split(key)[0])
        
        return action, new_agent_state
```

## Troubleshooting and Best Practices

### Common Issues and Solutions

#### 1. Shape Mismatches
```python
# Problem: Dynamic shapes cause JIT compilation issues
# Solution: Use static shapes defined in config
config = JaxArcConfig(
    dataset=DatasetConfig(
        max_grid_height=30,  # Always use these dimensions
        max_grid_width=30,
        max_train_pairs=10,
        max_test_pairs=3
    )
)
```

#### 2. Memory Issues with Large Batches
```python
# Problem: Out of memory with large batch sizes
# Solution: Use gradient accumulation or smaller batches
def accumulate_gradients(params, batch_data, num_accumulation_steps):
    """Accumulate gradients over multiple mini-batches."""
    total_grads = jax.tree_map(jnp.zeros_like, params)
    
    for i in range(num_accumulation_steps):
        mini_batch = get_mini_batch(batch_data, i, num_accumulation_steps)
        _, grads = jax.value_and_grad(loss_fn)(params, mini_batch)
        total_grads = jax.tree_map(lambda x, y: x + y, total_grads, grads)
    
    # Average gradients
    return jax.tree_map(lambda x: x / num_accumulation_steps, total_grads)
```

#### 3. Slow Training Due to Recompilation
```python
# Problem: JIT recompilation on every call
# Solution: Use static_argnums for config parameters
@functools.partial(jax.jit, static_argnums=(2,))  # config is static
def training_step(state, action, config):
    return arc_step(state, action, config)
```

#### 4. Action Validation Errors
```python
# Problem: Invalid actions causing errors
# Solution: Use action clipping and validation
def safe_action_execution(state, raw_action, config):
    """Safely execute action with validation and clipping."""
    try:
        # Validate and clip action
        clipped_action = clip_action_to_bounds(raw_action, state, config)
        
        # Execute action
        return arc_step(state, clipped_action, config)
    except Exception as e:
        # Fallback to no-op action
        noop_action = create_point_action(operation=34, row=0, col=0)  # Submit
        return arc_step(state, noop_action, config)
```

### Performance Best Practices

1. **Use Functional API**: Always prefer `arc_reset` and `arc_step` over the class-based API for RL training.

2. **Static Shapes**: Configure fixed grid dimensions and batch sizes for optimal JIT compilation.

3. **Batch Processing**: Use `jax.vmap` for batch operations and `jax.pmap` for multi-device training.

4. **Memory Management**: Pre-allocate arrays and reuse state templates when possible.

5. **Gradient Computation**: Use `jax.grad` and `jax.value_and_grad` for efficient gradient computation.

### Debugging Tips

1. **Use Debug Callbacks**: Leverage `jax.debug.callback` for visualization during training.

2. **Console Logging**: Use `log_grid_to_console` for quick state inspection.

3. **SVG Generation**: Generate SVG visualizations for detailed analysis.

4. **Action History**: Use the built-in action history system to track agent behavior.

5. **Configuration Validation**: Always validate your configuration before training.

### Integration with Popular RL Libraries

#### Complete RL Training Loop Example

Based on the `miniarc_rl_loop.py` notebook, here's a complete example:

```python
def run_complete_rl_training():
    """Complete RL training loop with logging and visualization."""
    
    # 1. Setup configuration with Hydra
    config_overrides = [
        "dataset=mini_arc",
        "action=raw", 
        "action.selection_format=bbox",
        "visualization=full",
        "logging=full",
        "storage=research",
        "wandb=research",
    ]
    
    hydra_config = get_config(overrides=config_overrides)
    config = JaxArcConfig.from_hydra(hydra_config)
    
    # 2. Setup experiment logger
    experiment_logger = ExperimentLogger(config)
    
    # 3. Load dataset
    parser = MiniArcParser(config.dataset)
    training_tasks = parser.get_available_task_ids()
    
    # 4. Initialize agent with action controller
    action_controller = ActionSpaceController()
    agent = RandomAgent(
        grid_shape=(config.dataset.max_grid_height, config.dataset.max_grid_width),
        action_controller=action_controller
    )
    
    # 5. Start run logging
    if 'svg' in experiment_logger.handlers:
        experiment_logger.handlers['svg'].start_run(f"training_run_{int(time.time())}")
    
    # 6. Training loop
    key = jax.random.PRNGKey(42)
    
    for episode_idx in range(num_episodes):
        # Select task
        key, task_key = jax.random.split(key)
        task_id_index = jax.random.randint(task_key, (), 0, len(training_tasks))
        task_id = training_tasks[int(task_id_index)]
        task = parser.get_task_by_id(task_id)
        
        # Start episode logging
        if 'svg' in experiment_logger.handlers:
            experiment_logger.handlers['svg'].start_episode(episode_idx)
        
        # Reset environment
        key, env_key, agent_key = jax.random.split(key, 3)
        state, observation = arc_reset(env_key, config, task_data=task)
        agent_state = agent.init_agent(agent_key)
        
        total_reward = 0.0
        
        # Episode loop
        for step_num in range(max_steps_per_episode):
            # Select action
            action, agent_state = agent.select_action(agent_state, state, config)
            
            # Store state before step
            state_before = state
            
            # Step environment
            state, observation, reward, done, info = arc_step(state, action, config)
            total_reward += reward
            
            # Prepare logging data
            step_data = {
                "step_num": step_num,
                "episode_num": episode_idx,
                "before_state": state_before,
                "after_state": state,
                "action": {
                    "operation": int(action.operation),
                    "r1": int(action.r1),
                    "c1": int(action.c1), 
                    "r2": int(action.r2),
                    "c2": int(action.c2),
                    "selection": action.to_selection_mask(state_before.working_grid.shape),
                },
                "reward": float(reward),
                "info": {
                    **info,
                    "metrics": {  # For wandb logging
                        "reward": float(reward),
                        "total_reward": float(total_reward),
                        "similarity": float(info.get("similarity", 0.0)),
                        "episode": episode_idx,
                        "operation": int(action.operation),
                    }
                },
                "task_id": task_id,
                "task_pair_index": state.current_example_idx,
                "total_task_pairs": task.num_train_pairs,
            }
            
            # Log step
            experiment_logger.log_step(step_data)
            
            if done:
                break
        
        # Log episode summary
        summary_data = {
            "episode_num": episode_idx,
            "total_steps": int(state.step_count),
            "total_reward": float(total_reward),
            "final_similarity": float(state.similarity_score),
            "task_id": task_id,
            "success": float(state.similarity_score) >= 0.99,
        }
        
        experiment_logger.log_episode_summary(summary_data)
    
    # Cleanup
    experiment_logger.close()

#### Batched Environment Processing

```python
from jaxarc.envs.functional import batch_reset, batch_step

class BatchedRandomAgent:
    """Agent for batched environments with proper operation validation."""
    
    def __init__(self, grid_shape, action_controller):
        self.grid_height, self.grid_width = grid_shape
        self.action_controller = action_controller
    
    def select_batch_actions(self, keys, states, config):
        """Select actions for batch of environments."""
        
        # Get allowed operations for each environment
        def get_allowed_ops_single(state):
            return self.action_controller.get_allowed_operations(state, config.action)
        
        get_allowed_ops_batch = jax.vmap(get_allowed_ops_single)
        allowed_masks = get_allowed_ops_batch(states)
        
        # Select actions for each environment
        def select_action_single(key, allowed_mask):
            op_key, coord_key = jax.random.split(key, 2)
            
            # Get allowed operations
            allowed_indices = jnp.where(allowed_mask, size=allowed_mask.shape[0], fill_value=-1)[0]
            num_allowed = jnp.sum(allowed_indices >= 0)
            
            # Select operation
            def select_from_allowed():
                random_idx = jax.random.randint(op_key, (), 0, num_allowed)
                return allowed_indices[random_idx]
            
            def fallback_operation():
                return jnp.array(0, dtype=jnp.int32)
            
            operation = jax.lax.cond(num_allowed > 0, select_from_allowed, fallback_operation)
            
            # Generate coordinates
            k1, k2, k3, k4 = jax.random.split(coord_key, 4)
            r1 = jax.random.randint(k1, (), 0, self.grid_height)
            c1 = jax.random.randint(k2, (), 0, self.grid_width)
            r2 = jax.random.randint(k3, (), 0, self.grid_height)
            c2 = jax.random.randint(k4, (), 0, self.grid_width)
            
            # Ensure proper ordering
            min_r, max_r = jnp.minimum(r1, r2), jnp.maximum(r1, r2)
            min_c, max_c = jnp.minimum(c1, c2), jnp.maximum(c1, c2)
            
            return operation, min_r, min_c, max_r, max_c
        
        # Vectorize over batch
        select_actions_batch = jax.vmap(select_action_single)
        operations, r1, c1, r2, c2 = select_actions_batch(keys, allowed_masks)
        
        return create_bbox_action(operation=operations, r1=r1, c1=c1, r2=r2, c2=c2)

def run_batched_training(batch_size=1000, num_steps=10):
    """Run batched environment training."""
    
    # Setup
    config = setup_batched_configuration()
    typed_config = JaxArcConfig.from_hydra(config)
    
    # Load tasks
    parser = MiniArcParser(typed_config.dataset)
    training_tasks = parser.get_available_task_ids()
    task = parser.get_task_by_id(training_tasks[0])
    
    # Initialize batched environments
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, batch_size)
    
    # Batch reset
    states, observations = batch_reset(keys, typed_config, task)
    
    # Initialize agent
    action_controller = ActionSpaceController()
    agent = BatchedRandomAgent(
        grid_shape=(typed_config.dataset.max_grid_height, typed_config.dataset.max_grid_width),
        action_controller=action_controller
    )
    
    # Training loop
    for step in range(num_steps):
        # Generate keys for this step
        step_key = jax.random.PRNGKey(42 + step)
        step_keys = jax.random.split(step_key, batch_size)
        
        # Select actions
        actions = agent.select_batch_actions(step_keys, states, typed_config)
        
        # Step environments
        states, observations, rewards, dones, infos = batch_step(states, actions, typed_config)
        
        # Log progress
        avg_reward = float(jnp.mean(rewards))
        num_done = int(jnp.sum(dones))
        print(f"Step {step + 1}: Avg Reward={avg_reward:.3f}, Done={num_done}/{batch_size}")
```

This comprehensive reference should provide everything needed to implement RL algorithms on top of JaxARC. The key is to use the functional API (`arc_reset`, `arc_step`) with proper JAX patterns for optimal performance.