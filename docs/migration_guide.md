# Migration Guide

This guide helps you migrate from older JaxARC patterns to the new unified
configuration system with Equinox, JAXTyping, and consolidated codebase
structure.

## Overview of Changes

The JaxARC codebase has been comprehensively refactored to:

- **Unify configuration systems** with single JaxArcConfig replacing dual config
  patterns
- **Eliminate code duplication** with centralized type definitions and
  consolidated examples
- **Modernize JAX patterns** using Equinox and JAXTyping throughout
- **Consolidate debug configurations** into a clear hierarchy
- **Streamline API consistency** with standardized patterns
- **Improve type safety** with comprehensive validation

## Breaking Changes Summary

### 1. Unified Configuration System (Major Change)

**Before (Dual Configuration Pattern):**

```python
from jaxarc.envs import ArcEnvironment
from jaxarc.envs.config import ArcEnvConfig

# Confusing dual configuration pattern
env_config = ArcEnvConfig(max_episode_steps=100)
hydra_config = OmegaConf.create({
    "debug": {"level": "standard"},
    "visualization": {"enabled": True}
})

# Which config contains what parameters?
env = ArcEnvironment(config=env_config, hydra_config=hydra_config)
```

**After (Single Unified Configuration):**

```python
from jaxarc.envs import ArcEnvironment
from jaxarc.envs.config import JaxArcConfig
from jaxarc.envs.factory import ConfigFactory

# Single configuration object containing all parameters
config = ConfigFactory.create_development_config(
    max_episode_steps=100,
    debug_level="standard",
    visualization_enabled=True
)

# Clear, single source of truth
env = ArcEnvironment(config)
```

### 2. Debug Configuration Consolidation

**Before (Multiple Conflicting Files):**

```yaml
# conf/debug/on.yaml - unclear what "on" means
enabled: true
log_steps: true

# conf/debug/full.yaml - overlaps with research.yaml
enabled: true
log_steps: true
log_grids: true
save_episodes: true
```

**After (Clear Hierarchy):**

```yaml
# conf/debug/standard.yaml - replaces on.yaml
level: "standard"
log_steps: true
log_grids: false
save_episodes: false

# conf/debug/research.yaml - replaces full.yaml
level: "research"
log_steps: true
log_grids: true
save_episodes: true
```

### 3. Configuration Parameter Grouping

**Before (Scattered Parameters):**

```python
# Parameters scattered across different configs
config = ArcEnvConfig(
    max_episode_steps=100,
    debug_enabled=True,  # Debug param in env config
    visualization_level="standard"  # Viz param in env config
)
hydra_config = {
    "debug": {"log_steps": True},  # More debug params in hydra
    "storage": {"policy": "standard"}  # Storage params elsewhere
}
```

**After (Logical Grouping):**

```python
config = JaxArcConfig(
    environment=EnvironmentConfig(max_episode_steps=100),
    debug=DebugConfig(level="standard", log_steps=True),
    visualization=VisualizationConfig(level="standard"),
    storage=StorageConfig(policy="standard")
)
```

### 4. Factory Function Standardization

**Before (Inconsistent Naming):**

```python
from jaxarc.envs.factory import (
    create_standard_config,  # create_*
    make_debug_config,       # make_*
    build_viz_config,        # build_*
    get_default_config       # get_*
)
```

**After (Consistent Patterns):**

```python
from jaxarc.envs.factory import ConfigFactory

# All factory methods follow consistent patterns
config = ConfigFactory.create_development_config()
config = ConfigFactory.create_research_config()
config = ConfigFactory.create_production_config()
config = ConfigFactory.from_hydra(hydra_cfg)
config = ConfigFactory.from_preset("standard", overrides={})
```

## Step-by-Step Migration

### Step 1: Replace Dual Configuration Pattern

**Old pattern (ArcEnvironment with dual configs):**

```python
from jaxarc.envs import ArcEnvironment
from jaxarc.envs.config import ArcEnvConfig
from omegaconf import OmegaConf

# Confusing dual configuration
env_config = ArcEnvConfig(max_episode_steps=100)
hydra_config = OmegaConf.create({
    "debug": {"level": "standard"},
    "visualization": {"enabled": True}
})

env = ArcEnvironment(config=env_config, hydra_config=hydra_config)
```

**New pattern (Single unified config):**

```python
from jaxarc.envs import ArcEnvironment
from jaxarc.envs.factory import ConfigFactory

# Single configuration object
config = ConfigFactory.create_development_config(
    max_episode_steps=100,
    debug_level="standard",
    visualization_enabled=True
)

env = ArcEnvironment(config)
```

### Step 2: Update Configuration Creation

**Old factory functions (multiple inconsistent patterns):**

```python
from jaxarc.envs.factory import (
    create_standard_config,
    make_debug_config,
    build_visualization_config
)

# Scattered configuration creation
env_config = create_standard_config(max_episode_steps=100)
debug_config = make_debug_config(level="standard")
viz_config = build_visualization_config(enabled=True)
```

**New unified factory (consistent patterns):**

```python
from jaxarc.envs.factory import ConfigFactory

# Method 1: Use preset with overrides
config = ConfigFactory.from_preset("development", {
    "environment.max_episode_steps": 100,
    "debug.level": "standard",
    "visualization.enabled": True
})

# Method 2: Use specific factory method
config = ConfigFactory.create_development_config(
    max_episode_steps=100,
    debug_level="standard",
    visualization_enabled=True
)

# Method 3: Convert from Hydra (for existing Hydra users)
config = ConfigFactory.from_hydra(hydra_cfg)
```

### Step 3: Update Debug Configuration References

**Old debug config files (remove these):**

```yaml
# conf/debug/on.yaml - DELETE THIS FILE
enabled: true
log_steps: true

# conf/debug/full.yaml - DELETE THIS FILE
enabled: true
log_steps: true
log_grids: true
```

**New debug config usage:**

```python
# In your code, replace references to old debug configs
# Old:
# defaults:
#   - debug: on

# New:
# defaults:
#   - debug: standard

# Or in Python:
config = ConfigFactory.create_development_config(debug_level="standard")
```

### Step 4: Update Configuration Access Patterns

**Old scattered parameter access:**

```python
# Parameters scattered across different objects
max_steps = env_config.max_episode_steps
debug_enabled = hydra_config.debug.enabled
viz_level = hydra_config.visualization.level
storage_policy = hydra_config.storage.policy
```

**New grouped parameter access:**

```python
# All parameters in logical groups within single config
max_steps = config.environment.max_episode_steps
debug_enabled = config.debug.level != "off"
viz_level = config.visualization.level
storage_policy = config.storage.policy
```

### Step 5: Update Examples and Scripts

**Old example patterns:**

```python
# examples/old_config_demo.py
def main():
    env_config = create_standard_config()
    hydra_config = OmegaConf.load("conf/config.yaml")
    env = ArcEnvironment(env_config, hydra_config)
```

**New example patterns:**

```python
# examples/unified_config_demo.py
def main():
    # Method 1: Factory function
    config = ConfigFactory.create_development_config()
    env = ArcEnvironment(config)

    # Method 2: From YAML
    config = ConfigFactory.from_yaml("conf/presets/development.yaml")
    env = ArcEnvironment(config)

    # Method 3: With Hydra
    @hydra.main(config_path="conf", config_name="config")
    def hydra_main(cfg):
        config = ConfigFactory.from_hydra(cfg)
        env = ArcEnvironment(config)
```

### Step 6: Migrate Custom Configurations

**Old custom config creation:**

```python
# Custom configuration with scattered parameters
custom_env_config = ArcEnvConfig(
    max_episode_steps=200,
    reward_config=RewardConfig(success_bonus=20.0)
)
custom_hydra_config = OmegaConf.create({
    "debug": {"level": "research", "save_episodes": True},
    "visualization": {"level": "full", "show_coordinates": True}
})
```

**New custom config creation:**

```python
# Method 1: Start with preset and override
config = ConfigFactory.from_preset("research", {
    "environment.max_episode_steps": 200,
    "reward.success_bonus": 20.0,
    "visualization.show_coordinates": True
})

# Method 2: Build from components
config = JaxArcConfig(
    environment=EnvironmentConfig(max_episode_steps=200),
    reward=RewardConfig(success_bonus=20.0),
    debug=DebugConfig(level="research", save_episodes=True),
    visualization=VisualizationConfig(level="full", show_coordinates=True),
    storage=StorageConfig(policy="research"),
    logging=LoggingConfig(level="DEBUG"),
    wandb=WandbConfig(enabled=False)
)
```

## New Features and Capabilities

### 1. Unified Configuration System

```python
from jaxarc.envs.config import JaxArcConfig
from jaxarc.envs.factory import ConfigFactory

# Single configuration object with all parameters logically grouped
config = JaxArcConfig(
    environment=EnvironmentConfig(max_episode_steps=100),
    debug=DebugConfig(level="standard", log_steps=True),
    visualization=VisualizationConfig(enabled=True, level="standard"),
    storage=StorageConfig(policy="standard", max_size_gb=5.0),
    logging=LoggingConfig(level="INFO", console_logging=True),
    wandb=WandbConfig(enabled=False)
)

# Comprehensive validation with clear error messages
validation_result = config.validate()
if validation_result.errors:
    for error in validation_result.errors:
        print(f"Configuration error: {error}")
```

### 2. Configuration Factory System

```python
from jaxarc.envs.factory import ConfigFactory

# Preset-based configuration for common use cases
dev_config = ConfigFactory.create_development_config()
research_config = ConfigFactory.create_research_config()
production_config = ConfigFactory.create_production_config()

# Flexible preset system with overrides
config = ConfigFactory.from_preset("development", {
    "environment.max_episode_steps": 200,
    "debug.level": "verbose",
    "visualization.show_coordinates": True
})

# Seamless Hydra integration
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    config = ConfigFactory.from_hydra(cfg)
    env = ArcEnvironment(config)
```

### 3. Configuration Parameter Grouping

```python
# Clear parameter organization with type safety
config = JaxArcConfig(
    environment=EnvironmentConfig(
        max_episode_steps=100,
        grid_size=(30, 30),
        reward_on_submit_only=False
    ),
    debug=DebugConfig(
        level="standard",  # off, minimal, standard, verbose, research
        log_steps=True,
        log_grids=False,
        save_episodes=False
    ),
    visualization=VisualizationConfig(
        enabled=True,
        level="standard",
        show_grids=True,
        show_actions=True,
        color_scheme="default"
    ),
    storage=StorageConfig(
        policy="standard",  # none, minimal, standard, research
        max_size_gb=5.0,
        cleanup_on_exit=True,
        compression=True
    )
)

# Access parameters through logical grouping
max_steps = config.environment.max_episode_steps
debug_level = config.debug.level
viz_enabled = config.visualization.enabled
storage_policy = config.storage.policy
```

### 4. YAML-Python Bidirectional Conversion

```python
# Export configuration to YAML
yaml_content = config.to_yaml()
with open("my_config.yaml", "w") as f:
    f.write(yaml_content)

# Load configuration from YAML
config = JaxArcConfig.from_yaml("my_config.yaml")

# Perfect 1:1 mapping between YAML and Python
# YAML structure matches Python object structure exactly
```

### 5. Migration and Backward Compatibility

```python
from jaxarc.envs.migration import ConfigMigrator

# Automatic migration from legacy configurations
migrator = ConfigMigrator()

# Migrate old dual config pattern
legacy_env_config = ArcEnvConfig(max_episode_steps=100)
legacy_hydra_config = {"debug": {"enabled": True}}

new_config = migrator.migrate_dual_config(legacy_env_config, legacy_hydra_config)

# Migration report with warnings and suggestions
report = migrator.create_migration_report(legacy_config)
for warning in report.warnings:
    print(f"Warning: {warning}")
for suggestion in report.suggestions:
    print(f"Suggestion: {suggestion}")
```

### 6. Enhanced Validation and Error Handling

```python
from jaxarc.envs.config import ConfigValidationError

try:
    config = JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=-10),  # Invalid
        debug=DebugConfig(level="invalid_level"),  # Invalid
        storage=StorageConfig(max_size_gb=-1.0)  # Invalid
    )
except ConfigValidationError as e:
    print(f"Configuration validation failed:")
    for field, error in e.field_errors.items():
        print(f"  {field}: {error}")

    # Specific error messages with suggestions
    # Output:
    # environment.max_episode_steps: Must be positive, got -10
    # debug.level: Must be one of ['off', 'minimal', 'standard', 'verbose', 'research'], got 'invalid_level'
    # storage.max_size_gb: Must be non-negative, got -1.0
```

## Performance Considerations

### JAX Transformations

**Equinox modules work seamlessly with JAX:**

```python
# JIT compilation
@jax.jit
def fast_step(state: ArcEnvState, action: dict) -> ArcEnvState:
    return arc_step(state, action, config)

# Vectorization
batch_step = jax.vmap(fast_step, in_axes=(0, 0))

# Parallel processing
parallel_step = jax.pmap(fast_step, in_axes=(0, 0))
```

### Memory Usage

**Equinox provides better memory efficiency:**

```python
from jaxarc.utils.equinox_utils import module_memory_usage

# Analyze memory usage
memory_info = module_memory_usage(state)
print(f"Total memory: {memory_info['total_bytes']} bytes")
print(f"Total elements: {memory_info['total_elements']}")
```

## Common Migration Issues

### Issue 1: Dual Configuration Pattern Errors

**Problem:**

```python
TypeError: ArcEnvironment.__init__() got an unexpected keyword argument 'hydra_config'
```

**Solution:**

```python
# Old dual pattern (no longer supported):
# env = ArcEnvironment(config=env_config, hydra_config=hydra_config)

# New unified pattern:
from jaxarc.envs.factory import ConfigFactory

config = ConfigFactory.from_hydra(hydra_config)  # Convert hydra to unified config
env = ArcEnvironment(config)  # Single config parameter
```

### Issue 2: Debug Configuration File Not Found

**Problem:**

```python
FileNotFoundError: [Errno 2] No such file or directory: 'conf/debug/on.yaml'
```

**Solution:**

```python
# Old debug config files have been removed:
# - conf/debug/on.yaml -> use conf/debug/standard.yaml
# - conf/debug/full.yaml -> use conf/debug/research.yaml

# Update your config defaults:
# Old:
# defaults:
#   - debug: on

# New:
# defaults:
#   - debug: standard
```

### Issue 3: Configuration Parameter Not Found

**Problem:**

```python
AttributeError: 'JaxArcConfig' object has no attribute 'max_episode_steps'
```

**Solution:**

```python
# Parameters are now grouped logically
# Old direct access:
# max_steps = config.max_episode_steps

# New grouped access:
max_steps = config.environment.max_episode_steps
debug_level = config.debug.level
viz_enabled = config.visualization.enabled
```

### Issue 4: Factory Function Import Errors

**Problem:**

```python
ImportError: cannot import name 'create_standard_config' from 'jaxarc.envs.factory'
```

**Solution:**

```python
# Old scattered factory functions have been consolidated
# Old imports:
# from jaxarc.envs.factory import create_standard_config, make_debug_config

# New unified factory:
from jaxarc.envs.factory import ConfigFactory

# Use consistent factory methods:
config = ConfigFactory.create_development_config()  # Replaces create_standard_config
config = ConfigFactory.create_research_config()     # Replaces create_full_config
```

### Issue 5: Configuration Validation Errors

**Problem:**

```python
ConfigValidationError: debug.level must be one of ['off', 'minimal', 'standard', 'verbose', 'research'], got 'on'
```

**Solution:**

```python
# Old debug levels have been renamed for clarity:
# 'on' -> 'standard'
# 'full' -> 'research'

# Update your configuration:
config = JaxArcConfig(
    debug=DebugConfig(level="standard"),  # Not "on"
    # ... other config
)
```

### Issue 6: Missing Configuration Groups

**Problem:**

```python
AttributeError: 'JaxArcConfig' object has no attribute 'storage'
```

**Solution:**

```python
# All configuration groups must be explicitly provided
# The unified config requires all groups to be specified

config = JaxArcConfig(
    environment=EnvironmentConfig(),
    debug=DebugConfig(),
    visualization=VisualizationConfig(),
    storage=StorageConfig(),        # Don't forget this
    logging=LoggingConfig(),        # And this
    wandb=WandbConfig()            # And this
)

# Or use factory functions that provide all groups:
config = ConfigFactory.create_development_config()
```

## Testing Your Migration

### 1. Validate Configuration Structure

```python
from jaxarc.envs.config import JaxArcConfig
from jaxarc.envs.factory import ConfigFactory

def test_config_migration():
    # Test unified configuration creation
    config = ConfigFactory.create_development_config()
    assert isinstance(config, JaxArcConfig)

    # Test configuration validation
    validation_result = config.validate()
    assert len(validation_result.errors) == 0, f"Config validation failed: {validation_result.errors}"

    # Test parameter access through groups
    assert hasattr(config, 'environment')
    assert hasattr(config, 'debug')
    assert hasattr(config, 'visualization')
    assert hasattr(config, 'storage')
    assert hasattr(config, 'logging')
    assert hasattr(config, 'wandb')

    # Test parameter values
    assert config.environment.max_episode_steps > 0
    assert config.debug.level in ["off", "minimal", "standard", "verbose", "research"]
```

### 2. Test Environment Integration

```python
from jaxarc.envs import ArcEnvironment

def test_environment_migration():
    # Test single configuration pattern
    config = ConfigFactory.create_development_config()
    env = ArcEnvironment(config)  # Should work without hydra_config parameter

    # Test configuration access within environment
    assert env.config.environment.max_episode_steps == config.environment.max_episode_steps
    assert env.config.debug.level == config.debug.level

    # Test environment functionality
    import jax
    key = jax.random.PRNGKey(42)
    state = env.reset(key)
    assert state is not None
```

### 3. Test YAML-Python Conversion

```python
import tempfile
import os

def test_yaml_conversion():
    # Create configuration
    original_config = ConfigFactory.create_development_config(
        max_episode_steps=150,
        debug_level="verbose"
    )

    # Export to YAML
    yaml_content = original_config.to_yaml()

    # Save and reload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        # Load from YAML
        loaded_config = JaxArcConfig.from_yaml(temp_path)

        # Verify equivalence
        assert loaded_config.environment.max_episode_steps == 150
        assert loaded_config.debug.level == "verbose"

    finally:
        os.unlink(temp_path)
```

### 4. Test Migration Tools

```python
from jaxarc.envs.migration import ConfigMigrator

def test_migration_tools():
    # Test legacy config migration
    migrator = ConfigMigrator()

    # Simulate old dual config pattern
    legacy_env_config = {"max_episode_steps": 100}
    legacy_hydra_config = {
        "debug": {"enabled": True, "level": "on"},
        "visualization": {"enabled": True}
    }

    # Test migration
    migrated_config = migrator.migrate_dual_config(legacy_env_config, legacy_hydra_config)
    assert isinstance(migrated_config, JaxArcConfig)
    assert migrated_config.environment.max_episode_steps == 100
    assert migrated_config.debug.level == "standard"  # "on" -> "standard"
```

## Enhanced ARC Step Logic Migration

The ARC environment has been significantly enhanced with multi-demonstration training, test pair evaluation, action history tracking, and dynamic action space control. This section covers migrating to these new capabilities.

### Overview of Enhanced Features

The enhanced environment introduces:

- **Multi-demonstration training** on all available demonstration pairs
- **Test pair evaluation mode** without target grid access
- **Comprehensive action history tracking** with configurable storage
- **Dynamic action space control** with context-aware filtering
- **Enhanced observation space** separate from internal state
- **Episode management system** with flexible termination criteria

### Breaking Changes in Enhanced Environment

#### 1. Enhanced State Structure

**Before (Basic State):**

```python
# Old ArcEnvState with basic fields
class ArcEnvState(eqx.Module):
    task_data: JaxArcTask
    working_grid: GridArray
    target_grid: GridArray
    step_count: StepCount
    episode_done: EpisodeDone
    current_example_idx: EpisodeIndex
    selected: SelectionArray
    clipboard: GridArray
    similarity_score: SimilarityScore
```

**After (Enhanced State):**

```python
# Enhanced ArcEnvState with multi-demonstration support
class ArcEnvState(eqx.Module):
    # Original fields (unchanged)
    task_data: JaxArcTask
    working_grid: GridArray
    target_grid: GridArray
    step_count: StepCount
    episode_done: EpisodeDone
    current_example_idx: EpisodeIndex
    selected: SelectionArray
    clipboard: GridArray
    similarity_score: SimilarityScore
    
    # New enhanced fields
    episode_mode: EpisodeMode  # 0=train, 1=test
    available_demo_pairs: AvailablePairs
    available_test_pairs: AvailablePairs
    demo_completion_status: CompletionStatus
    test_completion_status: CompletionStatus
    action_history: ActionHistory
    action_history_length: HistoryLength
    allowed_operations_mask: OperationMask
```

#### 2. Enhanced Reset Function

**Before (Basic Reset):**

```python
from jaxarc.envs import arc_reset

# Simple reset with task data
state = arc_reset(key, config, task_data=task)
```

**After (Enhanced Reset):**

```python
from jaxarc.envs import arc_reset

# Enhanced reset with episode mode and pair selection
state, observation = arc_reset(
    key, 
    config, 
    task_data=task,
    episode_mode="train",  # "train" or "test"
    initial_pair_idx=None  # Let episode manager select
)

# Note: Now returns (state, observation) tuple
```

#### 3. Enhanced Step Function

**Before (Basic Step):**

```python
from jaxarc.envs import arc_step

# Basic step function
state, reward, done, info = arc_step(state, action, config)
```

**After (Enhanced Step):**

```python
from jaxarc.envs import arc_step

# Enhanced step with separate observation
state, observation, reward, done, info = arc_step(state, action, config)

# Note: Now returns (state, observation, reward, done, info)
```

#### 4. Expanded Action Space

**Before (35 Operations):**

```python
# Original ARCLE operations (0-34)
action = {
    "selection": selection_mask,
    "operation": 0  # FILL operation (0-34 range)
}
```

**After (42 Operations):**

```python
# Enhanced operations including control operations (0-41)
action = {
    "selection": selection_mask,
    "operation": 35  # SWITCH_TO_NEXT_DEMO_PAIR (new control operations 35-41)
}

# New control operations:
# 35: SWITCH_TO_NEXT_DEMO_PAIR
# 36: SWITCH_TO_PREV_DEMO_PAIR
# 37: SWITCH_TO_NEXT_TEST_PAIR
# 38: SWITCH_TO_PREV_TEST_PAIR
# 39: RESET_CURRENT_PAIR
# 40: SWITCH_TO_FIRST_UNSOLVED_DEMO
# 41: SWITCH_TO_FIRST_UNSOLVED_TEST
```

### Step-by-Step Enhanced Migration

#### Step 1: Update Configuration for Enhanced Features

**Old basic configuration:**

```python
from jaxarc.envs.factory import ConfigFactory

# Basic configuration without enhanced features
config = ConfigFactory.create_development_config(
    max_episode_steps=100,
    debug_level="standard"
)
```

**New enhanced configuration:**

```python
from jaxarc.envs.factory import ConfigFactory

# Enhanced configuration with new feature groups
config = ConfigFactory.create_research_config(
    # Basic environment settings
    max_episode_steps=100,
    debug_level="standard",
    
    # Episode management settings
    episode_mode="train",
    demo_selection_strategy="random",
    allow_demo_switching=True,
    max_pairs_per_episode=4,
    
    # Action history settings
    history_enabled=True,
    max_history_length=1000,
    store_selection_data=True,
    
    # Action space control
    dynamic_action_filtering=True,
    context_dependent_operations=True,
    
    # Enhanced observation
    include_completion_status=True,
    include_action_space_info=True,
    observation_format="standard"
)
```

#### Step 2: Update Reset Function Calls

**Old reset pattern:**

```python
# Old reset returns only state
state = arc_reset(key, config, task_data=task)

# Access target grid directly from state
target = state.target_grid
```

**New reset pattern:**

```python
# New reset returns (state, observation) tuple
state, observation = arc_reset(
    key, 
    config, 
    task_data=task,
    episode_mode="train"  # Specify training or evaluation mode
)

# Access target grid from observation (may be None in test mode)
target = observation.target_grid  # None in test mode for proper evaluation
```

#### Step 3: Update Step Function Calls

**Old step pattern:**

```python
# Old step returns (state, reward, done, info)
state, reward, done, info = arc_step(state, action, config)

# Direct state access
current_pair = state.current_example_idx
```

**New step pattern:**

```python
# New step returns (state, observation, reward, done, info)
state, observation, reward, done, info = arc_step(state, action, config)

# Access information through observation for agent view
current_pair = observation.current_pair_idx
episode_mode = observation.episode_mode
allowed_ops = observation.allowed_operations_mask
```

#### Step 4: Handle New Action Operations

**Old action handling:**

```python
# Limited to original 35 operations
if action["operation"] >= 35:
    raise ValueError(f"Invalid operation: {action['operation']}")

# Simple action execution
state, reward, done, info = arc_step(state, action, config)
```

**New action handling:**

```python
# Support for 42 operations including control operations
if action["operation"] >= 42:
    raise ValueError(f"Invalid operation: {action['operation']}")

# Handle control operations (35-41)
if action["operation"] >= 35:
    # Control operations may switch pairs or reset state
    print(f"Executing control operation: {action['operation']}")

state, observation, reward, done, info = arc_step(state, action, config)

# Check if pair was switched
if action["operation"] in [35, 36, 37, 38, 40, 41]:
    print(f"Switched to pair: {observation.current_pair_idx}")
```

#### Step 5: Update Multi-Demonstration Training

**Old single-demonstration training:**

```python
# Old pattern: only used first demonstration pair
state = arc_reset(key, config, task_data=task)

# Training loop on single pair
for step in range(100):
    action = generate_action()  # Your action generation
    state, reward, done, info = arc_step(state, action, config)
    if done:
        break
```

**New multi-demonstration training:**

```python
# New pattern: can train on all demonstration pairs
state, observation = arc_reset(
    key, 
    config, 
    task_data=task, 
    episode_mode="train"
)

# Training loop with pair switching
for step in range(200):
    # Regular grid operations
    if step % 50 != 0:
        action = generate_grid_action()  # Your grid action generation
    else:
        # Periodically switch to next demonstration pair
        action = {
            "selection": jnp.zeros((30, 30), dtype=bool),
            "operation": 35  # SWITCH_TO_NEXT_DEMO_PAIR
        }
    
    state, observation, reward, done, info = arc_step(state, action, config)
    
    # Monitor progress across all demonstration pairs
    print(f"Demo completion: {observation.demo_completion_status}")
    
    if done:
        break
```

#### Step 6: Add Test Pair Evaluation

**Old evaluation (no proper test mode):**

```python
# Old pattern: evaluation was same as training
state = arc_reset(key, config, task_data=task)
# Agent could see target grids during evaluation (incorrect)
```

**New proper test evaluation:**

```python
# New pattern: proper evaluation mode without target access
state, observation = arc_reset(
    key, 
    config, 
    task_data=task, 
    episode_mode="test"  # Switch to evaluation mode
)

# Evaluation loop without target access
for step in range(100):
    action = generate_action()  # Your action generation
    state, observation, reward, done, info = arc_step(state, action, config)
    
    # Verify no target access during evaluation
    assert observation.target_grid is None, "Target should not be available in test mode"
    
    # Rewards only on submission in test mode
    if action["operation"] == 34:  # SUBMIT
        print(f"Submission reward: {reward}")
    else:
        assert reward == 0.0, "Should only get rewards on submission in test mode"
    
    if done:
        break
```

#### Step 7: Add Action History Analysis

**Old pattern (no action history):**

```python
# No action history tracking available
state, reward, done, info = arc_step(state, action, config)
# Could not analyze agent behavior over time
```

**New pattern (with action history):**

```python
# Enable action history in configuration
config = ConfigFactory.create_research_config(
    history_enabled=True,
    max_history_length=1000,
    store_selection_data=True
)

# Action history is automatically tracked
state, observation, reward, done, info = arc_step(state, action, config)

# Access action history for analysis
print(f"Action history length: {state.action_history_length}")

# Analyze recent actions in observation
if observation.recent_actions is not None:
    print(f"Recent actions available: {len(observation.recent_actions)}")

# Full history analysis (see action_history_analysis_example.py)
from jaxarc.envs.action_history import ActionHistoryTracker
history_tracker = ActionHistoryTracker()
action_sequence = history_tracker.get_action_sequence(state, 0, 10)
```

### Enhanced Configuration Options

#### Episode Management Configuration

```python
from jaxarc.envs.config import ArcEpisodeConfig

episode_config = ArcEpisodeConfig(
    episode_mode="train",  # "train" or "test"
    demo_selection_strategy="random",  # "sequential" or "random"
    allow_demo_switching=True,
    require_all_demos_solved=False,
    terminate_on_first_success=False,
    max_pairs_per_episode=4,
    training_reward_frequency="step",  # "step" or "submit"
    evaluation_reward_frequency="submit"  # Only "submit" for proper evaluation
)
```

#### Action History Configuration

```python
from jaxarc.envs.config import HistoryConfig

# Memory-efficient configuration
minimal_history = HistoryConfig(
    enabled=True,
    max_history_length=100,
    store_selection_data=False,  # Save memory
    store_intermediate_grids=False,
    compress_repeated_actions=True
)

# Research configuration with full tracking
research_history = HistoryConfig(
    enabled=True,
    max_history_length=2000,
    store_selection_data=True,
    store_intermediate_grids=True,  # Memory intensive
    compress_repeated_actions=False
)
```

#### Action Space Control Configuration

```python
from jaxarc.envs.config import ActionConfig

# Restricted action space for focused training
restricted_config = ActionConfig(
    allowed_operations=[0, 1, 2, 3, 4],  # Only basic operations
    max_operations=5,
    dynamic_action_filtering=True,
    context_dependent_operations=True,
    validate_actions=True,
    allow_invalid_actions=False
)

# Full action space with dynamic control
full_config = ActionConfig(
    allowed_operations=None,  # All 42 operations
    max_operations=42,
    dynamic_action_filtering=True,
    context_dependent_operations=True,
    validate_actions=True,
    allow_invalid_actions=True
)
```

#### Enhanced Observation Configuration

```python
from jaxarc.envs.config import ObservationConfig

# Minimal observation for production
minimal_obs = ObservationConfig(
    include_target_grid=True,
    include_completion_status=False,
    include_action_space_info=False,
    include_recent_actions=False,
    observation_format="minimal"
)

# Rich observation for research
research_obs = ObservationConfig(
    include_target_grid=True,
    include_completion_status=True,
    include_action_space_info=True,
    include_recent_actions=True,
    recent_action_count=10,
    observation_format="rich"
)
```

### Common Enhanced Migration Issues

#### Issue 1: Reset Function Return Value Changed

**Problem:**

```python
# Old code expecting single return value
state = arc_reset(key, config, task_data=task)
# TypeError: cannot unpack non-sequence ArcEnvState
```

**Solution:**

```python
# Update to handle tuple return
state, observation = arc_reset(key, config, task_data=task, episode_mode="train")

# Or if you only need state (not recommended)
state, _ = arc_reset(key, config, task_data=task, episode_mode="train")
```

#### Issue 2: Step Function Return Value Changed

**Problem:**

```python
# Old code expecting 4 return values
state, reward, done, info = arc_step(state, action, config)
# ValueError: too many values to unpack (expected 4)
```

**Solution:**

```python
# Update to handle 5 return values
state, observation, reward, done, info = arc_step(state, action, config)

# Or if you don't need observation
state, _, reward, done, info = arc_step(state, action, config)
```

#### Issue 3: Invalid Operation Range

**Problem:**

```python
# Old code assuming 35 operations max
action = {"selection": mask, "operation": 40}
# Error: operation 40 not recognized
```

**Solution:**

```python
# Update operation range validation
if action["operation"] >= 42:  # Changed from 35 to 42
    raise ValueError(f"Invalid operation: {action['operation']}")

# Handle new control operations
if action["operation"] >= 35:
    print("Executing control operation")
```

#### Issue 4: Missing Enhanced Configuration Groups

**Problem:**

```python
# Old config missing new groups
config = JaxArcConfig(
    environment=EnvironmentConfig(),
    debug=DebugConfig()
    # Missing new configuration groups
)
# AttributeError: 'JaxArcConfig' object has no attribute 'episode'
```

**Solution:**

```python
# Include all enhanced configuration groups
config = JaxArcConfig(
    environment=EnvironmentConfig(),
    debug=DebugConfig(),
    visualization=VisualizationConfig(),
    storage=StorageConfig(),
    logging=LoggingConfig(),
    wandb=WandbConfig(),
    # New enhanced groups
    episode=ArcEpisodeConfig(),
    history=HistoryConfig(),
    action=ActionConfig(),
    observation=ObservationConfig()
)

# Or use factory functions that include all groups
config = ConfigFactory.create_research_config()
```

#### Issue 5: Target Grid Access in Test Mode

**Problem:**

```python
# Trying to access target in test mode
state, observation = arc_reset(key, config, episode_mode="test")
target = observation.target_grid  # This will be None
# AttributeError or unexpected None value
```

**Solution:**

```python
# Check episode mode before accessing target
if observation.target_grid is not None:
    # Training mode - target available
    target = observation.target_grid
    print("Training mode: target available")
else:
    # Test mode - no target access (correct behavior)
    print("Test mode: target hidden for proper evaluation")
    # Use alternative evaluation metrics
```

### Testing Enhanced Migration

#### Test Enhanced State Structure

```python
def test_enhanced_state():
    config = ConfigFactory.create_research_config()
    key = jax.random.PRNGKey(42)
    
    # Test enhanced reset
    state, observation = arc_reset(key, config, episode_mode="train")
    
    # Verify enhanced state fields
    assert hasattr(state, 'episode_mode')
    assert hasattr(state, 'available_demo_pairs')
    assert hasattr(state, 'action_history')
    assert hasattr(state, 'allowed_operations_mask')
    
    # Verify observation structure
    assert hasattr(observation, 'working_grid')
    assert hasattr(observation, 'episode_mode')
    assert hasattr(observation, 'current_pair_idx')
    assert hasattr(observation, 'allowed_operations_mask')
```

#### Test Multi-Demonstration Training

```python
def test_multi_demo_training():
    config = ConfigFactory.create_research_config(
        episode_mode="train",
        allow_demo_switching=True
    )
    
    key = jax.random.PRNGKey(42)
    state, observation = arc_reset(key, config, episode_mode="train")
    
    # Test pair switching
    switch_action = {
        "selection": jnp.zeros((30, 30), dtype=bool),
        "operation": 35  # SWITCH_TO_NEXT_DEMO_PAIR
    }
    
    old_pair = observation.current_pair_idx
    state, observation, reward, done, info = arc_step(state, switch_action, config)
    
    # Verify pair switching worked
    if state.available_demo_pairs.sum() > 1:
        assert observation.current_pair_idx != old_pair
```

#### Test Action History Tracking

```python
def test_action_history():
    config = ConfigFactory.create_research_config(
        history_enabled=True,
        max_history_length=100
    )
    
    key = jax.random.PRNGKey(42)
    state, observation = arc_reset(key, config, episode_mode="train")
    
    # Execute some actions
    for i in range(5):
        action = {
            "selection": jnp.zeros((30, 30), dtype=bool).at[i:i+2, i:i+2].set(True),
            "operation": i % 10
        }
        state, observation, reward, done, info = arc_step(state, action, config)
    
    # Verify history tracking
    assert state.action_history_length == 5
    
    # Test history retrieval
    from jaxarc.envs.action_history import ActionHistoryTracker
    tracker = ActionHistoryTracker()
    history = tracker.get_action_sequence(state, 0, 5)
    assert len(history) == 5
```

### Performance Considerations for Enhanced Features

#### Memory Usage

The enhanced features add memory overhead:

```python
# Memory-efficient configuration for production
efficient_config = ConfigFactory.create_production_config(
    # Minimal history tracking
    history_enabled=True,
    max_history_length=100,
    store_selection_data=False,
    store_intermediate_grids=False,
    
    # Minimal observation
    observation_format="minimal",
    include_recent_actions=False,
    
    # Basic action space
    dynamic_action_filtering=False
)

# Research configuration with full features (higher memory usage)
research_config = ConfigFactory.create_research_config(
    # Full history tracking
    max_history_length=2000,
    store_selection_data=True,
    store_intermediate_grids=True,
    
    # Rich observation
    observation_format="rich",
    include_recent_actions=True,
    
    # Dynamic action space
    dynamic_action_filtering=True
)
```

#### JAX Compatibility

All enhanced features maintain full JAX compatibility:

```python
# JIT compilation works with enhanced features
@jax.jit
def enhanced_step(state, action, config):
    return arc_step(state, action, config)

# Vectorization across multiple environments
batch_step = jax.vmap(enhanced_step, in_axes=(0, 0, None))

# Parallel processing
parallel_step = jax.pmap(enhanced_step, in_axes=(0, 0, None))
```

### Enhanced Migration Checklist

- [ ] Update reset function calls to handle (state, observation) tuple
- [ ] Update step function calls to handle (state, observation, reward, done, info) tuple
- [ ] Update action operation range validation (35 → 42 operations)
- [ ] Add enhanced configuration groups (episode, history, action, observation)
- [ ] Test multi-demonstration training with pair switching
- [ ] Test proper test pair evaluation without target access
- [ ] Configure action history tracking based on memory requirements
- [ ] Update action space control for dynamic filtering
- [ ] Test enhanced observation space usage
- [ ] Verify JAX compatibility with enhanced features
- [ ] Update examples and documentation for enhanced features
- [ ] Performance test with enhanced features enabled

## Getting Help

If you encounter issues during migration:

1. **Check the examples** in `examples/` directory for updated patterns
2. **Review the API reference** for new function signatures
3. **Use validation utilities** to catch configuration errors early
4. **Enable debug logging** to understand what's happening
5. **Test enhanced features incrementally** starting with basic multi-demo training
6. **Check memory usage** when enabling full action history tracking
7. **Open an issue** on GitHub with your specific migration problem

## Gradual Migration Strategy

You can migrate incrementally without breaking existing code:

### Phase 1: Update Configuration Creation (Low Risk)

1. **Replace factory function calls** with `ConfigFactory` methods
2. **Update debug config references** from "on"/"full" to "standard"/"research"
3. **Test configuration creation** without changing environment usage

### Phase 2: Migrate Environment Instantiation (Medium Risk)

1. **Replace dual config pattern** with single unified config
2. **Update ArcEnvironment constructor calls** to use single config parameter
3. **Test environment creation and basic functionality**

### Phase 3: Update Configuration Access (Medium Risk)

1. **Replace direct parameter access** with grouped access patterns
2. **Update configuration parameter references** throughout your code
3. **Test all configuration-dependent functionality**

### Phase 4: Clean Up and Optimize (Low Risk)

1. **Remove old configuration files** (on.yaml, full.yaml)
2. **Update examples and documentation** to use new patterns
3. **Add comprehensive validation** and error handling

### Backward Compatibility Notes

- **Factory functions**: Old factory functions are deprecated but still work
- **Configuration validation**: New validation is more strict but provides clear
  error messages
- **Debug levels**: Old debug levels ("on", "full") are automatically migrated
  with warnings
- **Parameter access**: Old direct access patterns will show deprecation
  warnings

### Migration Checklist

- [ ] Update configuration creation to use `ConfigFactory`
- [ ] Replace dual config pattern with single unified config
- [ ] Update debug config references (on → standard, full → research)
- [ ] Update parameter access to use grouped patterns
- [ ] Test environment creation and functionality
- [ ] Update examples and scripts
- [ ] Remove old configuration files
- [ ] Add comprehensive validation and error handling
