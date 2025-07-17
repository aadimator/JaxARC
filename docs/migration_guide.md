# Migration Guide

This guide helps you migrate from older JaxARC patterns to the new modernized codebase with Equinox, JAXTyping, and streamlined configuration.

## Overview of Changes

The JaxARC codebase has been significantly refactored to:

- **Eliminate code duplication** with centralized type definitions
- **Modernize JAX patterns** using Equinox and JAXTyping
- **Simplify configuration** with Hydra-first approach
- **Improve type safety** with runtime validation
- **Streamline action handling** with cleaner architecture

## Breaking Changes Summary

### 1. State Management (Equinox Migration)

**Before (chex dataclass):**
```python
from jaxarc.envs.arc_base import ArcEnvState  # Multiple definitions existed

@chex.dataclass
class ArcEnvState:
    working_grid: jnp.ndarray
    # ... manual validation in __post_init__
```

**After (Equinox Module):**
```python
from jaxarc.state import ArcEnvState  # Single canonical definition

class ArcEnvState(eqx.Module):
    working_grid: GridArray  # JAXTyping provides shape validation
    # ... automatic PyTree registration and validation
```

### 2. Type Annotations (JAXTyping)

**Before (generic arrays):**
```python
def compute_similarity(grid1: jnp.ndarray, grid2: jnp.ndarray) -> jnp.ndarray:
    pass
```

**After (precise type annotations):**
```python
from jaxarc.utils.jax_types import GridArray, SimilarityScore

def compute_similarity(grid1: GridArray, grid2: GridArray) -> SimilarityScore:
    pass
```

### 3. Configuration System

**Before (verbose factory functions):**
```python
def create_standard_config(max_steps=100, penalty=-0.01, ...):
    reward_config = RewardConfig(...)
    grid_config = GridConfig(...)
    # ... 50+ lines of boilerplate
```

**After (Hydra composition):**
```python
# conf/presets/standard.yaml handles composition
config = ArcEnvConfig.from_hydra(hydra_cfg)
```

## Step-by-Step Migration

### Step 1: Update Imports

**Old imports:**
```python
from jaxarc.envs.arc_base import ArcEnvState
from jaxarc.envs.functional import ArcEnvState  # Duplicate definition
```

**New imports:**
```python
from jaxarc.state import ArcEnvState  # Single source of truth
from jaxarc.utils.jax_types import GridArray, MaskArray, SimilarityScore
from jaxarc.utils.equinox_utils import tree_map_with_path, validate_state_shapes
```

### Step 2: Update State Usage

**Old state updates:**
```python
# Using chex dataclass replace
new_state = state.replace(
    step_count=state.step_count + 1,
    episode_done=True
)
```

**New state updates (Equinox patterns):**
```python
# Method 1: Using Equinox tree_at (recommended for single field)
new_state = eqx.tree_at(
    lambda s: s.step_count, 
    state, 
    state.step_count + 1
)

# Method 2: Using replace method (for multiple fields)
new_state = state.replace(
    step_count=state.step_count + 1,
    episode_done=True
)

# Method 3: Using tree_at for multiple fields
new_state = eqx.tree_at(
    lambda s: (s.step_count, s.episode_done),
    state,
    (state.step_count + 1, jnp.array(True))
)
```

### Step 3: Add Type Annotations

**Before:**
```python
def process_grid(grid, mask):
    return grid * mask

def calculate_reward(old_grid, new_grid, target_grid):
    similarity = compute_similarity(new_grid, target_grid)
    return similarity
```

**After:**
```python
from jaxarc.utils.jax_types import GridArray, MaskArray, SimilarityScore

def process_grid(grid: GridArray, mask: MaskArray) -> GridArray:
    return grid * mask

def calculate_reward(
    old_grid: GridArray, 
    new_grid: GridArray, 
    target_grid: GridArray
) -> SimilarityScore:
    similarity = compute_similarity(new_grid, target_grid)
    return similarity
```

### Step 4: Update Configuration Usage

**Old configuration creation:**
```python
from jaxarc.envs.factory import create_standard_config

config = create_standard_config(
    max_episode_steps=100,
    success_bonus=10.0,
    step_penalty=-0.01
)
```

**New configuration (factory functions still work):**
```python
from jaxarc.envs.factory import create_standard_config

# Factory functions still work for backward compatibility
config = create_standard_config(
    max_episode_steps=100,
    success_bonus=10.0,
    step_penalty=-0.01
)
```

**New configuration (Hydra approach - recommended):**
```python
import hydra
from omegaconf import DictConfig
from jaxarc.envs.config import ArcEnvConfig

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    config = ArcEnvConfig.from_hydra(cfg)
    # ... rest of your code
```

### Step 5: Update Parser Usage

**Parser usage remains the same:**
```python
from jaxarc.parsers import ArcAgiParser, ConceptArcParser, MiniArcParser

# No changes needed - parsers use inheritance now but API is the same
parser = ArcAgiParser(parser_config)
task = parser.get_random_task(key)
```

## New Features and Capabilities

### 1. Enhanced Type Safety with JAXTyping

```python
from jaxarc.utils.jax_types import (
    GridArray, MaskArray, SelectionArray, SimilarityScore,
    PointCoords, BboxCoords, OperationId
)

# Precise type annotations catch errors early
def apply_operation(
    grid: GridArray,
    selection: SelectionArray, 
    operation: OperationId
) -> GridArray:
    # JAXTyping validates shapes and types automatically
    pass

# Batch operations work with the same types
def batch_apply_operation(
    grids: GridArray,  # Shape: (batch, height, width)
    selections: SelectionArray,  # Shape: (batch, height, width)
    operations: OperationId  # Shape: (batch,)
) -> GridArray:  # Shape: (batch, height, width)
    pass
```

### 2. Equinox Utilities

```python
from jaxarc.utils.equinox_utils import (
    tree_map_with_path, 
    validate_state_shapes,
    create_state_diff,
    print_state_summary
)

# Debug state contents
def debug_state(state: ArcEnvState) -> None:
    print_state_summary(state, "Current State")
    
    # Validate state structure
    if not validate_state_shapes(state):
        print("State validation failed!")
    
    # Map function over state with path information
    def print_shapes(path: str, value: Any) -> Any:
        if hasattr(value, 'shape'):
            print(f"{path}: {value.shape}")
        return value
    
    tree_map_with_path(print_shapes, state)

# Compare states
def compare_states(old_state: ArcEnvState, new_state: ArcEnvState) -> None:
    diff = create_state_diff(old_state, new_state)
    for path, change_info in diff.items():
        print(f"Changed: {path}")
        print(f"  Type: {change_info['type']}")
        if 'old' in change_info:
            print(f"  Old: {change_info['old']}")
        if 'new' in change_info:
            print(f"  New: {change_info['new']}")
```

### 3. Enhanced Configuration Validation

```python
from jaxarc.envs.config import ArcEnvConfig, RewardConfig, ActionConfig

# Comprehensive validation with clear error messages
try:
    config = ArcEnvConfig(
        max_episode_steps=100,
        reward=RewardConfig(
            success_bonus=10.0,
            step_penalty=-0.01,
            similarity_weight=1.5
        ),
        action=ActionConfig(
            selection_format="point",
            num_operations=35,
            allowed_operations=[0, 1, 2, 3, 4, 5]
        )
    )
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
```

### 4. Runtime Type Checking (Optional)

```python
from jaxtyping import jaxtyped
from beartype import beartype

@jaxtyped
@beartype
def process_grid_safe(grid: GridArray) -> MaskArray:
    """Function with runtime type validation."""
    # Automatic shape and type validation at runtime
    return grid > 0
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

### Issue 1: Import Errors

**Problem:**
```python
ImportError: cannot import name 'ArcEnvState' from 'jaxarc.envs.arc_base'
```

**Solution:**
```python
# Change from:
from jaxarc.envs.arc_base import ArcEnvState

# To:
from jaxarc.state import ArcEnvState
```

### Issue 2: Type Annotation Errors

**Problem:**
```python
TypeError: Expected GridArray, got ndarray
```

**Solution:**
```python
# Add proper imports and type annotations
from jaxarc.utils.jax_types import GridArray
import jax.numpy as jnp

# Ensure arrays have correct type
grid: GridArray = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
```

### Issue 3: Configuration Validation Errors

**Problem:**
```python
ConfigValidationError: step_penalty must be in range [-10.0, 1.0], got 5.0
```

**Solution:**
```python
# Fix configuration values to be within valid ranges
config = ArcEnvConfig(
    reward=RewardConfig(
        step_penalty=-0.01,  # Should be negative or zero
        success_bonus=10.0,   # Should be positive
    )
)
```

### Issue 4: State Update Patterns

**Problem:**
```python
AttributeError: 'ArcEnvState' object has no attribute 'replace'
```

**Solution:**
```python
# Use Equinox patterns for state updates
import equinox as eqx

# Method 1: tree_at for single field
new_state = eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)

# Method 2: Custom replace method (if available)
new_state = state.replace(step_count=state.step_count + 1)
```

## Testing Your Migration

### 1. Validate State Structure

```python
from jaxarc.utils.equinox_utils import validate_state_shapes

def test_state_migration():
    # Create state using new patterns
    state = create_initial_state()
    
    # Validate structure
    assert validate_state_shapes(state), "State validation failed"
    
    # Test JAX transformations
    @jax.jit
    def test_jit(s):
        return s.step_count + 1
    
    result = test_jit(state)
    assert result == state.step_count + 1
```

### 2. Test Configuration

```python
def test_config_migration():
    # Test old factory function still works
    old_config = create_standard_config()
    assert isinstance(old_config, ArcEnvConfig)
    
    # Test new Hydra approach
    hydra_cfg = OmegaConf.create({
        "max_episode_steps": 100,
        "reward": {"success_bonus": 10.0}
    })
    new_config = ArcEnvConfig.from_hydra(hydra_cfg)
    assert isinstance(new_config, ArcEnvConfig)
```

### 3. Performance Benchmarks

```python
import time

def benchmark_migration():
    # Compare old vs new patterns
    state = create_initial_state()
    
    # Benchmark state updates
    start = time.time()
    for _ in range(1000):
        state = eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)
    equinox_time = time.time() - start
    
    print(f"Equinox state updates: {equinox_time:.4f}s")
```

## Getting Help

If you encounter issues during migration:

1. **Check the examples** in `examples/` directory for updated patterns
2. **Review the API reference** for new function signatures
3. **Use validation utilities** to catch configuration errors early
4. **Enable debug logging** to understand what's happening
5. **Open an issue** on GitHub with your specific migration problem

## Gradual Migration Strategy

You don't need to migrate everything at once:

1. **Start with imports** - update to use centralized definitions
2. **Add type annotations** gradually to catch errors early  
3. **Update configuration** when you need new features
4. **Migrate state handling** when you encounter issues
5. **Add Equinox utilities** for debugging and validation as needed

The old patterns are still supported for backward compatibility, so you can migrate incrementally.