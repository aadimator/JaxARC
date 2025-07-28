# Structured Actions API Reference

This document provides comprehensive API documentation for JaxARC's structured action system, which replaces dictionary-based actions with JAX-compatible Equinox modules for optimal performance and type safety.

## Overview

Structured actions are JAX-compatible data structures that represent agent actions in the ARC environment. They provide:

- **Type Safety**: Compile-time type checking with proper annotations
- **JAX Compatibility**: Full support for JIT compilation, vmap, and other JAX transformations
- **Memory Efficiency**: Format-specific storage optimizations
- **Validation**: Built-in validation methods for action parameters
- **Conversion**: Efficient conversion to selection masks for grid operations

## Base Action Interface

All structured actions inherit from the `BaseAction` class:

```python
from jaxarc.envs.structured_actions import BaseAction

class BaseAction(eqx.Module):
    """Base class for all structured actions."""
    
    operation: OperationId
    
    @abc.abstractmethod
    def to_selection_mask(self, grid_shape: tuple[int, int]) -> SelectionArray:
        """Convert action to selection mask."""
        pass
    
    @abc.abstractmethod
    def validate(self, grid_shape: tuple[int, int], max_operations: int = 42) -> 'BaseAction':
        """Validate action parameters and return validated action."""
        pass
```

## Action Types

### PointAction

Represents a single-point selection on the grid.

#### Class Definition

```python
class PointAction(BaseAction):
    """Point-based action using single coordinate.
    
    This action type selects a single point on the grid using row and column
    coordinates. It's the most memory-efficient action format.
    
    Attributes:
        operation: ARCLE operation ID (0-41)
        row: Row coordinate (0-based)
        col: Column coordinate (0-based)
    """
    
    operation: jnp.int32
    row: jnp.int32
    col: jnp.int32
```

#### Usage Examples

```python
from jaxarc.envs.structured_actions import PointAction
import jax.numpy as jnp

# Create a point action
action = PointAction(
    operation=jnp.array(0),  # Fill operation
    row=jnp.array(5),        # Row 5
    col=jnp.array(7)         # Column 7
)

# Convert to selection mask
grid_shape = (30, 30)
selection_mask = action.to_selection_mask(grid_shape)
# Result: Boolean mask with single point at (5, 7) set to True

# Validate action
validated_action = action.validate(grid_shape, max_operations=42)
```

#### Batch Usage

```python
# Create batch of point actions
batch_size = 16
batch_actions = PointAction(
    operation=jnp.zeros(batch_size, dtype=jnp.int32),
    row=jnp.arange(batch_size, dtype=jnp.int32) % 10 + 5,
    col=jnp.full(batch_size, 7, dtype=jnp.int32)
)

# Convert batch to selection masks
batch_masks = jax.vmap(
    lambda action: action.to_selection_mask(grid_shape)
)(batch_actions)
# Result: (batch_size, height, width) boolean array
```

#### Memory Usage

- **Fields per action**: 3 (operation, row, col)
- **Memory per action**: ~12 bytes
- **Memory efficiency**: 99.3% reduction vs mask actions

### BboxAction

Represents a rectangular bounding box selection on the grid.

#### Class Definition

```python
class BboxAction(BaseAction):
    """Bounding box action using rectangular coordinates.
    
    This action type selects a rectangular region on the grid using
    top-left and bottom-right coordinates.
    
    Attributes:
        operation: ARCLE operation ID (0-41)
        r1: Top-left row coordinate (0-based)
        c1: Top-left column coordinate (0-based)
        r2: Bottom-right row coordinate (0-based)
        c2: Bottom-right column coordinate (0-based)
    """
    
    operation: jnp.int32
    r1: jnp.int32
    c1: jnp.int32
    r2: jnp.int32
    c2: jnp.int32
```

#### Usage Examples

```python
from jaxarc.envs.structured_actions import BboxAction

# Create a bounding box action
action = BboxAction(
    operation=jnp.array(0),  # Fill operation
    r1=jnp.array(3),         # Top-left row
    c1=jnp.array(3),         # Top-left column
    r2=jnp.array(7),         # Bottom-right row
    c2=jnp.array(7)          # Bottom-right column
)

# Convert to selection mask
selection_mask = action.to_selection_mask(grid_shape)
# Result: Boolean mask with rectangle from (3,3) to (7,7) set to True

# Get selected area
selected_cells = jnp.sum(selection_mask)  # 25 cells (5x5 rectangle)
```

#### Coordinate System

```
Grid coordinates (0-based):
  0 1 2 3 4 5 6 7 8 9
0 . . . . . . . . . .
1 . . . . . . . . . .
2 . . . . . . . . . .
3 . . . ■ ■ ■ ■ ■ . .  <- r1=3, c1=3
4 . . . ■ ■ ■ ■ ■ . .
5 . . . ■ ■ ■ ■ ■ . .
6 . . . ■ ■ ■ ■ ■ . .
7 . . . ■ ■ ■ ■ ■ . .  <- r2=7, c2=7
8 . . . . . . . . . .
9 . . . . . . . . . .
```

#### Validation

```python
# Automatic coordinate ordering
action = BboxAction(
    operation=jnp.array(0),
    r1=jnp.array(7),  # Will be swapped with r2
    c1=jnp.array(7),  # Will be swapped with c2
    r2=jnp.array(3),  # Smaller coordinate
    c2=jnp.array(3)   # Smaller coordinate
)

validated_action = action.validate(grid_shape)
# Result: r1=3, c1=3, r2=7, c2=7 (coordinates properly ordered)
```

#### Memory Usage

- **Fields per action**: 5 (operation, r1, c1, r2, c2)
- **Memory per action**: ~20 bytes
- **Memory efficiency**: 99.1% reduction vs mask actions

### MaskAction

Represents an arbitrary selection pattern using a boolean mask.

#### Class Definition

```python
class MaskAction(BaseAction):
    """Mask-based action using arbitrary selection pattern.
    
    This action type allows for complex, arbitrary selection patterns
    using a boolean mask. It provides maximum flexibility but uses
    the most memory.
    
    Attributes:
        operation: ARCLE operation ID (0-41)
        selection: Boolean mask indicating selected cells
    """
    
    operation: jnp.int32
    selection: SelectionArray  # Shape: (height, width)
```

#### Usage Examples

```python
from jaxarc.envs.structured_actions import MaskAction

# Create custom selection pattern
mask = jnp.zeros((30, 30), dtype=jnp.bool_)

# L-shaped selection
mask = mask.at[10:15, 10].set(True)    # Vertical line
mask = mask.at[14, 10:15].set(True)    # Horizontal line

# Create mask action
action = MaskAction(
    operation=jnp.array(0),  # Fill operation
    selection=mask
)

# Convert to selection mask (returns the mask directly)
selection_mask = action.to_selection_mask(grid_shape)
assert jnp.array_equal(selection_mask, mask)
```

#### Complex Patterns

```python
# Circular selection
def create_circular_mask(center_row, center_col, radius, grid_shape):
    rows, cols = jnp.meshgrid(
        jnp.arange(grid_shape[0]), 
        jnp.arange(grid_shape[1]), 
        indexing='ij'
    )
    distances = jnp.sqrt((rows - center_row)**2 + (cols - center_col)**2)
    return distances <= radius

circular_mask = create_circular_mask(15, 15, 5, (30, 30))
circular_action = MaskAction(
    operation=jnp.array(0),
    selection=circular_mask
)

# Checkerboard pattern
checkerboard_mask = (jnp.arange(30)[:, None] + jnp.arange(30)) % 2 == 0
checkerboard_action = MaskAction(
    operation=jnp.array(0),
    selection=checkerboard_mask
)
```

#### Memory Usage

- **Fields per action**: 1 + mask size (operation + height × width)
- **Memory per action**: ~3.6MB for 30×30 grid
- **Use case**: Complex selection patterns that cannot be represented by point/bbox

## Factory Functions

Convenience functions for creating structured actions:

### create_point_action

```python
from jaxarc.envs.structured_actions import create_point_action

def create_point_action(
    operation: int, 
    row: int, 
    col: int
) -> PointAction:
    """Create a PointAction with automatic type conversion."""
    return PointAction(
        operation=jnp.array(operation, dtype=jnp.int32),
        row=jnp.array(row, dtype=jnp.int32),
        col=jnp.array(col, dtype=jnp.int32)
    )

# Usage
action = create_point_action(operation=0, row=5, col=7)
```

### create_bbox_action

```python
from jaxarc.envs.structured_actions import create_bbox_action

def create_bbox_action(
    operation: int,
    r1: int, c1: int,
    r2: int, c2: int
) -> BboxAction:
    """Create a BboxAction with automatic type conversion."""
    return BboxAction(
        operation=jnp.array(operation, dtype=jnp.int32),
        r1=jnp.array(r1, dtype=jnp.int32),
        c1=jnp.array(c1, dtype=jnp.int32),
        r2=jnp.array(r2, dtype=jnp.int32),
        c2=jnp.array(c2, dtype=jnp.int32)
    )

# Usage
action = create_bbox_action(operation=0, r1=3, c1=3, r2=7, c2=7)
```

### create_mask_action

```python
from jaxarc.envs.structured_actions import create_mask_action

def create_mask_action(
    operation: int,
    selection: jnp.ndarray
) -> MaskAction:
    """Create a MaskAction with automatic type conversion."""
    return MaskAction(
        operation=jnp.array(operation, dtype=jnp.int32),
        selection=selection.astype(jnp.bool_)
    )

# Usage
mask = jnp.zeros((30, 30), dtype=jnp.bool_)
mask = mask.at[10:15, 10:15].set(True)
action = create_mask_action(operation=0, selection=mask)
```

## JAX Transformations

### JIT Compilation

All structured actions are JIT-compatible:

```python
import equinox as eqx

@eqx.filter_jit
def process_action(action, grid_shape):
    """JIT-compiled action processing."""
    selection_mask = action.to_selection_mask(grid_shape)
    return jnp.sum(selection_mask)  # Count selected cells

# Works with all action types
point_count = process_action(point_action, grid_shape)
bbox_count = process_action(bbox_action, grid_shape)
mask_count = process_action(mask_action, grid_shape)
```

### Vectorization (vmap)

Batch processing with `jax.vmap`:

```python
# Create batch of mixed action types (same structure)
batch_point_actions = PointAction(
    operation=jnp.array([0, 1, 0, 1]),
    row=jnp.array([5, 6, 7, 8]),
    col=jnp.array([5, 5, 5, 5])
)

# Vectorized processing
batch_masks = jax.vmap(
    lambda action: action.to_selection_mask(grid_shape)
)(batch_point_actions)

# Result shape: (4, 30, 30)
print(f"Batch masks shape: {batch_masks.shape}")
```

### Gradient Computation

Actions can be used in differentiable computations:

```python
def action_loss(action_params, target_mask, grid_shape):
    """Compute loss for action parameters."""
    # Create action from parameters
    action = PointAction(
        operation=jnp.array(0),
        row=action_params[0],
        col=action_params[1]
    )
    
    # Get selection mask
    selection_mask = action.to_selection_mask(grid_shape)
    
    # Compute loss (e.g., overlap with target)
    overlap = jnp.sum(selection_mask & target_mask)
    return -overlap  # Maximize overlap

# Compute gradients
grad_fn = jax.grad(action_loss)
gradients = grad_fn(
    jnp.array([5.0, 7.0]),  # row, col parameters
    target_mask,
    grid_shape
)
```

## Validation and Error Handling

### Built-in Validation

All actions have built-in validation methods:

```python
# Validate point action
try:
    validated_action = point_action.validate(
        grid_shape=(30, 30),
        max_operations=42
    )
except ValueError as e:
    print(f"Validation error: {e}")

# Validation checks:
# - Operation ID in valid range [0, max_operations)
# - Coordinates within grid bounds
# - Bbox coordinates properly ordered
# - Mask shape matches grid shape
```

### JAX-Compatible Error Handling

Use with `equinox.error_if` for runtime validation:

```python
from jaxarc.utils.error_handling import JAXErrorHandler

@eqx.filter_jit
def safe_action_processing(action, config):
    """Process action with JAX-compatible error handling."""
    # Validate action
    validated_action = JAXErrorHandler.validate_action(action, config)
    
    # Process validated action
    return validated_action.to_selection_mask(
        (config.dataset.max_grid_height, config.dataset.max_grid_width)
    )
```

## Performance Considerations

### Memory Usage Guidelines

Choose action format based on selection pattern:

```python
# ✅ Good: Use point for single-cell operations
if selection_is_single_cell:
    action = PointAction(operation=op, row=r, col=c)

# ✅ Good: Use bbox for rectangular regions
elif selection_is_rectangular:
    action = BboxAction(operation=op, r1=r1, c1=c1, r2=r2, c2=c2)

# ✅ Good: Use mask only for complex patterns
else:
    action = MaskAction(operation=op, selection=complex_mask)
```

### Batch Size Optimization

Optimal batch sizes for different action types:

| Action Type | Optimal Batch Size | Memory per Batch (30×30 grid) |
|-------------|-------------------|-------------------------------|
| Point       | 1000+             | ~12KB                         |
| Bbox        | 500+              | ~20KB                         |
| Mask        | 16-64             | ~230MB                        |

### JIT Compilation Tips

1. **Consistent shapes**: Keep action batch sizes consistent to avoid recompilation
2. **Static operations**: Use static operation IDs when possible
3. **Warm-up**: Always warm up JIT functions before benchmarking

```python
# ✅ Good: Consistent batch shapes
def create_consistent_batch(batch_size, action_type="point"):
    if action_type == "point":
        return PointAction(
            operation=jnp.zeros(batch_size, dtype=jnp.int32),
            row=jnp.zeros(batch_size, dtype=jnp.int32),
            col=jnp.zeros(batch_size, dtype=jnp.int32)
        )
    # ... other types

# ❌ Bad: Inconsistent batch shapes cause recompilation
# batch_8 = create_batch(8)
# batch_16 = create_batch(16)  # Triggers recompilation
```

## Migration Guide

### From Dictionary Actions

**Before (Dictionary Actions - Deprecated):**
```python
# ❌ Old dictionary format
action = {
    "operation": 0,
    "selection": jnp.array([5, 7])  # Point coordinates
}

# ❌ Old bbox format
action = {
    "operation": 0,
    "selection": jnp.array([3, 3, 7, 7])  # Bbox coordinates
}

# ❌ Old mask format
action = {
    "operation": 0,
    "selection": mask_array  # Boolean mask
}
```

**After (Structured Actions):**
```python
# ✅ New point format
action = PointAction(
    operation=jnp.array(0),
    row=jnp.array(5),
    col=jnp.array(7)
)

# ✅ New bbox format
action = BboxAction(
    operation=jnp.array(0),
    r1=jnp.array(3), c1=jnp.array(3),
    r2=jnp.array(7), c2=jnp.array(7)
)

# ✅ New mask format
action = MaskAction(
    operation=jnp.array(0),
    selection=mask_array
)
```

### Conversion Utilities

For migrating existing code:

```python
def convert_dict_to_structured(dict_action, action_format="point"):
    """Convert dictionary action to structured action."""
    operation = jnp.array(dict_action["operation"], dtype=jnp.int32)
    selection = dict_action["selection"]
    
    if action_format == "point":
        return PointAction(
            operation=operation,
            row=jnp.array(selection[0], dtype=jnp.int32),
            col=jnp.array(selection[1], dtype=jnp.int32)
        )
    elif action_format == "bbox":
        return BboxAction(
            operation=operation,
            r1=jnp.array(selection[0], dtype=jnp.int32),
            c1=jnp.array(selection[1], dtype=jnp.int32),
            r2=jnp.array(selection[2], dtype=jnp.int32),
            c2=jnp.array(selection[3], dtype=jnp.int32)
        )
    elif action_format == "mask":
        return MaskAction(
            operation=operation,
            selection=selection.astype(jnp.bool_)
        )
    else:
        raise ValueError(f"Unknown action format: {action_format}")
```

## Examples

### Complete Usage Example

```python
import jax
import jax.numpy as jnp
from jaxarc.envs import JaxArcConfig, EnvironmentConfig, UnifiedActionConfig
from jaxarc.envs.functional import arc_reset, arc_step
from jaxarc.envs.structured_actions import PointAction, BboxAction, MaskAction

# Setup
config = JaxArcConfig(
    environment=EnvironmentConfig(max_episode_steps=100),
    action=UnifiedActionConfig(selection_format="point")
)
task = create_mock_task()
key = jax.random.PRNGKey(42)

# Reset environment
state, obs = arc_reset(key, config, task)

# Create different action types
point_action = PointAction(
    operation=jnp.array(0),  # Fill
    row=jnp.array(5),
    col=jnp.array(7)
)

bbox_action = BboxAction(
    operation=jnp.array(1),  # Different operation
    r1=jnp.array(3), c1=jnp.array(3),
    r2=jnp.array(7), c2=jnp.array(7)
)

# Step with different actions
state1, obs1, reward1, done1, info1 = arc_step(state, point_action, config)
state2, obs2, reward2, done2, info2 = arc_step(state1, bbox_action, config)

print(f"Point action reward: {reward1}")
print(f"Bbox action reward: {reward2}")
```

### Batch Processing Example

```python
# Batch processing with mixed action types
batch_size = 32

# Create batch of point actions
batch_actions = PointAction(
    operation=jnp.zeros(batch_size, dtype=jnp.int32),
    row=jax.random.randint(key, (batch_size,), 0, 30),
    col=jax.random.randint(key, (batch_size,), 0, 30)
)

# Batch reset
keys = jax.random.split(key, batch_size)
batch_states, batch_obs = batch_reset(keys, config, task)

# Batch step
batch_states, batch_obs, rewards, dones, infos = batch_step(
    batch_states, batch_actions, config
)

print(f"Batch rewards: {rewards}")
print(f"Average reward: {jnp.mean(rewards)}")
```

This comprehensive API reference covers all aspects of JaxARC's structured action system. For more examples and advanced usage patterns, see the example scripts in `examples/advanced/`.