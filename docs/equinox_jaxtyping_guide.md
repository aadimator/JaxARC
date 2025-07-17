# Equinox and JAXTyping Guide

This guide covers the modern JAX patterns used in JaxARC with Equinox for state management and JAXTyping for precise type annotations.

## Overview

JaxARC uses two key libraries to modernize JAX development:

- **Equinox**: Provides PyTree modules with automatic registration and better error messages
- **JAXTyping**: Enables precise array shape and dtype annotations for type safety

These libraries work together to provide:
- Better type safety and error catching
- Cleaner functional patterns
- Improved JAX transformation compatibility
- Enhanced debugging capabilities

## Equinox Integration

### What is Equinox?

Equinox is a JAX library that provides:
- **PyTree Modules**: Automatic PyTree registration for JAX transformations
- **Better Error Messages**: Clear shape mismatch and type error reporting
- **Functional Patterns**: Clean functional programming patterns for JAX
- **Validation**: Built-in validation through `__check_init__` methods

### ArcEnvState with Equinox

The core environment state uses Equinox Module:

```python
import equinox as eqx
from jaxarc.state import ArcEnvState
from jaxarc.utils.jax_types import GridArray, MaskArray, SimilarityScore

class ArcEnvState(eqx.Module):
    """ARC environment state with Equinox Module for better JAX integration."""
    
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
    
    def __check_init__(self) -> None:
        """Equinox validation method for state structure."""
        # Automatic validation of shapes and types
        # JAXTyping handles most validation automatically
        pass
```

### Creating and Using Equinox States

```python
import jax.numpy as jnp
from jaxarc.state import ArcEnvState

# Create state (typically done by environment)
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
    similarity_score=jnp.array(0.0, dtype=jnp.float32)
)
```

### State Updates with Equinox

Equinox provides several patterns for updating immutable state:

#### Method 1: tree_at (Recommended for single fields)

```python
import equinox as eqx

# Update single field
new_state = eqx.tree_at(
    lambda s: s.step_count, 
    state, 
    state.step_count + 1
)

# Update with computation
new_state = eqx.tree_at(
    lambda s: s.similarity_score,
    state,
    compute_similarity(state.working_grid, state.target_grid)
)
```

#### Method 2: tree_at for multiple fields

```python
# Update multiple fields at once
new_state = eqx.tree_at(
    lambda s: (s.step_count, s.episode_done, s.similarity_score),
    state,
    (
        state.step_count + 1,
        jnp.array(True),
        compute_similarity(state.working_grid, state.target_grid)
    )
)
```

#### Method 3: Custom replace method

```python
# Using the custom replace method for convenience
new_state = state.replace(
    step_count=state.step_count + 1,
    episode_done=True,
    similarity_score=compute_similarity(state.working_grid, state.target_grid)
)
```

### JAX Transformations with Equinox

Equinox modules work seamlessly with all JAX transformations:

```python
# JIT compilation
@jax.jit
def update_state(state: ArcEnvState) -> ArcEnvState:
    return eqx.tree_at(
        lambda s: s.step_count,
        state,
        state.step_count + 1
    )

# Vectorization (vmap)
def process_batch_states(states: ArcEnvState) -> ArcEnvState:
    """Process a batch of states."""
    return jax.vmap(update_state)(states)

# Gradient computation
def state_loss(state: ArcEnvState) -> float:
    """Compute loss from state for gradient-based optimization."""
    return jnp.sum((state.working_grid - state.target_grid) ** 2)

grad_fn = jax.grad(state_loss)
gradients = grad_fn(state)
```

## JAXTyping Integration

### What is JAXTyping?

JAXTyping provides precise array type annotations:
- **Shape Information**: Specify exact array shapes like `"height width"`
- **Dtype Information**: Specify array dtypes like `Int`, `Float`, `Bool`
- **Batch Support**: Use `*batch` for flexible batch dimensions
- **Runtime Validation**: Optional runtime type checking

### Core Type Definitions

```python
from jaxarc.utils.jax_types import (
    # Grid types (support both single and batched operations)
    GridArray,      # Int[Array, "*batch height width"]
    MaskArray,      # Bool[Array, "*batch height width"] 
    SelectionArray, # Bool[Array, "*batch height width"]
    
    # Action types
    PointCoords,    # Int[Array, "2"] - [row, col]
    BboxCoords,     # Int[Array, "4"] - [r1, c1, r2, c2]
    OperationId,    # Int[Array, ""] - scalar operation ID
    
    # Scoring types
    SimilarityScore, # Float[Array, "*batch"] - similarity scores
    RewardValue,     # Float[Array, "*batch"] - reward values
    
    # State types
    StepCount,      # Int[Array, ""] - scalar step count
    EpisodeIndex,   # Int[Array, ""] - scalar episode index
    EpisodeDone,    # Bool[Array, ""] - scalar boolean flag
)
```

### Using JAXTyping Annotations

#### Basic Function Annotations

```python
from jaxarc.utils.jax_types import GridArray, MaskArray, SimilarityScore

def compute_similarity(grid1: GridArray, grid2: GridArray) -> SimilarityScore:
    """Compute similarity between two grids.
    
    Args:
        grid1: First grid with shape (height, width) or (*batch, height, width)
        grid2: Second grid with shape (height, width) or (*batch, height, width)
        
    Returns:
        Similarity score with shape () or (*batch,)
    """
    # JAXTyping validates shapes automatically
    diff = jnp.abs(grid1 - grid2)
    return 1.0 - jnp.mean(diff) / 9.0  # Normalize by max color difference

def apply_mask(grid: GridArray, mask: MaskArray) -> GridArray:
    """Apply mask to grid, preserving background where mask is False.
    
    Args:
        grid: Input grid
        mask: Boolean mask
        
    Returns:
        Masked grid with same shape as input
    """
    return jnp.where(mask, grid, 0)  # Set masked areas to background
```

#### Batch Operations

The `*batch` modifier allows the same type to work for both single arrays and batched arrays:

```python
def batch_compute_similarity(
    grids1: GridArray,  # Shape: (batch, height, width)
    grids2: GridArray   # Shape: (batch, height, width)
) -> SimilarityScore:   # Shape: (batch,)
    """Compute similarity for a batch of grid pairs."""
    return jax.vmap(compute_similarity)(grids1, grids2)

def single_compute_similarity(
    grid1: GridArray,   # Shape: (height, width)
    grid2: GridArray    # Shape: (height, width)
) -> SimilarityScore:  # Shape: ()
    """Compute similarity for a single grid pair."""
    return compute_similarity(grid1, grid2)

# Both functions use the same type annotations!
```

#### Action Processing

```python
from jaxarc.utils.jax_types import PointCoords, BboxCoords, OperationId

def process_point_action(
    point: PointCoords,     # Shape: (2,) - [row, col]
    operation: OperationId, # Shape: () - scalar operation ID
    grid_shape: tuple[int, int]
) -> SelectionArray:        # Shape: (height, width)
    """Convert point action to selection mask."""
    row, col = point
    selection = jnp.zeros(grid_shape, dtype=bool)
    selection = selection.at[row, col].set(True)
    return selection

def process_bbox_action(
    bbox: BboxCoords,       # Shape: (4,) - [r1, c1, r2, c2]
    operation: OperationId, # Shape: () - scalar operation ID
    grid_shape: tuple[int, int]
) -> SelectionArray:        # Shape: (height, width)
    """Convert bounding box action to selection mask."""
    r1, c1, r2, c2 = bbox
    selection = jnp.zeros(grid_shape, dtype=bool)
    selection = selection.at[r1:r2+1, c1:c2+1].set(True)
    return selection
```

### Runtime Type Checking (Optional)

For additional safety, you can enable runtime type checking:

```python
from jaxtyping import jaxtyped
from beartype import beartype

@jaxtyped
@beartype
def safe_grid_operation(
    grid: GridArray,
    mask: MaskArray
) -> GridArray:
    """Function with runtime type validation.
    
    JAXTyping will validate:
    - Array shapes match the annotations
    - Array dtypes are correct
    - Batch dimensions are consistent
    """
    return grid * mask.astype(grid.dtype)

# Usage
grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
mask = jnp.array([[True, False], [False, True]], dtype=bool)

result = safe_grid_operation(grid, mask)  # ✓ Passes validation

# This would raise a runtime error:
# bad_mask = jnp.array([True, False], dtype=bool)  # Wrong shape
# result = safe_grid_operation(grid, bad_mask)  # ✗ Runtime error
```

## Equinox Utilities

JaxARC provides utilities for working with Equinox modules:

### State Debugging

```python
from jaxarc.utils.equinox_utils import (
    print_state_summary,
    tree_map_with_path,
    validate_state_shapes
)

def debug_state(state: ArcEnvState) -> None:
    """Debug state contents and structure."""
    
    # Print comprehensive state summary
    print_state_summary(state, "Current State")
    
    # Validate state structure
    if not validate_state_shapes(state):
        print("⚠️  State validation failed!")
    else:
        print("✅ State validation passed")
    
    # Map function over state with path information
    def print_array_info(path: str, value: Any) -> Any:
        if hasattr(value, 'shape'):
            print(f"  {path}: shape={value.shape}, dtype={value.dtype}")
            if hasattr(value, 'min'):
                print(f"    range=[{value.min():.3f}, {value.max():.3f}]")
        return value
    
    print("\nDetailed array information:")
    tree_map_with_path(print_array_info, state)
```

### State Comparison

```python
from jaxarc.utils.equinox_utils import create_state_diff

def compare_states(old_state: ArcEnvState, new_state: ArcEnvState) -> None:
    """Compare two states and show differences."""
    
    diff = create_state_diff(old_state, new_state)
    
    if not diff:
        print("States are identical")
        return
    
    print("State differences:")
    for path, change_info in diff.items():
        print(f"\n📝 {path}:")
        print(f"   Type: {change_info['type']}")
        
        if change_info['type'] == 'value_change':
            print(f"   Old: {change_info['old']}")
            print(f"   New: {change_info['new']}")
            if 'max_diff' in change_info and change_info['max_diff'] is not None:
                print(f"   Max difference: {change_info['max_diff']:.6f}")
        elif change_info['type'] == 'shape_change':
            print(f"   Old shape: {change_info['old']}")
            print(f"   New shape: {change_info['new']}")
```

### Memory Analysis

```python
from jaxarc.utils.equinox_utils import module_memory_usage

def analyze_memory(state: ArcEnvState) -> None:
    """Analyze memory usage of state."""
    
    memory_info = module_memory_usage(state)
    
    print(f"Total memory: {memory_info['total_bytes']:,} bytes")
    print(f"Total elements: {memory_info['total_elements']:,}")
    
    print("\nPer-array breakdown:")
    for path, info in memory_info['arrays'].items():
        print(f"  {path}:")
        print(f"    Shape: {info['shape']}")
        print(f"    Memory: {info['bytes']:,} bytes")
        print(f"    Elements: {info['elements']:,}")
```

## Best Practices

### 1. Type Annotation Guidelines

```python
# ✅ Good: Use specific JAXTyping annotations
def process_grid(grid: GridArray, mask: MaskArray) -> GridArray:
    pass

# ❌ Avoid: Generic jnp.ndarray annotations
def process_grid(grid: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    pass

# ✅ Good: Use batch-compatible types
def batch_process(grids: GridArray) -> GridArray:  # Works for any batch size
    pass

# ❌ Avoid: Fixed batch size annotations
def batch_process(grids: Int[Array, "32 height width"]) -> Int[Array, "32 height width"]:
    pass
```

### 2. State Update Patterns

```python
# ✅ Good: Use tree_at for single field updates
new_state = eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)

# ✅ Good: Use tree_at for multiple related updates
new_state = eqx.tree_at(
    lambda s: (s.working_grid, s.selected),
    state,
    (updated_grid, new_selection)
)

# ✅ Good: Use replace for many field updates
new_state = state.replace(
    step_count=state.step_count + 1,
    episode_done=True,
    similarity_score=new_similarity
)

# ❌ Avoid: Creating new state objects manually
new_state = ArcEnvState(
    task_data=state.task_data,
    working_grid=updated_grid,
    # ... copying all fields manually
)
```

### 3. JAX Transformation Compatibility

```python
# ✅ Good: Functions work with JAX transformations
@jax.jit
def update_state(state: ArcEnvState, action: dict) -> ArcEnvState:
    return eqx.tree_at(
        lambda s: s.working_grid,
        state,
        apply_action(state.working_grid, action)
    )

# ✅ Good: Batch processing with vmap
batch_update = jax.vmap(update_state, in_axes=(0, 0))

# ✅ Good: Gradient computation
def state_loss(state: ArcEnvState) -> float:
    return jnp.sum((state.working_grid - state.target_grid) ** 2)

grad_fn = jax.grad(state_loss)
```

### 4. Error Handling

```python
from jaxarc.utils.equinox_utils import validate_state_shapes

def safe_state_operation(state: ArcEnvState) -> ArcEnvState:
    """Perform state operation with validation."""
    
    # Validate input state
    if not validate_state_shapes(state):
        raise ValueError("Input state validation failed")
    
    # Perform operation
    new_state = eqx.tree_at(
        lambda s: s.step_count,
        state,
        state.step_count + 1
    )
    
    # Validate output state
    if not validate_state_shapes(new_state):
        raise ValueError("Output state validation failed")
    
    return new_state
```

## Performance Tips

### 1. JIT Compilation

```python
# Equinox modules JIT compile efficiently
@jax.jit
def fast_state_update(state: ArcEnvState) -> ArcEnvState:
    return eqx.tree_at(lambda s: s.step_count, state, state.step_count + 1)

# First call compiles, subsequent calls are fast
state = create_initial_state()
fast_state = fast_state_update(state)  # Compilation happens here
fast_state = fast_state_update(fast_state)  # Fast execution
```

### 2. Memory Efficiency

```python
# Use tree_at for minimal memory allocation
def efficient_update(state: ArcEnvState, new_grid: GridArray) -> ArcEnvState:
    # Only allocates memory for the new state tree, reuses unchanged parts
    return eqx.tree_at(lambda s: s.working_grid, state, new_grid)

# Avoid unnecessary copying
def inefficient_update(state: ArcEnvState, new_grid: GridArray) -> ArcEnvState:
    # Creates entirely new state object
    return state.replace(
        task_data=state.task_data,  # Unnecessary copy
        working_grid=new_grid,
        working_grid_mask=state.working_grid_mask,  # Unnecessary copy
        # ... all other fields
    )
```

### 3. Batch Processing

```python
# Leverage JAXTyping's batch support for efficient vectorization
def process_batch_efficiently(states: ArcEnvState) -> ArcEnvState:
    """Process batch of states efficiently."""
    
    # Single vmap call processes entire batch
    return jax.vmap(lambda s: eqx.tree_at(
        lambda x: x.step_count, 
        s, 
        s.step_count + 1
    ))(states)

# Use consistent batch dimensions
def batch_similarity(
    grids1: GridArray,  # Shape: (batch, height, width)
    grids2: GridArray   # Shape: (batch, height, width)
) -> SimilarityScore:  # Shape: (batch,)
    return jax.vmap(compute_similarity)(grids1, grids2)
```

## Examples

### Complete State Management Example

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxarc.state import ArcEnvState
from jaxarc.utils.jax_types import GridArray, MaskArray
from jaxarc.utils.equinox_utils import print_state_summary, create_state_diff

def complete_example():
    """Complete example of Equinox state management."""
    
    # Create initial state
    initial_grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
    target_grid = jnp.array([[4, 3], [2, 1]], dtype=jnp.int32)
    
    state = ArcEnvState(
        task_data=None,  # Would be actual task data
        working_grid=initial_grid,
        working_grid_mask=jnp.ones_like(initial_grid, dtype=bool),
        target_grid=target_grid,
        step_count=jnp.array(0, dtype=jnp.int32),
        episode_done=jnp.array(False, dtype=bool),
        current_example_idx=jnp.array(0, dtype=jnp.int32),
        selected=jnp.zeros_like(initial_grid, dtype=bool),
        clipboard=jnp.zeros_like(initial_grid, dtype=jnp.int32),
        similarity_score=jnp.array(0.0, dtype=jnp.float32)
    )
    
    print("Initial state:")
    print_state_summary(state, "Initial")
    
    # Update state using Equinox patterns
    new_state = eqx.tree_at(
        lambda s: (s.step_count, s.working_grid, s.similarity_score),
        state,
        (
            state.step_count + 1,
            jnp.array([[4, 3], [2, 1]], dtype=jnp.int32),  # Match target
            jnp.array(1.0, dtype=jnp.float32)  # Perfect similarity
        )
    )
    
    print("\nAfter update:")
    print_state_summary(new_state, "Updated")
    
    # Compare states
    print("\nState differences:")
    diff = create_state_diff(state, new_state)
    for path, change_info in diff.items():
        print(f"  {path}: {change_info['type']}")
        if 'old' in change_info and 'new' in change_info:
            print(f"    {change_info['old']} → {change_info['new']}")

if __name__ == "__main__":
    complete_example()
```

This guide provides comprehensive coverage of Equinox and JAXTyping usage in JaxARC. These modern patterns provide better type safety, cleaner code, and improved JAX integration while maintaining high performance.