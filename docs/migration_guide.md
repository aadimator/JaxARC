# Migration Guide: Dictionary Actions to Structured Actions

This guide helps you migrate from the deprecated dictionary-based action system to the new structured action system in JaxARC. The structured action system provides better performance, type safety, and JAX compatibility.

## Overview

The migration involves replacing dictionary-based actions with structured action classes:

- **Before**: `{"operation": 0, "selection": [5, 7]}`
- **After**: `PointAction(operation=jnp.array(0), row=jnp.array(5), col=jnp.array(7))`

## Why Migrate?

### Benefits of Structured Actions

1. **JAX Compatibility**: Full support for JIT compilation and vectorization
2. **Type Safety**: Compile-time type checking and validation
3. **Memory Efficiency**: 99%+ memory reduction for point/bbox actions
4. **Performance**: 10-100x speedup with JAX optimizations
5. **Better APIs**: Clear, documented interfaces with validation

### Deprecated Dictionary Actions

Dictionary actions are no longer supported and will cause errors:

```python
# ❌ This will fail
action = {"operation": 0, "selection": [5, 7]}
state, obs, reward, done, info = arc_step(state, action, config)
# TypeError: Expected StructuredAction, got dict
```

## Migration Steps

### Step 1: Update Imports

Add imports for structured actions:

```python
# Add these imports
from jaxarc.envs.structured_actions import PointAction, BboxAction, MaskAction
import jax.numpy as jnp
```

### Step 2: Replace Action Creation

#### Point Actions

**Before:**
```python
# ❌ Dictionary format
action = {
    "operation": 0,
    "selection": [5, 7]  # [row, col]
}
```

**After:**
```python
# ✅ Structured format
action = PointAction(
    operation=jnp.array(0),
    row=jnp.array(5),
    col=jnp.array(7)
)
```

#### Bounding Box Actions

**Before:**
```python
# ❌ Dictionary format
action = {
    "operation": 1,
    "selection": [3, 3, 7, 7]  # [r1, c1, r2, c2]
}
```

**After:**
```python
# ✅ Structured format
action = BboxAction(
    operation=jnp.array(1),
    r1=jnp.array(3),
    c1=jnp.array(3),
    r2=jnp.array(7),
    c2=jnp.array(7)
)
```

#### Mask Actions

**Before:**
```python
# ❌ Dictionary format
mask = jnp.zeros((30, 30), dtype=jnp.bool_)
mask = mask.at[10:15, 10:15].set(True)

action = {
    "operation": 2,
    "selection": mask
}
```

**After:**
```python
# ✅ Structured format
mask = jnp.zeros((30, 30), dtype=jnp.bool_)
mask = mask.at[10:15, 10:15].set(True)

action = MaskAction(
    operation=jnp.array(2),
    selection=mask
)
```

### Step 3: Update Configuration

Ensure your configuration specifies the correct action format:

```python
from jaxarc.envs import JaxArcConfig, EnvironmentConfig, UnifiedActionConfig

# Specify action format in configuration
config = JaxArcConfig(
    environment=EnvironmentConfig(max_episode_steps=100),
    action=UnifiedActionConfig(
        selection_format="point"  # or "bbox" or "mask"
    )
)
```

### Step 4: Update Batch Processing

#### Batch Point Actions

**Before:**
```python
# ❌ List of dictionaries
batch_actions = [
    {"operation": 0, "selection": [5, 5]},
    {"operation": 0, "selection": [6, 6]},
    {"operation": 0, "selection": [7, 7]},
    {"operation": 0, "selection": [8, 8]}
]
```

**After:**
```python
# ✅ Single structured action with batch dimension
batch_actions = PointAction(
    operation=jnp.zeros(4, dtype=jnp.int32),
    row=jnp.array([5, 6, 7, 8]),
    col=jnp.array([5, 6, 7, 8])
)
```

#### Batch Bounding Box Actions

**Before:**
```python
# ❌ List of dictionaries
batch_actions = [
    {"operation": 1, "selection": [3, 3, 7, 7]},
    {"operation": 1, "selection": [4, 4, 8, 8]},
    {"operation": 1, "selection": [5, 5, 9, 9]}
]
```

**After:**
```python
# ✅ Single structured action with batch dimension
batch_actions = BboxAction(
    operation=jnp.ones(3, dtype=jnp.int32),
    r1=jnp.array([3, 4, 5]),
    c1=jnp.array([3, 4, 5]),
    r2=jnp.array([7, 8, 9]),
    c2=jnp.array([7, 8, 9])
)
```

## Conversion Utilities

### Automatic Conversion Function

Use this utility to convert existing dictionary actions:

```python
def convert_dict_to_structured(dict_action, action_format="point"):
    """Convert dictionary action to structured action.
    
    Args:
        dict_action: Dictionary with 'operation' and 'selection' keys
        action_format: 'point', 'bbox', or 'mask'
        
    Returns:
        Structured action object
    """
    operation = jnp.array(dict_action["operation"], dtype=jnp.int32)
    selection = dict_action["selection"]
    
    if action_format == "point":
        if isinstance(selection, (list, tuple)) and len(selection) == 2:
            return PointAction(
                operation=operation,
                row=jnp.array(selection[0], dtype=jnp.int32),
                col=jnp.array(selection[1], dtype=jnp.int32)
            )
        else:
            raise ValueError(f"Point action requires [row, col], got {selection}")
    
    elif action_format == "bbox":
        if isinstance(selection, (list, tuple)) and len(selection) == 4:
            return BboxAction(
                operation=operation,
                r1=jnp.array(selection[0], dtype=jnp.int32),
                c1=jnp.array(selection[1], dtype=jnp.int32),
                r2=jnp.array(selection[2], dtype=jnp.int32),
                c2=jnp.array(selection[3], dtype=jnp.int32)
            )
        else:
            raise ValueError(f"Bbox action requires [r1, c1, r2, c2], got {selection}")
    
    elif action_format == "mask":
        if hasattr(selection, 'shape') and len(selection.shape) == 2:
            return MaskAction(
                operation=operation,
                selection=selection.astype(jnp.bool_)
            )
        else:
            raise ValueError(f"Mask action requires 2D array, got {selection}")
    
    else:
        raise ValueError(f"Unknown action format: {action_format}")

# Usage examples
old_point_action = {"operation": 0, "selection": [5, 7]}
new_point_action = convert_dict_to_structured(old_point_action, "point")

old_bbox_action = {"operation": 1, "selection": [3, 3, 7, 7]}
new_bbox_action = convert_dict_to_structured(old_bbox_action, "bbox")
```

### Batch Conversion Function

Convert lists of dictionary actions to batch structured actions:

```python
def convert_batch_dict_to_structured(dict_actions, action_format="point"):
    """Convert list of dictionary actions to batch structured action.
    
    Args:
        dict_actions: List of dictionary actions
        action_format: 'point', 'bbox', or 'mask'
        
    Returns:
        Batch structured action object
    """
    if not dict_actions:
        raise ValueError("Empty action list")
    
    batch_size = len(dict_actions)
    operations = jnp.array([a["operation"] for a in dict_actions], dtype=jnp.int32)
    
    if action_format == "point":
        rows = jnp.array([a["selection"][0] for a in dict_actions], dtype=jnp.int32)
        cols = jnp.array([a["selection"][1] for a in dict_actions], dtype=jnp.int32)
        
        return PointAction(
            operation=operations,
            row=rows,
            col=cols
        )
    
    elif action_format == "bbox":
        r1s = jnp.array([a["selection"][0] for a in dict_actions], dtype=jnp.int32)
        c1s = jnp.array([a["selection"][1] for a in dict_actions], dtype=jnp.int32)
        r2s = jnp.array([a["selection"][2] for a in dict_actions], dtype=jnp.int32)
        c2s = jnp.array([a["selection"][3] for a in dict_actions], dtype=jnp.int32)
        
        return BboxAction(
            operation=operations,
            r1=r1s, c1=c1s,
            r2=r2s, c2=c2s
        )
    
    elif action_format == "mask":
        selections = jnp.stack([a["selection"] for a in dict_actions])
        
        return MaskAction(
            operation=operations,
            selection=selections.astype(jnp.bool_)
        )
    
    else:
        raise ValueError(f"Unknown action format: {action_format}")

# Usage example
old_batch_actions = [
    {"operation": 0, "selection": [5, 5]},
    {"operation": 0, "selection": [6, 6]},
    {"operation": 0, "selection": [7, 7]}
]

new_batch_actions = convert_batch_dict_to_structured(old_batch_actions, "point")
```

## Common Migration Patterns

### Pattern 1: Random Action Generation

**Before:**
```python
def generate_random_action(key, grid_shape, action_format="point"):
    if action_format == "point":
        row = jax.random.randint(key, (), 0, grid_shape[0])
        col = jax.random.randint(key, (), 0, grid_shape[1])
        return {
            "operation": 0,
            "selection": [int(row), int(col)]
        }
    # ... other formats
```

**After:**
```python
def generate_random_action(key, grid_shape, action_format="point"):
    if action_format == "point":
        key1, key2 = jax.random.split(key)
        row = jax.random.randint(key1, (), 0, grid_shape[0])
        col = jax.random.randint(key2, (), 0, grid_shape[1])
        return PointAction(
            operation=jnp.array(0),
            row=row,
            col=col
        )
    # ... other formats
```

### Pattern 2: Action Validation

**Before:**
```python
def validate_action(action, grid_shape):
    if not isinstance(action, dict):
        return False
    
    if "operation" not in action or "selection" not in action:
        return False
    
    selection = action["selection"]
    if len(selection) == 2:  # Point
        row, col = selection
        return 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]
    
    return False
```

**After:**
```python
def validate_action(action, grid_shape):
    """Validation is now built into structured actions."""
    try:
        validated_action = action.validate(grid_shape)
        return True
    except ValueError:
        return False

# Or use the built-in validation directly
validated_action = action.validate(grid_shape, max_operations=42)
```

### Pattern 3: Action Processing in Loops

**Before:**
```python
def process_episode(initial_state, actions_list, config):
    state = initial_state
    rewards = []
    
    for action_dict in actions_list:
        state, obs, reward, done, info = arc_step(state, action_dict, config)
        rewards.append(reward)
        
        if done:
            break
    
    return rewards
```

**After:**
```python
def process_episode(initial_state, actions_list, config):
    state = initial_state
    rewards = []
    
    for action in actions_list:  # Now structured actions
        state, obs, reward, done, info = arc_step(state, action, config)
        rewards.append(reward)
        
        if done:
            break
    
    return rewards

# Or better yet, use batch processing
def process_episode_batch(initial_states, batch_actions, config):
    states = initial_states
    all_rewards = []
    
    for step_actions in batch_actions:  # Each is a batch structured action
        states, obs, rewards, dones, infos = batch_step(states, step_actions, config)
        all_rewards.append(rewards)
    
    return jnp.stack(all_rewards)  # Shape: (steps, batch_size)
```

### Pattern 4: Policy Network Integration

**Before:**
```python
class PolicyNetwork:
    def __call__(self, state):
        # ... network computation ...
        
        # Output dictionary action
        return {
            "operation": int(operation_logits.argmax()),
            "selection": [int(row_logits.argmax()), int(col_logits.argmax())]
        }
```

**After:**
```python
class PolicyNetwork:
    def __call__(self, state):
        # ... network computation ...
        
        # Output structured action
        return PointAction(
            operation=operation_logits.argmax(),
            row=row_logits.argmax(),
            col=col_logits.argmax()
        )
    
    def sample_action(self, state, key):
        """Sample action from policy distribution."""
        # ... network computation ...
        
        key1, key2, key3 = jax.random.split(key, 3)
        
        operation = jax.random.categorical(key1, operation_logits)
        row = jax.random.categorical(key2, row_logits)
        col = jax.random.categorical(key3, col_logits)
        
        return PointAction(
            operation=operation,
            row=row,
            col=col
        )
```

## Testing Your Migration

### Validation Tests

Create tests to ensure your migration is correct:

```python
def test_action_conversion():
    """Test that converted actions work correctly."""
    # Test point action
    old_action = {"operation": 0, "selection": [5, 7]}
    new_action = convert_dict_to_structured(old_action, "point")
    
    # Verify conversion
    assert new_action.operation == 0
    assert new_action.row == 5
    assert new_action.col == 7
    
    # Test selection mask conversion
    grid_shape = (30, 30)
    mask = new_action.to_selection_mask(grid_shape)
    assert mask[5, 7] == True
    assert jnp.sum(mask) == 1
    
    print("✅ Point action conversion test passed")

def test_batch_conversion():
    """Test batch action conversion."""
    old_batch = [
        {"operation": 0, "selection": [5, 5]},
        {"operation": 1, "selection": [6, 6]},
        {"operation": 0, "selection": [7, 7]}
    ]
    
    new_batch = convert_batch_dict_to_structured(old_batch, "point")
    
    # Verify batch structure
    assert new_batch.operation.shape == (3,)
    assert new_batch.row.shape == (3,)
    assert new_batch.col.shape == (3,)
    
    # Verify values
    assert jnp.array_equal(new_batch.operation, jnp.array([0, 1, 0]))
    assert jnp.array_equal(new_batch.row, jnp.array([5, 6, 7]))
    assert jnp.array_equal(new_batch.col, jnp.array([5, 6, 7]))
    
    print("✅ Batch action conversion test passed")

def test_environment_integration():
    """Test that converted actions work with environment."""
    config = JaxArcConfig(
        environment=EnvironmentConfig(max_episode_steps=100),
        action=UnifiedActionConfig(selection_format="point")
    )
    
    task = create_mock_task()
    key = jax.random.PRNGKey(42)
    
    # Reset environment
    state, obs = arc_reset(key, config, task)
    
    # Test structured action
    action = PointAction(
        operation=jnp.array(0),
        row=jnp.array(5),
        col=jnp.array(7)
    )
    
    # Should work without errors
    new_state, new_obs, reward, done, info = arc_step(state, action, config)
    
    print("✅ Environment integration test passed")

# Run tests
test_action_conversion()
test_batch_conversion()
test_environment_integration()
```

### Performance Comparison

Compare performance before and after migration:

```python
def benchmark_migration_performance():
    """Benchmark performance improvements from migration."""
    import time
    
    config = JaxArcConfig(
        action=UnifiedActionConfig(selection_format="point")
    )
    task = create_mock_task()
    key = jax.random.PRNGKey(42)
    
    # Setup
    state, obs = arc_reset(key, config, task)
    
    # Create structured actions
    structured_actions = [
        PointAction(
            operation=jnp.array(0),
            row=jnp.array(i % 10 + 5),
            col=jnp.array(5)
        )
        for i in range(100)
    ]
    
    # Benchmark structured actions
    start_time = time.perf_counter()
    current_state = state
    
    for action in structured_actions:
        current_state, obs, reward, done, info = arc_step(current_state, action, config)
    
    structured_time = time.perf_counter() - start_time
    
    print(f"Structured actions: {structured_time:.3f}s ({structured_time*10:.1f}ms per step)")
    print(f"Performance: {len(structured_actions)/structured_time:.0f} steps/sec")
    
    # Test batch processing (new capability)
    batch_size = 16
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    batch_states, batch_obs = batch_reset(keys, config, task)
    
    batch_actions = PointAction(
        operation=jnp.zeros(batch_size, dtype=jnp.int32),
        row=jnp.full(batch_size, 5, dtype=jnp.int32),
        col=jnp.full(batch_size, 5, dtype=jnp.int32)
    )
    
    start_time = time.perf_counter()
    for _ in range(100):
        batch_states, batch_obs, rewards, dones, infos = batch_step(
            batch_states, batch_actions, config
        )
    batch_time = time.perf_counter() - start_time
    
    batch_throughput = (100 * batch_size) / batch_time
    
    print(f"Batch processing: {batch_time:.3f}s")
    print(f"Batch throughput: {batch_throughput:.0f} steps/sec")
    print(f"Batch speedup: {batch_throughput / (len(structured_actions)/structured_time):.1f}x")

benchmark_migration_performance()
```

## Troubleshooting

### Common Issues

#### Issue 1: Type Errors

**Problem:**
```python
# ❌ This causes a type error
action = PointAction(
    operation=0,  # Should be jnp.array(0)
    row=5,        # Should be jnp.array(5)
    col=7         # Should be jnp.array(7)
)
```

**Solution:**
```python
# ✅ Use JAX arrays
action = PointAction(
    operation=jnp.array(0),
    row=jnp.array(5),
    col=jnp.array(7)
)
```

#### Issue 2: Batch Dimension Mismatch

**Problem:**
```python
# ❌ Inconsistent batch dimensions
batch_actions = PointAction(
    operation=jnp.zeros(4, dtype=jnp.int32),  # Batch size 4
    row=jnp.array([5, 6, 7]),                 # Batch size 3 - mismatch!
    col=jnp.array([5, 6, 7])                  # Batch size 3
)
```

**Solution:**
```python
# ✅ Consistent batch dimensions
batch_size = 4
batch_actions = PointAction(
    operation=jnp.zeros(batch_size, dtype=jnp.int32),
    row=jnp.array([5, 6, 7, 8]),
    col=jnp.array([5, 6, 7, 8])
)
```

#### Issue 3: Configuration Mismatch

**Problem:**
```python
# ❌ Using bbox actions with point configuration
config = JaxArcConfig(
    action=UnifiedActionConfig(selection_format="point")
)

action = BboxAction(...)  # Wrong action type for config
```

**Solution:**
```python
# ✅ Match action type to configuration
config = JaxArcConfig(
    action=UnifiedActionConfig(selection_format="bbox")
)

action = BboxAction(...)  # Correct action type
```

### Migration Checklist

Use this checklist to ensure complete migration:

- [ ] **Imports Updated**: Added structured action imports
- [ ] **Action Creation**: Replaced all dictionary actions with structured actions
- [ ] **Configuration**: Updated action format in configuration
- [ ] **Batch Processing**: Converted list of actions to batch structured actions
- [ ] **Validation**: Updated action validation logic
- [ ] **Policy Integration**: Updated policy networks to output structured actions
- [ ] **Testing**: Created tests to verify migration correctness
- [ ] **Performance**: Benchmarked performance improvements
- [ ] **Documentation**: Updated code documentation and comments
- [ ] **Error Handling**: Updated error handling for structured actions

### Getting Help

If you encounter issues during migration:

1. **Check Examples**: Look at `examples/advanced/` for working examples
2. **Read Documentation**: Review the structured actions API documentation
3. **Test Incrementally**: Migrate one component at a time
4. **Use Conversion Utilities**: Leverage the provided conversion functions
5. **Validate Results**: Ensure migrated code produces expected results

## Summary

The migration from dictionary actions to structured actions provides significant benefits:

- **Performance**: 10-100x speedup with JAX optimizations
- **Memory**: 99%+ reduction for point/bbox actions
- **Type Safety**: Compile-time error detection
- **Batch Processing**: Efficient parallel environment execution
- **Future-Proof**: Compatible with upcoming JaxARC features

The migration process involves:

1. Updating imports and action creation
2. Converting batch processing patterns
3. Using conversion utilities for existing code
4. Testing and validating the migration
5. Benchmarking performance improvements

With structured actions, your JaxARC code will be faster, more reliable, and ready for advanced RL training scenarios.