# Migration Guide: Class-Based to Config-Based API

This guide helps you migrate from JaxARC's old class-based API to the new
config-based functional API. The new system provides better JAX compatibility,
type safety, and easier configuration management.

## 🔄 Overview of Changes

### What's New

- **Functional API**: Pure functions (`arc_reset`, `arc_step`) instead of class
  methods
- **Typed Configurations**: Frozen dataclasses with validation
- **Factory Functions**: Easy configuration creation with presets
- **Hydra Integration**: Seamless configuration management
- **Better JAX Compatibility**: Improved JIT compilation and transformations

### What's Deprecated (but still works)

- **Class-based API**: `ArcEnvironment` class methods
- **Manual configuration**: Direct dictionary-based configs
- **Separate configs**: Environment and dataset configs passed separately

## 📋 Migration Checklist

- [ ] Replace `ArcEnvironment` instantiation with factory functions
- [ ] Replace `env.reset()` with `arc_reset()`
- [ ] Replace `env.step()` with `arc_step()`
- [ ] Update configuration creation to use factory functions
- [ ] Update JAX transformations to use static configs
- [ ] Update tests to use functional API
- [ ] Update Hydra configurations if needed

## 🔄 Side-by-Side Comparisons

### Basic Environment Usage

#### Old Way (Class-Based)

```python
from jaxarc.envs import ArcEnvironment

# Create environment with separate configs
env_config = {
    "max_episode_steps": 100,
    "log_operations": True,
    "reward": {
        "success_bonus": 10.0,
        "step_penalty": -0.01,
    },
}

dataset_config = {
    "max_grid_height": 30,
    "max_grid_width": 30,
    "max_colors": 10,
}

# Instantiate environment
env = ArcEnvironment(env_config, dataset_config)

# Reset and step
key = jax.random.PRNGKey(42)
state, obs = env.reset(key)
state, obs, reward, done, info = env.step(state, action)
```

#### New Way (Config-Based)

```python
from jaxarc.envs import arc_reset, arc_step, create_standard_config

# Create configuration using factory function
config = create_standard_config(
    max_episode_steps=100, success_bonus=10.0, step_penalty=-0.01, log_operations=True
)

# Use functional API
key = jax.random.PRNGKey(42)
state, obs = arc_reset(key, config)
state, obs, reward, done, info = arc_step(state, action, config)
```

### JAX Transformations

#### Old Way

```python
# JIT compilation was limited
@jax.jit
def step_fn(state, action):
    # env had to be defined outside
    return env.step(state, action)
```

#### New Way

```python
# Clean JIT compilation with static config
@jax.jit
def step_fn(state, action, config):
    return arc_step(state, action, config)


# Or with static_argnums
jitted_step = jax.jit(arc_step, static_argnums=(2,))
```

### Custom Configuration

#### Old Way

```python
# Manual dictionary creation
config = {
    "max_episode_steps": 150,
    "reward": {
        "reward_on_submit_only": True,
        "success_bonus": 15.0,
        "similarity_weight": 2.0,
    },
    "grid": {
        "max_grid_height": 25,
        "max_grid_width": 25,
    },
    "action": {
        "selection_format": "point",
        "num_operations": 20,
    },
}
```

#### New Way

```python
from jaxarc.envs import ArcEnvConfig, RewardConfig, GridConfig, ActionConfig

# Typed configuration with validation
config = ArcEnvConfig(
    max_episode_steps=150,
    reward=RewardConfig(
        reward_on_submit_only=True,
        success_bonus=15.0,
        similarity_weight=2.0,
    ),
    grid=GridConfig(
        max_grid_height=25,
        max_grid_width=25,
    ),
    action=ActionConfig(
        selection_format="point",
        num_operations=20,
    ),
)
```

### Hydra Integration

#### Old Way

```python
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    env = ArcEnvironment(cfg.environment, cfg.dataset)
    # ... rest of code
```

#### New Way

```python
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Direct usage with Hydra config
    state, obs = arc_reset(key, cfg.environment)
    # ... rest of code
```

## 🚀 Step-by-Step Migration

### Step 1: Update Imports

```python
# Old imports
from jaxarc.envs import ArcEnvironment

# New imports
from jaxarc.envs import (
    arc_reset,
    arc_step,
    create_standard_config,
    ArcEnvConfig,
    RewardConfig,
    GridConfig,
    ActionConfig,
)
```

### Step 2: Replace Environment Creation

```python
# Old: Class instantiation
env = ArcEnvironment(env_config, dataset_config)

# New: Factory function
config = create_standard_config(**your_parameters)
```

### Step 3: Replace Reset Calls

```python
# Old
state, obs = env.reset(key)

# New
state, obs = arc_reset(key, config)
```

### Step 4: Replace Step Calls

```python
# Old
state, obs, reward, done, info = env.step(state, action)

# New
state, obs, reward, done, info = arc_step(state, action, config)
```

### Step 5: Update JAX Transformations

```python
# Old: Limited JIT support
@jax.jit
def training_step(state, action):
    return env.step(state, action)


# New: Full JIT support
@jax.jit
def training_step(state, action, config):
    return arc_step(state, action, config)
```

## 🔧 Common Migration Patterns

### Pattern 1: Simple Environment Setup

```python
# OLD
env = ArcEnvironment(
    {"max_episode_steps": 100}, {"max_grid_height": 30, "max_grid_width": 30}
)

# NEW
config = create_standard_config(max_episode_steps=100)
```

### Pattern 2: Custom Reward Configuration

```python
# OLD
env_config = {
    "reward": {
        "success_bonus": 20.0,
        "step_penalty": -0.02,
        "reward_on_submit_only": True,
    }
}

# NEW
config = create_standard_config(
    success_bonus=20.0,
    step_penalty=-0.02,
    reward_on_submit_only=True,
)
```

### Pattern 3: Different Action Formats

```python
# OLD
env_config = {"action": {"selection_format": "point"}}

# NEW
config = create_point_config()
```

### Pattern 4: Training Loop

```python
# OLD
def training_loop():
    env = ArcEnvironment(env_config, dataset_config)
    for episode in range(num_episodes):
        state, obs = env.reset(key)
        while not done:
            action = policy(obs)
            state, obs, reward, done, info = env.step(state, action)


# NEW
def training_loop():
    config = create_standard_config()
    for episode in range(num_episodes):
        state, obs = arc_reset(key, config)
        while not done:
            action = policy(obs)
            state, obs, reward, done, info = arc_step(state, action, config)
```

## 🎯 Environment Type Migration

### Choosing the Right Configuration

| Old Configuration  | New Factory Function       | Use Case                |
| ------------------ | -------------------------- | ----------------------- |
| Minimal operations | `create_raw_config()`      | Basic testing           |
| Standard setup     | `create_standard_config()` | Regular training        |
| All operations     | `create_full_config()`     | Advanced research       |
| Point actions      | `create_point_config()`    | Single-point operations |
| Box actions        | `create_bbox_config()`     | Rectangular selections  |

### Configuration Mapping

```python
# OLD: Manual operation selection
env_config = {
    "allowed_operations": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 34],  # Colors + submit
    "max_episode_steps": 50,
}

# NEW: Use raw config
config = create_raw_config(max_episode_steps=50)
```

## 🧪 Testing Migration

### Update Test Files

```python
# OLD
def test_environment():
    env = ArcEnvironment(env_config, dataset_config)
    state, obs = env.reset(key)
    assert state.working_grid.shape == (30, 30)


# NEW
def test_environment():
    config = create_standard_config()
    state, obs = arc_reset(key, config)
    assert state.working_grid.shape == (30, 30)
```

### Verify Behavior

```python
# Create test to ensure identical behavior
def test_migration_compatibility():
    # Old way
    old_env = ArcEnvironment(env_config, dataset_config)
    old_state, old_obs = old_env.reset(key)

    # New way
    new_config = create_standard_config()
    new_state, new_obs = arc_reset(key, new_config)

    # Verify identical results
    assert jnp.array_equal(old_state.working_grid, new_state.working_grid)
    assert jnp.array_equal(old_obs, new_obs)
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Issue: "Config is not hashable for JIT"

```python
# Problem: Using mutable config
config = {"max_episode_steps": 100}  # Dict is not hashable

# Solution: Use typed config
config = create_standard_config(max_episode_steps=100)  # Frozen dataclass
```

#### Issue: "Selection shape mismatch"

```python
# Problem: Wrong selection shape
action = {"selection": jnp.ones((10, 10)), "operation": 1}

# Solution: Match grid shape
action = {
    "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
    "operation": 1,
}
```

#### Issue: "Invalid operation ID"

```python
# Problem: Operation ID out of range
action = {"selection": selection, "operation": 50}  # > 34

# Solution: Use valid operation ID
action = {"selection": selection, "operation": 1}  # 0-34 range
```

#### Issue: "Missing parser argument"

```python
# Problem: No parser specified
state, obs = arc_reset(key, config)  # No task data

# Solution: Provide parser or use demo task
from jaxarc.parsers import ArcAgiParser

parser = ArcAgiParser(dataset_config)
state, obs = arc_reset(key, config, parser=parser)
```

### Debug Tips

1. **Enable logging**: Set `log_operations=True` in config
2. **Check config**: Use `validate_config(config)` to catch issues early
3. **Inspect info**: The `info` dict contains detailed debug information
4. **Use type hints**: Enable static type checking in your IDE

## 📚 Advanced Migration Topics

### Batch Processing Migration

```python
# OLD: Limited batch support
def process_batch(keys, actions):
    results = []
    for key, action in zip(keys, actions):
        state, obs = env.reset(key)
        result = env.step(state, action)
        results.append(result)
    return results


# NEW: Native vmap support
def process_batch(keys, actions):
    def single_episode(key, action):
        state, obs = arc_reset(key, config)
        return arc_step(state, action, config)

    return jax.vmap(single_episode)(keys, actions)
```

### Custom Configuration Classes

```python
# Create your own configuration
@dataclass(frozen=True)
class MyCustomConfig:
    my_param: int = 42

    def to_arc_config(self) -> ArcEnvConfig:
        return create_standard_config(
            max_episode_steps=self.my_param * 2,
            success_bonus=self.my_param / 2,
        )
```

## ✅ Migration Verification

### Checklist for Complete Migration

- [ ] All `ArcEnvironment` instantiations replaced with factory functions
- [ ] All `env.reset()` calls replaced with `arc_reset()`
- [ ] All `env.step()` calls replaced with `arc_step()`
- [ ] JAX transformations updated to use static configs
- [ ] Tests updated and passing
- [ ] Configuration files updated if using Hydra
- [ ] Documentation updated to reflect new API
- [ ] Performance benchmarks maintained or improved

### Validation Script

```python
def validate_migration():
    """Script to validate successful migration."""

    # Test basic functionality
    config = create_standard_config()
    key = jax.random.PRNGKey(42)
    state, obs = arc_reset(key, config)

    action = {
        "selection": jnp.ones_like(state.working_grid, dtype=jnp.bool_),
        "operation": jnp.array(1),
    }

    state, obs, reward, done, info = arc_step(state, action, config)

    # Test JAX compatibility
    @jax.jit
    def test_jit(state, action, config):
        return arc_step(state, action, config)

    result = test_jit(state, action, config)

    print("✅ Migration validation successful!")
    return True
```

## 🎓 Best Practices After Migration

1. **Use factory functions**: Prefer `create_standard_config()` over manual
   creation
2. **Validate configurations**: Use `validate_config()` in development
3. **Static configs for JIT**: Mark configs as static for JAX transformations
4. **Type hints**: Use proper type hints for better IDE support
5. **Error handling**: Handle configuration errors gracefully
6. **Performance monitoring**: Benchmark before and after migration

## 🔗 Additional Resources

- [Config API Documentation](CONFIG_API_README.md)
- [Architecture Overview](../planning-docs/PROJECT_ARCHITECTURE.md)
- [Examples Directory](../examples/)
- [Test Suite](../tests/envs/test_config_api.py)

## 🤝 Getting Help

If you encounter issues during migration:

1. Check the [troubleshooting section](#troubleshooting) above
2. Review the [examples](../examples/) for reference implementations
3. Open an issue on [GitHub](https://github.com/aadimator/JaxARC/issues)
4. Start a discussion on
   [GitHub Discussions](https://github.com/aadimator/JaxARC/discussions)

Remember: Both APIs coexist, so you can migrate gradually without breaking
existing code!
