# JAX Compatibility Troubleshooting Guide

This guide helps resolve common JAX compatibility issues in JaxARC, including JIT compilation errors, memory problems, and performance issues.

## Common Issues and Solutions

### 1. JIT Compilation Errors

#### Issue: `TypeError: unhashable type`

**Problem**: Configuration objects contain unhashable types
```python
# ❌ This causes the error
@jax.jit(static_argnames=['config'])
def my_function(state, action, config):
    return arc_step(state, action, config)
```

**Solution**: Use `equinox.filter_jit` instead
```python
# ✅ This works correctly
@eqx.filter_jit
def my_function(state, action, config):
    return arc_step(state, action, config)
```

#### Issue: Abstract array interpretation errors

**Problem**: Dynamic shapes or control flow in JIT functions
**Solution**: Use static shapes with padding and masks

### 2. Memory Issues

#### Issue: Out of memory with batch processing

**Problem**: Using mask actions with large batch sizes
**Solution**: Use point or bbox actions for memory efficiency
```python
# ✅ Memory efficient
config = JaxArcConfig(
    action=UnifiedActionConfig(selection_format="point")  # 99% less memory
)
```

### 3. Performance Issues

#### Issue: Slow performance despite JIT compilation

**Problem**: Functions being recompiled repeatedly
**Solution**: Warm up functions and use consistent signatures

### 4. Action System Issues

#### Issue: Dictionary actions no longer work

**Problem**: Using deprecated dictionary action format
**Solution**: Migrate to structured actions
```python
# ❌ Old format (deprecated)
action = {"operation": 0, "selection": [5, 7]}

# ✅ New format
action = PointAction(operation=jnp.array(0), row=jnp.array(5), col=jnp.array(7))
```

## Debugging Tools

### Configuration Validation
```python
def validate_config(config):
    try:
        hash(config)
        print("✅ Configuration is hashable")
    except TypeError as e:
        print(f"❌ Configuration error: {e}")
```

### Performance Monitoring
```python
def monitor_performance(func, *args):
    import time
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    print(f"Function took {elapsed*1000:.2f}ms")
    return result
```

## Environment Variables

Configure error handling behavior:
```python
import os
os.environ['EQX_ON_ERROR'] = 'raise'      # Default: raise errors
# os.environ['EQX_ON_ERROR'] = 'nan'      # Return NaN on errors
# os.environ['EQX_ON_ERROR'] = 'breakpoint'  # Debug on errors
```

## Getting Help

1. Check the examples in `examples/advanced/`
2. Review the JAX optimization guide
3. Use the performance profiling tools
4. Test with minimal examples first

For detailed solutions, see the full documentation guides.