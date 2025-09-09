---
applyTo: "**"
---

# Development Guidelines

## Code Quality Principles

### Simplicity First

- Keep implementations as simple as needed - avoid overengineering
- Prefer clear, readable code over clever optimizations unless performance is
  critical
- Use straightforward solutions that are easy to understand and maintain
- This is a research project, not a production intended application, so
  simplicity is preferred

### JAX Compliance Requirements

- **Pure Functions**: All core functionality must be JAX-compliant with pure
  functions
- **JIT Compatibility**: Ensure functions work with `jax.jit`, `jax.vmap`, and
  `jax.pmap`
- **Immutable State**: Use `equinox.Module` for state management with automatic
  PyTree registration
- **Static Shapes**: Maintain static array shapes using padding and masks
- **PRNG Management**: Use explicit PRNG key management with `jax.random.split`
- **Type Safety**: Use `jaxtyping` annotations for precise array shape
  documentation

### Implementation Standards

- Implement required functionality completely - no shortcuts or partial
  implementations
- Follow the single responsibility principle for functions and classes
- Use type hints consistently throughout the codebase

## Development Workflow

### Testing and Validation

- Write temporary test scripts instead of using `pixi run python -c "code"` for
  validation
- Create simple `test_*.py` files to verify functionality works as expected
- Run tests with `pixi run python test_script.py` to confirm behavior
- Delete temporary test files once functionality is confirmed working
- Use the comprehensive test suite in `test/` for permanent validation

### Code Evolution Policy

- **No Backwards Compatibility**: Replace old implementations when introducing
  new features
- **Single Source of Truth**: Maintain only one correct way to accomplish each
  task
- **Clean Migration**: When replacing functionality, update all usage sites and
  remove old code
- **Consistent API**: Ensure new implementations follow established patterns and
  conventions

## Architecture Patterns

### Configuration Management

- Use `equinox.Module` for configuration structures with automatic validation
- Leverage Hydra for complex parameter hierarchies
- Use `JaxArcConfig` as the unified entry point for all configuration
- Validate configuration parameters at creation time with cross-config
  consistency checks

### Functional Design

- Prefer pure functions that return new state rather than mutating existing
  state
- Pass dependencies explicitly rather than relying on global state
- Design functions to be composable and easily testable
- Use functional patterns that work well with JAX transformations

### Error Handling

- Use `equinox` patterns and `chex` assertions for runtime validation in
  JAX-compatible code
- Provide clear error messages that help developers understand issues
- Validate inputs early and fail fast with descriptive errors
- Handle edge cases explicitly rather than relying on implicit behavior
- Use `__check_init__` methods in Equinox modules for automatic validation
