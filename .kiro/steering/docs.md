---
inclusion: always
---

# Documentation Guidelines

## MyST Markdown Standards

- **MyST-MD**: All documentation uses MyST Markdown with Jupyter Book for
  rendering
- **Configuration**: Documentation structure defined in `docs/myst.yml`
- **Build System**: Use `pixi run docs-serve` for local development
- **Structure**: Follow the established TOC hierarchy (index → getting-started →
  datasets → configuration → api_reference → examples)

## Documentation Conventions

### File Organization

- **Core docs**: Place in `docs/` root (getting-started.md, configuration.md,
  etc.)
- **Examples**: Organize in `docs/examples/` with descriptive names
- **API Reference**: Maintain comprehensive API docs with code examples
- **Planning docs**: Keep architectural docs in `planning-docs/` separate from
  user docs

### Writing Style

- **Code Examples**: Always include working, runnable code snippets
- **JAX Focus**: Emphasize JAX-native patterns and functional programming
- **Type Hints**: Show complete type annotations in examples
- **Configuration**: Document Hydra config patterns and factory functions
- **Environment Setup**: Always show `pixi run python` commands for consistency

### Content Standards

- **Headers**: Use descriptive, hierarchical headers (# → ## → ###)
- **Code Blocks**: Specify language for syntax highlighting
- **Cross-references**: Link between related documentation sections
- **Examples**: Provide both basic and advanced usage patterns
- **Validation**: Include `chex` assertions and error handling examples

### MyST-Specific Features

- **Directives**: Use MyST directives for callouts, warnings, and code execution
- **Cross-references**: Leverage MyST's cross-referencing capabilities
- **Math**: Use MyST math rendering for algorithms and formulas
- **Notebooks**: Integrate Jupyter notebooks seamlessly with MyST

## API Documentation Requirements

### Function Documentation

````python
def reset(env_params: EnvParams, key: PRNGKey) -> TimeStep:
    """Reset the ARC environment to initial state.

    Args:
        env_params: Environment parameters with task buffer and configuration
        key: JAX PRNG key for reproducible randomization

    Returns:
        TimeStep object containing embedded state and observation

    Example:
        ```python
        import jax
        from jaxarc import JaxArcConfig
        from jaxarc.registration import make, available_task_ids

        config = JaxArcConfig()
        available_ids = available_task_ids("Mini", config=config, auto_download=True)
        task_id = available_ids[0]
        env, env_params = make(f"Mini-{task_id}", config=config)

        key = jax.random.PRNGKey(42)
        timestep = env.reset(env_params, key)
        ```
    """
````

### Class Documentation

- **Purpose**: Clear description of class role in the system
- **Configuration**: Document all config parameters with types
- **Examples**: Show initialization and common usage patterns
- **JAX Compatibility**: Note JIT/vmap/pmap compatibility where relevant

## Build and Deployment

### Local Development

```bash
pixi run docs-serve    # Serve docs locally with hot reload
```

### Content Validation

- **Links**: Verify all internal and external links work
- **Code**: Test all code examples for correctness
- **Structure**: Ensure TOC matches actual file organization
- **Formatting**: Validate MyST syntax and rendering

### Publishing Standards

- **Completeness**: All public APIs must have documentation
- **Examples**: Every major feature needs working examples
- **Consistency**: Maintain consistent style across all docs
- **Accuracy**: Keep docs synchronized with code changes
