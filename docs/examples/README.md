# JaxARC Examples

This directory contains practical examples and usage patterns for JaxARC,
organized by use case and complexity level.

## Example Categories

### [Basic Usage](basic-usage.md)

Core functionality examples covering:

- Environment setup and configuration
- Task loading and parsing
- Basic action patterns
- JAX transformations and JIT compilation

### [ConceptARC Examples](conceptarc-examples.md)

ConceptARC-specific patterns including:

- Concept group exploration
- Systematic evaluation workflows
- Concept-specific task selection
- Performance analysis across concepts

### [MiniARC Examples](miniarc-examples.md)

MiniARC rapid prototyping examples:

- Fast iteration workflows
- Performance comparisons
- Batch processing patterns
- Development and testing utilities

### [Advanced Patterns](advanced-patterns.md)

Advanced usage patterns featuring:

- JAX transformations (vmap, pmap)
- Custom configuration patterns
- Batch processing and parallel execution
- Integration with training frameworks

## Running Examples

All examples can be run from the project root:

```bash
# Activate the environment
pixi shell

# Run example scripts
pixi run python examples/config_api_demo.py
pixi run python examples/conceptarc_usage_example.py
pixi run python examples/miniarc_usage_example.py
pixi run python examples/visualization_demo.py
```

## Code Standards

All examples follow these standards:

- Complete, runnable code snippets
- Clear comments explaining each step
- Error handling and validation
- Performance considerations noted
- Links to relevant documentation sections
