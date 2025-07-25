# Advanced Examples

This directory contains sophisticated examples that demonstrate advanced JaxARC
features, performance optimization, and complex workflows.

## Examples

### JAX Integration

- **`jax_callbacks_demo.py`** - Advanced JAX callback system demonstration
  - Shows how to use JAX debug callbacks for visualization
  - Demonstrates performance monitoring during JAX transformations
  - Illustrates batch processing with callbacks
  - Error handling in JAX callback contexts
  - **Run with**: `pixi run python examples/advanced/jax_callbacks_demo.py`

### Performance Optimization

- **`memory_optimization_demo.py`** - Memory management and optimization
  techniques
  - Lazy loading patterns for large datasets
  - Compressed storage for visualization data
  - Garbage collection optimization
  - Memory usage monitoring and cleanup
  - **Run with**:
    `pixi run python examples/advanced/memory_optimization_demo.py`

### Analysis and Replay

- **`replay_analysis_demo.py`** - Episode replay and analysis system
  - Episode recording and structured logging
  - Replay system for debugging and analysis
  - Performance metrics analysis
  - Failure mode analysis and comparison tools
  - **Run with**: `pixi run python examples/advanced/replay_analysis_demo.py`

### Enhanced Environment Features

- **`multi_demonstration_training_example.py`** - Multi-demonstration training
  - Training on multiple demonstration pairs within episodes
  - Dynamic pair switching using non-parametric control operations
  - Episode management with configurable termination criteria
  - Progress tracking across demonstration pairs
  - **Run with**: `pixi run python examples/advanced/multi_demonstration_training_example.py`

- **`test_pair_evaluation_example.py`** - Test pair evaluation mode
  - Evaluation mode without access to target grids
  - Test pair switching and management
  - Alternative reward signals during evaluation
  - Proper separation of training and evaluation phases
  - **Run with**: `pixi run python examples/advanced/test_pair_evaluation_example.py`

- **`action_history_analysis_example.py`** - Action history tracking and analysis
  - Comprehensive action sequence recording with metadata
  - Memory-efficient history configuration options
  - Action history analysis and pattern detection
  - History-based replay and debugging capabilities
  - **Run with**: `pixi run python examples/advanced/action_history_analysis_example.py`

- **`restricted_action_space_example.py`** - Dynamic action space control
  - Dynamic action filtering based on context
  - Custom operation restrictions for focused training
  - Action validation and error handling policies
  - Context-dependent operation availability
  - **Run with**: `pixi run python examples/advanced/restricted_action_space_example.py`

## Prerequisites

Before exploring these examples, ensure you understand:

- Basic JaxARC configuration (see `examples/basic/`)
- JAX fundamentals (transformations, pure functions)
- Python memory management concepts

## Key Concepts Demonstrated

### JAX Integration

- **Debug Callbacks**: Using `jax.debug.callback` for visualization during
  transformations
- **Performance Monitoring**: Tracking callback performance and errors
- **Batch Processing**: Efficient processing of multiple items with JAX
- **Error Resilience**: Handling errors in JAX callback contexts

### Memory Management

- **Lazy Loading**: Loading data only when needed with automatic caching
- **Compressed Storage**: Efficient storage of large visualization datasets
- **Memory Monitoring**: Real-time memory usage tracking and reporting
- **Garbage Collection**: Optimized GC settings for visualization workloads

### Analysis Tools

- **Structured Logging**: Comprehensive episode and step logging
- **Replay System**: Deterministic episode replay for debugging
- **Performance Analysis**: Success rates, failure modes, and comparisons
- **Visualization Generation**: Automated visualization creation during replay

### Enhanced Environment Capabilities

- **Multi-Demonstration Training**: Training on all available demonstration pairs
- **Test Pair Evaluation**: Proper evaluation mode without target access
- **Action History Tracking**: Complete action sequence recording and analysis
- **Dynamic Action Space**: Context-aware operation filtering and validation
- **Episode Management**: Sophisticated episode lifecycle control
- **Progressive Learning**: Gradual action space expansion for curriculum learning

## Performance Considerations

These examples demonstrate production-ready patterns for:

- **Memory Efficiency**: Techniques for handling large datasets
- **JAX Optimization**: Proper use of JAX transformations with side effects
- **Storage Management**: Efficient data persistence and retrieval
- **Analysis Scalability**: Tools that work with large numbers of episodes

## Integration with Basic Examples

These advanced examples build upon concepts from basic examples:

- Use the unified configuration system from `config_factory_demo.py`
- Extend environment testing patterns from `test_config_environments.py`
- Add sophisticated monitoring and analysis capabilities

## Next Steps

After mastering these advanced examples:

- Explore integration examples for external tool connectivity
- Apply these patterns to your own research or production workflows
- Contribute improvements or additional advanced examples
