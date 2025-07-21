# Integration Examples

This directory contains examples that demonstrate how to integrate JaxARC with
external tools and services.

## Examples

### Weights & Biases (WandB) Integration

#### Core Integration

- **`wandb_integration_demo.py`** - Basic WandB integration demonstration
  - Shows fundamental WandB setup and configuration
  - Demonstrates step and episode logging
  - Basic image and metric logging
  - **Run with**:
    `pixi run python examples/integration/wandb_integration_demo.py`

#### Enhanced Features

- **`enhanced_wandb_demo.py`** - Advanced WandB features and optimization
  - Image optimization with resizing and format conversion
  - Automatic experiment tagging based on configuration
  - Intelligent run organization with groups and job types
  - Research and development configuration presets
  - **Run with**: `pixi run python examples/integration/enhanced_wandb_demo.py`

#### Error Handling and Offline Support

- **`wandb_error_handling_demo.py`** - Robust WandB integration with error
  handling
  - Automatic offline mode switching on network errors
  - Offline data caching with compression
  - Automatic sync when connectivity is restored
  - Cache size management and cleanup
  - Retry logic with exponential backoff
  - **Run with**:
    `pixi run python examples/integration/wandb_error_handling_demo.py`

## Prerequisites

Before using these integration examples:

- Install required dependencies: `pixi install`
- For WandB examples: Set up a WandB account (optional for offline mode)
- Understand basic JaxARC configuration (see `examples/basic/`)

## Key Integration Patterns

### WandB Integration Architecture

- **Configuration-Driven**: WandB settings managed through JaxARC configuration
- **Automatic Tagging**: Intelligent experiment organization based on config
- **Offline-First**: Robust offline support with automatic sync
- **Error Resilience**: Graceful handling of network issues and API errors

### Common Integration Features

- **Structured Logging**: Consistent logging patterns across integrations
- **Configuration Management**: Integration settings via unified config system
- **Error Handling**: Robust error recovery and fallback mechanisms
- **Performance Optimization**: Efficient data transfer and caching

## WandB Integration Details

### Configuration Options

```python
# Basic WandB configuration
wandb_config = {
    "enabled": True,
    "project_name": "my-jaxarc-project",
    "offline_mode": False,
    "log_frequency": 10,
}

# Advanced configuration with error handling
advanced_config = {
    "auto_offline_on_error": True,
    "retry_attempts": 3,
    "cache_size_limit_gb": 1.0,
    "sync_batch_size": 50,
}
```

### Logging Capabilities

- **Step Metrics**: Reward, similarity, action details
- **Episode Summaries**: Success/failure, total rewards, step counts
- **Images**: Grid visualizations with automatic optimization
- **Configuration**: Automatic config logging and versioning
- **System Metrics**: Memory usage, performance statistics

### Offline Support

- **Automatic Detection**: Network connectivity monitoring
- **Local Caching**: Compressed storage of offline data
- **Sync Management**: Intelligent sync when connectivity returns
- **Cache Cleanup**: Automatic cleanup of old cached data

## Usage Patterns

### Development Workflow

1. Start with `wandb_integration_demo.py` for basic setup
2. Use `enhanced_wandb_demo.py` for production features
3. Implement `wandb_error_handling_demo.py` patterns for robustness

### Production Deployment

- Enable offline mode for unreliable networks
- Configure appropriate cache limits
- Set up automatic sync schedules
- Monitor cache usage and cleanup

## Extending Integrations

To add new integrations:

1. Follow the configuration-driven pattern
2. Implement offline/error handling capabilities
3. Add comprehensive logging and monitoring
4. Create example demonstrating key features
5. Document configuration options and usage patterns

## Troubleshooting

### WandB Issues

- **Network Errors**: Examples automatically handle offline mode
- **API Limits**: Built-in retry logic with exponential backoff
- **Cache Issues**: Utilities for cache inspection and cleanup
- **Sync Problems**: Detailed logging for debugging sync issues

### General Integration Issues

- Check configuration validation in basic examples
- Verify dependencies are installed correctly
- Review logs for detailed error information
- Use offline modes when available for testing
