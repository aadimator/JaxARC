# Basic Examples

This directory contains fundamental examples that demonstrate core JaxARC concepts and are ideal for getting started.

## Examples

### Configuration System
- **`config_factory_demo.py`** - Comprehensive demonstration of the unified configuration system
  - Shows how to use ConfigFactory for creating configurations
  - Demonstrates preset system and configuration validation
  - Illustrates the elimination of dual configuration patterns
  - **Run with**: `pixi run python examples/basic/config_factory_demo.py`

### Environment Testing
- **`test_config_environments.py`** - Tests different environment configurations
  - Validates various environment setups (raw, standard, full)
  - Tests different action formats (mask, point, bbox)
  - Demonstrates configuration validation
  - **Run with**: `pixi run python examples/basic/test_config_environments.py`

## Getting Started

If you're new to JaxARC, start with these examples in order:

1. **config_factory_demo.py** - Learn the configuration system
2. **test_config_environments.py** - Understand environment setup

## Key Concepts Demonstrated

- **Unified Configuration System**: Single JaxArcConfig instead of dual configs
- **Configuration Factory**: Easy creation of common configurations
- **Preset System**: Pre-built configurations for common use cases
- **Environment Setup**: Basic environment initialization and testing
- **Action Formats**: Different ways to specify actions (mask, point, bbox)
- **Configuration Validation**: Ensuring configurations are valid

## Next Steps

After mastering these basic examples, explore:
- **Advanced Examples** (`examples/advanced/`) for performance optimization and complex features
- **Integration Examples** (`examples/integration/`) for external tool integrations