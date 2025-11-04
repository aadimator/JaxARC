# Configuration API

JaxARC uses a comprehensive configuration system based on Hydra and Equinox.

Configuration in JaxARC is handled through the `JaxArcConfig` class, which
provides:

- Type-safe configuration with Equinox modules
- Hydra integration for YAML-based configs
- Presets for common use cases
- Runtime validation

## Module Contents

```{eval-rst}
.. automodule:: jaxarc.configs
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### From Python

```python
from jaxarc import JaxArcConfig, make

# Default configuration
config = JaxArcConfig()
env, env_params = make("Mini", config=config)

# Custom configuration
config = JaxArcConfig(
    grid_size=32,
    max_episode_steps=1000,
)
env, env_params = make("Mini", config=config)
```

### From Hydra Config

```python
from jaxarc.utils.core import get_config

# Load from YAML with overrides
hydra_config = get_config(
    overrides=["dataset=mini_arc", "action=point", "grid_size=32"]
)

config = JaxArcConfig.from_hydra(hydra_config)
```

### From YAML File

```yaml
# config.yaml
dataset: mini_arc
action: point
grid_size: 32
max_episode_steps: 1000
```

```python
from hydra import compose, initialize
from jaxarc import JaxArcConfig

with initialize(config_path=".", version_base=None):
    cfg = compose(config_name="config")
    config = JaxArcConfig.from_hydra(cfg)
```
