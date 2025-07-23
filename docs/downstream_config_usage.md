# Using JaxARC Configurations in Downstream Packages

This guide demonstrates how downstream packages can use and extend JaxARC's Hydra configurations.

## Installation

Install JaxARC with config support:

```bash
pip install jaxarc
```

## Basic Usage

### 1. Load JaxARC Base Configurations

```python
from jaxarc import load_jaxarc_config

# Load default JaxARC configuration
cfg = load_jaxarc_config()

# Load with overrides
cfg = load_jaxarc_config(
    overrides=[
        "dataset=concept_arc",
        "environment.max_episode_steps=50",
        "visualization.enabled=true"
    ]
)
```

### 2. Explore Available Configurations

```python
from jaxarc import list_available_configs

# See all available configuration groups
configs = list_available_configs()
print("Available datasets:", configs["dataset"])
print("Available actions:", configs["action"])
print("Available environments:", configs["environment"])

# Example output:
# Available datasets: ['arc_agi_1', 'arc_agi_2', 'concept_arc', 'mini_arc']
# Available actions: ['full', 'raw', 'standard']
# Available environments: ['evaluation', 'training']
```

### 3. Get JaxARC Config Directory

```python
from jaxarc import get_jaxarc_config_dir
from pathlib import Path

# Get path to JaxARC's bundled configs
config_dir = get_jaxarc_config_dir()
print(f"JaxARC configs located at: {config_dir}")

# You can copy/reference these configs
dataset_configs = config_dir / "dataset"
for config_file in dataset_configs.glob("*.yaml"):
    print(f"Dataset config: {config_file.name}")
```

## Advanced Usage

### 1. Extend JaxARC Configurations

Create your own project with custom configs that build on JaxARC:

```python
from jaxarc import extend_jaxarc_config
from pathlib import Path

# Directory structure:
# my_project/
#   conf/
#     config.yaml          # Your main config
#     my_custom_group/
#       special.yaml       # Custom config group
#     overrides/
#       research.yaml      # Environment-specific overrides

# Extend JaxARC config with your custom settings
cfg = extend_jaxarc_config(
    base_config_name="config",  # JaxARC base config
    custom_config_dir=Path("my_project/conf"),
    overrides=["my_custom_group=special", "++experiment_name=test_run"],
    merge_strategy="custom_overrides_jaxarc"  # Your configs take precedence
)
```

### 2. Create Configuration Templates

Set up a template structure for a new project:

```python
from jaxarc import create_config_template
from pathlib import Path

# Creates a template config structure
create_config_template(
    output_dir=Path("my_new_project/conf"),
    include_examples=True
)

# This creates:
# my_new_project/conf/
#   config.yaml          # Template main config
#   examples/
#     custom_dataset.yaml
#     custom_environment.yaml
#   README.md           # Usage instructions
```

### 3. Use in Hydra Applications

Integrate with your own Hydra app:

```python
# my_project/main.py
import hydra
from omegaconf import DictConfig
from jaxarc import ArcEnvironment
from jaxarc.envs.factory import ConfigFactory

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Convert to JaxArcConfig - works with extended configs
    jaxarc_config = ConfigFactory.from_hydra(cfg)
    
    # Create environment
    env = ArcEnvironment(jaxarc_config)
    
    # Your training/evaluation code here
    print(f"Using dataset: {jaxarc_config.dataset.dataset_name}")
    print(f"Max episode steps: {jaxarc_config.environment.max_episode_steps}")

if __name__ == "__main__":
    main()
```

## Configuration File Examples

### Your Project's Main Config

```yaml
# my_project/conf/config.yaml
defaults:
  - _self_
  - base_jaxarc: config  # Use JaxARC as base
  - my_custom_group: default
  - override dataset: concept_arc  # Override JaxARC's dataset choice

# Your custom settings
experiment_name: "my_experiment"
save_interval: 100

# Override JaxARC settings
environment:
  max_episode_steps: 200  # Override default

# Add new settings that don't exist in JaxARC
my_custom_settings:
  learning_rate: 0.001
  batch_size: 32
```

### Custom Configuration Group

```yaml
# my_project/conf/my_custom_group/default.yaml
# Custom settings specific to your project
model:
  hidden_size: 256
  num_layers: 3

training:
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 100

logging:
  log_every: 10
  save_checkpoints: true
```

## Integration with Existing Hydra Projects

If you already have a Hydra project, you can reference JaxARC configs:

```yaml
# your_existing_project/conf/config.yaml
defaults:
  - _self_
  - your_existing_groups...
  
# Load JaxARC config as a group
jaxarc: ${oc.create:${jaxarc:load_jaxarc_config}}

# Or specific overrides
jaxarc_dataset: ${oc.create:${jaxarc:load_jaxarc_config:dataset=concept_arc}}
```

Then in Python:

```python
from jaxarc import load_jaxarc_config

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Access your settings
    your_settings = cfg.your_group
    
    # Get JaxARC config
    if hasattr(cfg, 'jaxarc'):
        jaxarc_cfg = cfg.jaxarc
    else:
        # Load manually with your preferred overrides
        jaxarc_cfg = load_jaxarc_config(
            overrides=["dataset=concept_arc"]
        )
    
    # Convert and use
    jaxarc_config = ConfigFactory.from_hydra(jaxarc_cfg)
    env = ArcEnvironment(jaxarc_config)
```

## Tips and Best Practices

1. **Start with Templates**: Use `create_config_template()` to get started quickly.

2. **Explore Available Options**: Use `list_available_configs()` to see what JaxARC provides.

3. **Use Merge Strategies**: Choose appropriate merge strategy based on whether you want your settings or JaxARC's to take precedence.

4. **Override Selectively**: Only override what you need; let JaxARC provide sensible defaults.

5. **Validate Configs**: Always test your configuration merging with small examples first.

6. **Version Control**: Keep your custom configs in version control separate from JaxARC.

## Troubleshooting

### Config Not Found
```python
from jaxarc import get_jaxarc_config_dir

# Check if JaxARC configs are properly installed
try:
    config_dir = get_jaxarc_config_dir()
    print(f"✅ JaxARC configs found at: {config_dir}")
except FileNotFoundError as e:
    print(f"❌ JaxARC configs not found: {e}")
    print("Try reinstalling JaxARC with: pip install --force-reinstall jaxarc")
```

### Config Merging Issues
```python
from jaxarc import extend_jaxarc_config
from omegaconf import OmegaConf

# Debug config merging
try:
    cfg = extend_jaxarc_config(
        custom_config_dir=Path("my_project/conf"),
        overrides=["debug=true"]
    )
    print("✅ Config merged successfully")
    print(OmegaConf.to_yaml(cfg))
except Exception as e:
    print(f"❌ Config merge failed: {e}")
```

### Hydra Conflicts
```python
from hydra.core.global_hydra import GlobalHydra

# Clear Hydra state if you get conflicts
GlobalHydra.instance().clear()

# Then try loading again
cfg = load_jaxarc_config()
```
