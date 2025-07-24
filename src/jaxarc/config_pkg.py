"""Configuration package utilities for downstream consumption."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def get_jaxarc_config_dir() -> Path:
    """Get the path to JaxARC configuration directory.
    
    This function provides the path to JaxARC's bundled configuration files
    for downstream packages to use as a base or reference.
    
    Returns:
        Path to the JaxARC configuration directory
        
    Example:
        ```python
        from jaxarc.config_pkg import get_jaxarc_config_dir
        
        config_dir = get_jaxarc_config_dir()
        print(f"JaxARC configs at: {config_dir}")
        ```
    """
    try:
        pkg_root = importlib.resources.files('jaxarc')
        config_dir = pkg_root / "conf"
        if config_dir.exists():
            return config_dir
    except Exception as e:
        logger.warning(f"Could not locate config directory: {e}")
    
        msg = (
            "Could not locate JaxARC configuration directory. "
            "Make sure JaxARC is properly installed with configuration files."
        )
        raise FileNotFoundError(msg)


def load_jaxarc_config(
    config_name: str = "config",
    overrides: list[str] | None = None,
    use_jaxarc_configs: bool = True
) -> DictConfig:
    """Load JaxARC configuration with optional overrides.
    
    This provides a convenient way for downstream packages to load
    JaxARC configurations with their own customizations.
    
    Args:
        config_name: Name of the main config file (without .yaml extension)
        overrides: List of Hydra-style override strings
        use_jaxarc_configs: Whether to use JaxARC's bundled configs as base
        
    Returns:
        Loaded and composed Hydra configuration
        
    Example:
        ```python
        from jaxarc.config_pkg import load_jaxarc_config
        
        # Load default JaxARC config
        cfg = load_jaxarc_config()
        
        # Load with overrides
        cfg = load_jaxarc_config(
            overrides=["dataset=concept_arc", "environment.max_episode_steps=50"]
        )
        ```
    """
    if overrides is None:
        overrides = []
        
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    if use_jaxarc_configs:
        config_dir = get_jaxarc_config_dir()
    else:
        # Let user provide their own config dir via hydra.initialize
        msg = (
            "When use_jaxarc_configs=False, initialize Hydra yourself "
            "with your config directory before calling this function"
        )
        raise ValueError(msg)
    
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=config_name, overrides=overrides)
            logger.info(f"Loaded JaxARC config '{config_name}' with {len(overrides)} overrides")
            return cfg
    except Exception as e:
        logger.error(f"Failed to load config '{config_name}': {e}")
        raise


def extend_jaxarc_config(
    base_config_name: str = "config",
    custom_config_dir: Path | None = None,
    overrides: list[str] | None = None,
    merge_strategy: str = "custom_overrides_jaxarc"
) -> DictConfig:
    """Extend JaxARC configuration with custom configs.
    
    This allows downstream packages to build on top of JaxARC configs
    while adding their own configuration groups and settings.
    
    Args:
        base_config_name: JaxARC config to use as base
        custom_config_dir: Directory containing custom config files
        overrides: Additional override strings
        merge_strategy: How to merge configs ('custom_overrides_jaxarc' or 'jaxarc_overrides_custom')
        
    Returns:
        Merged configuration with custom extensions
        
    Example:
        ```python
        from jaxarc.config_pkg import extend_jaxarc_config
        from pathlib import Path
        
        # Extend JaxARC config with custom settings
        cfg = extend_jaxarc_config(
            custom_config_dir=Path("my_project/conf"),
            overrides=["my_custom_group=special_setting"]
        )
        ```
    """
    if overrides is None:
        overrides = []
        
    # Load base JaxARC config
    jaxarc_cfg = load_jaxarc_config(base_config_name, overrides=[])
    
    if custom_config_dir is None:
        # No custom configs, just apply overrides to JaxARC config
        if overrides:
            for override in overrides:
                if "=" in override:
                    key, value = override.split("=", 1)
                    try:
                        parsed_value = OmegaConf.create({key: value})[key]
                        OmegaConf.set(jaxarc_cfg, key, parsed_value)
                    except Exception as e:
                        logger.warning(f"Failed to apply override '{override}': {e}")
        return jaxarc_cfg
    
    # Load custom config if directory provided
    custom_config_dir = Path(custom_config_dir)
    if not custom_config_dir.exists():
        logger.warning(f"Custom config directory does not exist: {custom_config_dir}")
        return jaxarc_cfg
        
    # Clear Hydra and initialize with custom config dir
    GlobalHydra.instance().clear()
    
    try:
        with initialize_config_dir(config_dir=str(custom_config_dir), version_base=None):
            custom_cfg = compose(config_name="config", overrides=overrides)
            
        # Merge configurations based on strategy
        if merge_strategy == "custom_overrides_jaxarc":
            # Custom config takes precedence
            merged_cfg = OmegaConf.merge(jaxarc_cfg, custom_cfg)
        else:  # "jaxarc_overrides_custom"
            # JaxARC config takes precedence  
            merged_cfg = OmegaConf.merge(custom_cfg, jaxarc_cfg)
            
        logger.info("Successfully merged JaxARC and custom configurations")
        return merged_cfg
        
    except Exception as e:
        logger.error(f"Failed to load/merge custom config: {e}")
        return jaxarc_cfg


def list_available_configs() -> dict[str, list[str]]:
    """List all available configuration groups and options.
    
    Returns:
        Dictionary mapping config group names to available config files
        
    Example:
        ```python
        from jaxarc.config_pkg import list_available_configs
        
        configs = list_available_configs()
        print("Available datasets:", configs["dataset"])
        print("Available actions:", configs["action"])
        ```
    """
    config_dir = get_jaxarc_config_dir()
    
    config_groups = {}
    
    for item in config_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            group_name = item.name
            config_files = []
            
            for config_file in item.glob("*.yaml"):
                config_files.append(config_file.stem)
                
            for config_file in item.glob("*.yml"):
                config_files.append(config_file.stem)
                
            if config_files:
                config_groups[group_name] = sorted(config_files)
    
    return config_groups


def create_config_template(
    output_dir: Path,
    include_examples: bool = True
) -> None:
    """Create a template configuration structure for downstream projects.
    
    This helps downstream projects set up their own config structure
    that can extend JaxARC configurations.
    
    Args:
        output_dir: Directory to create template structure in
        include_examples: Whether to include example override files
        
    Example:
        ```python
        from jaxarc.config_pkg import create_config_template
        from pathlib import Path
        
        create_config_template(Path("my_project/conf"))
        ```
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main config template
    main_config = {
        "defaults": [
            "_self_",
            "base_jaxarc: config",  # Reference to JaxARC base
            # Custom groups can be added here
        ],
        "# Add your custom configuration here": None,
        "# This will be merged with JaxARC base configuration": None,
    }
    
    main_config_path = output_dir / "config.yaml"
    with main_config_path.open("w") as f:
        OmegaConf.save(main_config, f)
    
    if include_examples:
        # Create example override structure
        examples_dir = output_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Create example custom dataset config
        custom_dataset = {
            "# Example custom dataset configuration": None,
            "dataset_name": "my_custom_dataset",
            "task_split": "train",
            "# Override JaxARC defaults": None,
            "max_grid_height": 40,
            "max_grid_width": 40,
        }
        
        with (examples_dir / "custom_dataset.yaml").open("w") as f:
            OmegaConf.save(custom_dataset, f)
        
        # Create example environment override
        custom_env = {
            "# Example environment customization": None,
            "max_episode_steps": 200,
            "debug_level": "verbose",
            "# Add custom environment settings": None,
        }
        
        with (examples_dir / "custom_environment.yaml").open("w") as f:
            OmegaConf.save(custom_env, f)
    
    # Create README
    readme_content = """# Configuration Template

This directory contains configuration templates for extending JaxARC.

## Usage

1. Use the base JaxARC configurations:
   ```python
   from jaxarc.config_pkg import load_jaxarc_config
   cfg = load_jaxarc_config()
   ```

2. Extend with your custom configs:
   ```python
   from jaxarc.config_pkg import extend_jaxarc_config
   cfg = extend_jaxarc_config(
       custom_config_dir=Path("my_project/conf"),
       overrides=["my_custom_group=my_setting"]
   )
   ```

3. Available JaxARC config groups:
   - dataset/ - Dataset configurations
   - environment/ - Environment settings  
   - action/ - Action space configurations
   - reward/ - Reward function settings
   - visualization/ - Visualization options
   - storage/ - Storage and output settings
   - logging/ - Logging configurations
   - wandb/ - Weights & Biases integration

## Files

- config.yaml - Main configuration that extends JaxARC
- examples/ - Example override configurations
"""
    
    with (output_dir / "README.md").open("w") as f:
        f.write(readme_content)
    
    logger.info(f"Created configuration template in {output_dir}")
