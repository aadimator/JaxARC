# Utilities API

Helper functions and utilities for visualization, data loading, and more.

## Visualization

```{eval-rst}
.. automodule:: jaxarc.utils.visualization
   :members:
   :undoc-members:
   :show-inheritance:
```

## Core Utilities

```{eval-rst}
.. automodule:: jaxarc.utils.core
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Visualization

```python
from jaxarc.utils.visualization import draw_grid_svg, draw_task_pair_svg
import jax.numpy as jnp

# Create a simple grid
grid = jnp.array(
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
)

# Draw as SVG
svg = draw_grid_svg(grid)
display(svg)  # In Jupyter notebook
```

### Configuration

```python
from jaxarc.utils.core import get_config
from jaxarc import JaxArcConfig

# Load configuration with overrides
hydra_cfg = get_config(
    overrides=[
        "dataset=mini_arc",
    ]
)

config = JaxArcConfig.from_hydra(hydra_cfg)
```

## See Also

- {doc}`../tutorials/visualizing-tasks` - Tutorial on visualization
