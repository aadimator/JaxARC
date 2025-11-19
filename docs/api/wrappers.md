# Wrappers API

Wrappers transform environment interfaces for different use cases.

JaxARC provides two types of wrappers:

- **Action Wrappers**: Convert between action formats (dict → mask, bbox → mask,
  flatten)
- **Observation Wrappers**: Add channels to observations (input grid, answer,
  clipboard, context)
- **Visualization Wrappers**: Enhance rendering capabilities (step visualization)

## Action Wrappers

### PointActionWrapper

```{eval-rst}
.. autoclass:: jaxarc.wrappers.PointActionWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### BboxActionWrapper

```{eval-rst}
.. autoclass:: jaxarc.wrappers.BboxActionWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### FlattenActionWrapper

```{eval-rst}
.. autoclass:: jaxarc.wrappers.FlattenActionWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

## Observation Wrappers

### InputGridObservationWrapper

```{eval-rst}
.. autoclass:: jaxarc.wrappers.InputGridObservationWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### AnswerObservationWrapper

```{eval-rst}
.. autoclass:: jaxarc.wrappers.AnswerObservationWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### ClipboardObservationWrapper

```{eval-rst}
.. autoclass:: jaxarc.wrappers.ClipboardObservationWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### ContextualObservationWrapper

```{eval-rst}
.. autoclass:: jaxarc.wrappers.ContextualObservationWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

## Visualization Wrappers

### StepVisualizationWrapper

```{eval-rst}
.. autoclass:: jaxarc.wrappers.StepVisualizationWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Example

```python
from jaxarc import make
from jaxarc.wrappers import PointActionWrapper, InputGridObservationWrapper

# Create base environment
env, env_params = make("Mini")

# Add wrappers
env = PointActionWrapper(env)
env = InputGridObservationWrapper(env)

# Use wrapped environment
state, timestep = env.reset(key, env_params)
action = {"operation": 2, "row": 5, "col": 5}
state, timestep = env.step(state, action, env_params)
```

## See Also

- {doc}`../tutorials/using-wrappers` - Tutorial on using wrappers
- {doc}`core` - Core environment API
