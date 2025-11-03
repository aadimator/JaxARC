# Wrappers API

Wrappers transform environment interfaces for different use cases.

JaxARC provides two types of wrappers:

- **Action Wrappers**: Convert between action formats (dict → mask, bbox → mask, flatten)
- **Observation Wrappers**: Add channels to observations (input grid, answer, clipboard, context)

## Action Wrappers

### PointActionWrapper

```{eval-rst}
.. autoclass:: jaxarc.PointActionWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### BboxActionWrapper

```{eval-rst}
.. autoclass:: jaxarc.BboxActionWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### FlattenActionWrapper

```{eval-rst}
.. autoclass:: jaxarc.FlattenActionWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

## Observation Wrappers

### InputGridObservationWrapper

```{eval-rst}
.. autoclass:: jaxarc.InputGridObservationWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### AnswerObservationWrapper

```{eval-rst}
.. autoclass:: jaxarc.AnswerObservationWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### ClipboardObservationWrapper

```{eval-rst}
.. autoclass:: jaxarc.ClipboardObservationWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

### ContextualObservationWrapper

```{eval-rst}
.. autoclass:: jaxarc.ContextualObservationWrapper
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Example

```python
from jaxarc import make
from jaxarc import PointActionWrapper, InputGridObservationWrapper

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
