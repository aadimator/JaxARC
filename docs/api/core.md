# Core API

The core API provides essential functions and classes for creating and
interacting with JaxARC environments.

## Quick Example

```python
import jax
from jaxarc import make, Action

# Create environment
env, env_params = make("Mini")

# Reset and run episode
key = jax.random.PRNGKey(42)
state, timestep = env.reset(key, env_params)

# Take action
action = Action(...)  # Create action
state, timestep = env.step(state, action, env_params)
```

## Environment Creation

```{eval-rst}
.. autofunction:: jaxarc.make
```

## Core Classes

### Environment

```{eval-rst}
.. autoclass:: jaxarc.Environment
   :members:
   :undoc-members:
   :show-inheritance:
```

### State

```{eval-rst}
.. autoclass:: jaxarc.State
   :members:
   :undoc-members:
   :show-inheritance:
```

### Action

```{eval-rst}
.. autoclass:: jaxarc.Action
   :members:
   :undoc-members:
   :show-inheritance:
```
