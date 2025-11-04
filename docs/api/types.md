# Types API

Type definitions for JaxARC environment parameters and timesteps.

## Module Contents

```{eval-rst}
.. automodule:: jaxarc.types
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Example

```python
import jax
from jaxarc import make, EnvParams, TimeStep

# Create environment
env, env_params = make("Mini")

# env_params is of type EnvParams
assert isinstance(env_params, EnvParams)

# Reset returns State and TimeStep
key = jax.random.PRNGKey(42)
state, timestep = env.reset(key, env_params)

# timestep is of type TimeStep
assert isinstance(timestep, TimeStep)

# TimeStep contains:
# - observation: jax.Array
# - reward: float
# - discount: float
# - step_type: StepType (FIRST, MID, LAST)
# - extras: dict

print(f"Observation shape: {timestep.observation.shape}")
print(f"Reward: {timestep.reward}")
print(f"Done: {timestep.step_type.last()}")
```

## JAX Typing

JaxARC uses `jaxtyping` for array shape annotations:

```python
from jaxtyping import Array, Float, Int, Bool

# Example type hints
observation: Float[Array, "height width channels"]
selection: Bool[Array, "height width"]
operation: Int[Array, ""]
```
