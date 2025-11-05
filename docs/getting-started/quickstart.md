# Quick Start

Learn the JaxARC basics in 5 minutes. This guide covers the essential concepts
you need to start using JaxARC.

## Your First Environment

The simplest way to start with JaxARC is to create an environment and interact
with it:

```python
import jax
import jaxarc

# Create an environment (returns env and env_params)
# auto_download=True will download the dataset if it doesn't exist
env, env_params = jaxarc.make(
    "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True
)

# Reset the environment to get the initial state and timestep
key = jax.random.PRNGKey(0)
state, timestep = env.reset(key, env_params=env_params)

# Print basic information
print(f"Observation shape: {timestep.observation.shape}")
print(f"Step type: {timestep.step_type}")
```

### Breaking It Down

1. **`jaxarc.make("Mini-...")`** creates an environment instance and environment
   parameters. The ID specifies the dataset ("Mini" for MiniARC) and optionally
   a specific task.

2. **`env.reset(key, env_params=env_params)`** resets the environment and
   returns both `state` and `timestep`. Unlike Gymnasium, JAX environments
   require an explicit PRNG key for reproducibility.

3. **`state`** is an immutable object containing internal environment state

4. **`timestep`** contains the observable information:
   - `observation`: The current grid state (JAX array)
   - `step_type`: First, mid, or last step of episode
   - `reward`: Current reward (0.0 initially)
   - `discount`: Discount factor (1.0 initially)
   - `extras`: Additional information (dict)

## Understanding State

JaxARC uses immutable state objects. Once created, state values never change -
operations return new state objects instead:

```python
# Reset returns initial state and timestep
state, timestep = env.reset(key, env_params=env_params)
print(f"Initial: step_type={timestep.step_type}, reward={timestep.reward}")

# Stepping returns NEW state and timestep
action_space = env.action_space(env_params)
action = action_space.sample(key)
next_state, next_timestep = env.step(state, action, env_params=env_params)

# Original state is unchanged
print(f"Original step_type: {timestep.step_type}")
print(f"New step_type: {next_timestep.step_type}, reward={next_timestep.reward}")
```

**Why immutable?** This enables JAX's powerful transformations like `jax.jit`
(just-in-time compilation) and `jax.vmap` (vectorization).

## Taking Actions

Environments have action and observation spaces that define valid actions and
observations:

```python
# Check action space
action_space = env.action_space(env_params)
print(f"Action space: {action_space}")

# Sample a random action (requires a PRNG key)
key, subkey = jax.random.split(key)
action = action_space.sample(subkey)

# Take the action
next_state, next_timestep = env.step(state, action, env_params=env_params)
print(f"Reward: {next_timestep.reward}")
print(f"Step type: {next_timestep.step_type}")
```

## The Environment Loop

Here's the standard pattern for interacting with an environment:

```python
import jax
import jaxarc

# Setup
env, env_params = jaxarc.make("Mini-Most_Common_color_l6ab0lf3xztbyxsu3p")
key = jax.random.PRNGKey(42)
state, timestep = env.reset(key, env_params=env_params)

# Run episode
total_reward = 0.0
step_count = 0
action_space = env.action_space(env_params)

while not timestep.last() and step_count < 100:
    # Sample action
    key, subkey = jax.random.split(key)
    action = action_space.sample(subkey)

    # Take step
    state, timestep = env.step(state, action, env_params=env_params)

    # Accumulate reward
    total_reward += float(timestep.reward)
    step_count += 1

print(f"Episode finished after {step_count} steps")
print(f"Total reward: {total_reward}")
```

## PRNG Keys

JAX uses explicit random number generation for reproducibility. You must manage
PRNG keys:

```python
# Create initial key
key = jax.random.PRNGKey(0)

# Split key before each random operation
key, reset_key = jax.random.split(key)
state, timestep = env.reset(reset_key, env_params=env_params)

action_space = env.action_space(env_params)
key, action_key = jax.random.split(key)
action = action_space.sample(action_key)

# Using the same key twice gives the same result
key1 = jax.random.PRNGKey(0)
key2 = jax.random.PRNGKey(0)

state1, timestep1 = env.reset(key1, env_params=env_params)
state2, timestep2 = env.reset(key2, env_params=env_params)

# These will be identical
assert jax.numpy.array_equal(timestep1.observation, timestep2.observation)
```

**Key Point**: Always split your PRNG key before using it. Never reuse the same
key for multiple operations.

## Next Steps

Now that you understand the basics, try:

1. **[Complete Example](first-example.md)** - See a full random agent
   implementation
2. **[Downloading Datasets](../tutorials/downloading-datasets.md)** - Learn how
   to access ARC datasets
3. **[Creating Agents](../tutorials/creating-agents.md)** - Build your own
   agents

## Quick Reference

```python
import jax
import jaxarc

# Create environment
env, env_params = jaxarc.make(
    "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True
)

# Reset
key = jax.random.PRNGKey(0)
state, timestep = env.reset(key, env_params=env_params)

# Step
action_space = env.action_space(env_params)
action = action_space.sample(key)
next_state, next_timestep = env.step(state, action, env_params=env_params)

# Access timestep fields
observation = timestep.observation
reward = timestep.reward
step_type = timestep.step_type
discount = timestep.discount

# Check if episode is done
is_done = timestep.last()
```
