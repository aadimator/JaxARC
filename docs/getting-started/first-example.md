# Your First Complete Example

This guide walks through a complete, runnable example of using JaxARC to create
and run a random agent. By the end, you'll have a working baseline agent that
you can build upon.

## Complete Random Agent

Here's a complete example that creates an environment, runs a random agent for
multiple episodes, and tracks performance:

```python
import jax
import jax.numpy as jnp
import jaxarc

def run_random_agent(num_episodes=10, max_steps=100, seed=0):
    """
    Run a random agent on MiniARC for multiple episodes.

    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility

    Returns:
        Dictionary with episode statistics
    """
    # Create environment (downloads dataset if needed)
    env, env_params = jaxarc.make("Mini", auto_download=True)

    # Initialize PRNG key
    key = jax.random.PRNGKey(seed)

    # Get action space
    action_space = env.action_space(env_params)

    # Track statistics
    episode_rewards = []
    episode_lengths = []

    print(f"Running {num_episodes} episodes...")
    print(f"Observation shape: {env.observation_shape()}")
    print(f"Action space: {action_space}")
    print("-" * 60)

    for episode in range(num_episodes):
        # Reset environment
        key, reset_key = jax.random.split(key)
        state, timestep = env.reset(reset_key, env_params=env_params)

        episode_reward = 0.0
        step_count = 0

        # Run episode
        while not timestep.last() and step_count < max_steps:
            # Sample random action
            key, action_key = jax.random.split(key)
            action = action_space.sample(action_key)

            # Take step
            state, timestep = env.step(state, action, env_params=env_params)

            # Track metrics
            episode_reward += float(timestep.reward)
            step_count += 1

        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        # Print episode summary
        status = "✓ Solved" if timestep.last() else "✗ Max steps"
        print(f"Episode {episode+1:2d}: {status} | "
              f"Steps: {step_count:3d} | "
              f"Reward: {episode_reward:6.2f}")

    # Calculate overall statistics
    stats = {
        "mean_reward": jnp.mean(jnp.array(episode_rewards)),
        "std_reward": jnp.std(jnp.array(episode_rewards)),
        "mean_length": jnp.mean(jnp.array(episode_lengths)),
        "success_rate": sum(1 for r in episode_rewards if r > 0) / num_episodes,
    }

    print("-" * 60)
    print(f"Average reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Average episode length: {stats['mean_length']:.1f}")
    print(f"Success rate: {stats['success_rate']:.1%}")

    return stats

if __name__ == "__main__":
    # Run the agent
    stats = run_random_agent(num_episodes=10, max_steps=100, seed=42)
```

## Understanding the Code

Let's break down the key components:

### 1. Environment Setup

```python
env, env_params = jaxarc.make("Mini", auto_download=True)
key = jax.random.PRNGKey(seed)
action_space = env.action_space(env_params)
```

- `jaxarc.make()` creates the environment and parameters
- `auto_download=True` downloads the dataset if needed
- We initialize a PRNG key for reproducibility
- Get the action space once (it's fixed for the environment)

### 2. Episode Loop

```python
for episode in range(num_episodes):
    key, reset_key = jax.random.split(key)
    state, timestep = env.reset(reset_key, env_params=env_params)

    while not timestep.last() and step_count < max_steps:
        # ... episode logic
```

- Outer loop iterates over episodes
- Each episode starts with `reset()` which returns `(state, timestep)`
- Inner loop runs until `timestep.last()` or max steps
- We always split the PRNG key before using it

### 3. Action Sampling

```python
key, action_key = jax.random.split(key)
action = action_space.sample(action_key)
state, timestep = env.step(state, action, env_params=env_params)
```

- Split key to get a fresh random key
- Sample action from the action space
- Step returns new `(state, timestep)` tuple (immutable!)
- Always pass `env_params` to `step()`

### 4. Metrics Tracking

```python
episode_reward += float(timestep.reward)
step_count += 1
```

- Track cumulative reward per episode (from `timestep.reward`)
- Count steps to measure episode length
- Convert JAX arrays to Python floats for accumulation

## Running the Example

Save the code above to a file (e.g., `random_agent.py`) and run it:

```bash
python random_agent.py
```

**Expected output:**

```
Running 10 episodes...
Observation shape: (5, 5, 1)
Action space: DictSpace({operation=DiscreteSpace(num_values=35, dtype=int32, name='operation'), selection=MultiDiscreteSpace(num_values=[Array([2, 2, 2, 2, 2], dtype=int32), Array([2, 2, 2, 2, 2], dtype=int32), Array([2, 2, 2, 2, 2], dtype=int32), Array([2, 2, 2, 2, 2], dtype=int32), Array([2, 2, 2, 2, 2], dtype=int32)], dtype=int32, name='selection_mask')}, name='arc_action')
------------------------------------------------------------
Episode  1: ✓ Solved | Steps:  62 | Reward:  -1.63
Episode  2: ✓ Solved | Steps:  57 | Reward:  -1.64
Episode  3: ✗ Max steps | Steps: 100 | Reward:  -0.34
Episode  4: ✓ Solved | Steps:   3 | Reward:  -1.01
Episode  5: ✓ Solved | Steps:  11 | Reward:  -1.37
Episode  6: ✓ Solved | Steps:   5 | Reward:  -0.70
Episode  7: ✓ Solved | Steps:   4 | Reward:  -1.02
Episode  8: ✓ Solved | Steps:  70 | Reward:  -1.71
Episode  9: ✓ Solved | Steps:  41 | Reward:  -1.68
Episode 10: ✗ Max steps | Steps: 100 | Reward:  -0.82
------------------------------------------------------------
Average reward: -1.19 ± 0.46
Average episode length: 45.3
Success rate: 0.0%
```

**Note**: A random agent typically doesn't solve ARC tasks (success rate near
0%), but this provides a baseline for comparison.
