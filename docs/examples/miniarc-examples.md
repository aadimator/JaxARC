# MiniARC Examples

MiniARC rapid prototyping examples for fast experimentation with 5x5 grids.

## Rapid Prototyping Setup

### Quick MiniARC Configuration

```python
import jax
from jaxarc.parsers import MiniArcParser
from jaxarc.envs import create_standard_config, arc_reset, arc_step
from omegaconf import DictConfig
import jax.numpy as jnp

# Optimized MiniARC configuration for rapid iteration
parser_config = DictConfig(
    {
        "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
        "grid": {"max_grid_height": 5, "max_grid_width": 5},
        "max_train_pairs": 3,
        "max_test_pairs": 1,
    }
)

# Create parser and load task
parser = MiniArcParser(parser_config)
key = jax.random.PRNGKey(42)
task = parser.get_random_task(key)

print(f"MiniARC task loaded: {task.train_input_grids.shape}")
print(f"Grid size: {task.train_input_grids.shape[-2:]} (25 cells vs 900 for full ARC)")
```

### Performance-Optimized Environment

```python
# Create environment optimized for MiniARC
def create_miniarc_config():
    """Create configuration optimized for MiniARC rapid prototyping."""
    return create_standard_config(
        max_episode_steps=25,  # Shorter episodes for 5x5 grids
        success_bonus=5.0,  # Quick positive feedback
        step_penalty=-0.001,  # Lower penalty for experimentation
        log_operations=False,  # Disable logging for speed
    )


config = create_miniarc_config()
print(f"MiniARC config: {config.max_episode_steps} max steps")
```

## Performance Comparisons

### Speed Benchmarking

```python
import time


def benchmark_miniarc_vs_full():
    """Compare MiniARC vs full ARC performance."""

    # MiniARC setup
    miniarc_config = DictConfig(
        {
            "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
            "grid": {"max_grid_height": 5, "max_grid_width": 5},
        }
    )
    miniarc_parser = MiniArcParser(miniarc_config)

    # Full ARC setup (if available)
    full_config = DictConfig(
        {
            "challenges": {"path": "data/raw/ARC-AGI-2/data/training"},
            "solutions": {"path": "data/raw/ARC-AGI-2/data/training"},
            "grid": {"max_grid_height": 30, "max_grid_width": 30},
        }
    )

    # Benchmark task loading
    num_tasks = 100
    key = jax.random.PRNGKey(42)

    # MiniARC timing
    start = time.time()
    for i in range(num_tasks):
        task_key = jax.random.fold_in(key, i)
        task = miniarc_parser.get_random_task(task_key)
    miniarc_time = time.time() - start

    print(f"MiniARC: {num_tasks} tasks loaded in {miniarc_time:.3f}s")
    print(f"Average: {miniarc_time/num_tasks*1000:.2f}ms per task")

    # Memory comparison
    miniarc_task = miniarc_parser.get_random_task(key)
    miniarc_memory = miniarc_task.train_input_grids.size * 4  # 4 bytes per int32
    full_memory = 30 * 30 * 4  # Equivalent full ARC grid

    print(f"Memory usage:")
    print(f"  MiniARC: {miniarc_memory} bytes per grid")
    print(f"  Full ARC: {full_memory} bytes per grid")
    print(f"  Reduction: {full_memory/miniarc_memory:.1f}x less memory")


benchmark_miniarc_vs_full()
```

### Batch Processing Performance

```python
def benchmark_batch_processing():
    """Demonstrate MiniARC batch processing advantages."""

    # Create batch of MiniARC tasks
    batch_size = 1000
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

    # Load tasks in batch
    start = time.time()
    tasks = []
    for key in keys:
        task = parser.get_random_task(key)
        tasks.append(task)
    load_time = time.time() - start

    print(f"Loaded {batch_size} MiniARC tasks in {load_time:.3f}s")
    print(f"Rate: {batch_size/load_time:.0f} tasks/second")

    # Batch environment processing
    config = create_miniarc_config()

    def single_episode(episode_key, task):
        state, obs = arc_reset(episode_key, config, task)
        return state.working_grid.sum()  # Simple metric

    # Process batch
    start = time.time()
    results = []
    for i, task in enumerate(tasks[:100]):  # First 100 for demo
        result = single_episode(keys[i], task)
        results.append(result)
    batch_time = time.time() - start

    print(f"Processed {len(results)} episodes in {batch_time:.3f}s")
    print(f"Rate: {len(results)/batch_time:.0f} episodes/second")


benchmark_batch_processing()
```

## Rapid Development Workflows

### Quick Algorithm Testing

```python
def test_algorithm_quickly(algorithm_fn, num_tests=50):
    """Test an algorithm quickly on MiniARC tasks."""
    config = create_miniarc_config()
    results = []

    for i in range(num_tests):
        key = jax.random.PRNGKey(i)
        task = parser.get_random_task(key)

        # Initialize environment
        state, obs = arc_reset(key, config, task)

        # Run algorithm
        episode_reward = 0.0
        for step in range(10):  # Short episodes
            action = algorithm_fn(state, obs, step)
            state, obs, reward, done, info = arc_step(state, action, config)
            episode_reward += reward
            if done:
                break

        results.append(
            {
                "reward": episode_reward,
                "steps": step + 1,
                "similarity": info["similarity"],
            }
        )

    # Quick analysis
    rewards = [r["reward"] for r in results]
    similarities = [r["similarity"] for r in results]

    print(f"Algorithm tested on {num_tests} MiniARC tasks:")
    print(f"  Average reward: {jnp.mean(jnp.array(rewards)):.3f}")
    print(f"  Average similarity: {jnp.mean(jnp.array(similarities)):.3f}")
    print(
        f"  Success rate: {sum(1 for r in results if r['similarity'] > 0.9)/len(results)*100:.1f}%"
    )

    return results


# Example algorithm: random actions
def random_algorithm(state, obs, step):
    """Simple random algorithm for testing."""
    key = jax.random.PRNGKey(step)
    grid_shape = state.working_grid.shape

    # Random selection
    selection_key, op_key = jax.random.split(key)
    selection = jax.random.bernoulli(selection_key, 0.2, grid_shape)
    operation = jax.random.randint(op_key, (), 1, 6)  # Colors 1-5

    return {
        "selection": selection,
        "operation": operation,
    }


# Test the algorithm
results = test_algorithm_quickly(random_algorithm, 20)
```

### Hyperparameter Tuning

```python
def tune_hyperparameters():
    """Quick hyperparameter tuning on MiniARC."""

    # Parameter grid
    param_grid = {
        "max_episode_steps": [10, 25, 50],
        "success_bonus": [1.0, 5.0, 10.0],
        "step_penalty": [-0.001, -0.01, -0.1],
    }

    best_score = -float("inf")
    best_params = None

    # Grid search (simplified)
    for steps in param_grid["max_episode_steps"]:
        for bonus in param_grid["success_bonus"]:
            for penalty in param_grid["step_penalty"]:

                # Create config with these parameters
                config = create_standard_config(
                    max_episode_steps=steps,
                    success_bonus=bonus,
                    step_penalty=penalty,
                )

                # Quick evaluation
                total_reward = 0.0
                num_tests = 10  # Small for speed

                for i in range(num_tests):
                    key = jax.random.PRNGKey(i)
                    task = parser.get_random_task(key)
                    state, obs = arc_reset(key, config, task)

                    # Simple episode
                    for step in range(5):
                        action = {
                            "selection": jnp.ones((1, 1), dtype=jnp.bool_),
                            "operation": jnp.array(1, dtype=jnp.int32),
                        }
                        state, obs, reward, done, info = arc_step(state, action, config)
                        total_reward += reward
                        if done:
                            break

                avg_reward = total_reward / num_tests

                if avg_reward > best_score:
                    best_score = avg_reward
                    best_params = {
                        "max_episode_steps": steps,
                        "success_bonus": bonus,
                        "step_penalty": penalty,
                    }

                print(
                    f"Steps={steps}, Bonus={bonus}, Penalty={penalty}: {avg_reward:.3f}"
                )

    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.3f}")

    return best_params


# Run hyperparameter tuning
# best_params = tune_hyperparameters()
```

## Development and Testing Utilities

### Quick Visualization

```python
from jaxarc.utils.visualization import log_grid_to_console


def visualize_miniarc_task(task_id=None):
    """Quickly visualize a MiniARC task."""
    if task_id is not None:
        key = jax.random.PRNGKey(task_id)
    else:
        key = jax.random.PRNGKey(42)

    task = parser.get_random_task(key)

    print("MiniARC Task Visualization")
    print("=" * 30)

    for i in range(task.num_train_pairs):
        print(f"\nTraining Pair {i+1}:")
        print("Input:")
        log_grid_to_console(task.train_input_grids[i])
        print("Output:")
        log_grid_to_console(task.train_output_grids[i])

    print(f"\nTest Input:")
    log_grid_to_console(task.test_input_grids[0])

    return task


# Visualize a random task
task = visualize_miniarc_task(123)
```

### Debugging Utilities

```python
def debug_miniarc_episode():
    """Debug a MiniARC episode step by step."""
    config = create_miniarc_config()
    key = jax.random.PRNGKey(42)
    task = parser.get_random_task(key)

    print("Debugging MiniARC Episode")
    print("=" * 30)

    # Initialize
    state, obs = arc_reset(key, config, task)
    print("Initial state:")
    log_grid_to_console(state.working_grid)
    print(f"Target similarity: {obs['similarity']:.3f}")

    # Take a few debug steps
    for step in range(3):
        print(f"\nStep {step + 1}:")

        # Simple action
        action = {
            "selection": jnp.zeros((5, 5), dtype=jnp.bool_).at[step, step].set(True),
            "operation": jnp.array(step + 1, dtype=jnp.int32),
        }

        print(f"Action: Fill position ({step}, {step}) with color {step + 1}")

        state, obs, reward, done, info = arc_step(state, action, config)

        print("Result:")
        log_grid_to_console(state.working_grid)
        print(f"Reward: {reward:.3f}")
        print(f"Similarity: {info['similarity']:.3f}")
        print(f"Done: {done}")

        if done:
            print("Episode completed!")
            break


# Run debugging session
debug_miniarc_episode()
```

### Performance Profiling

```python
def profile_miniarc_operations():
    """Profile different operations on MiniARC."""
    import time

    config = create_miniarc_config()
    key = jax.random.PRNGKey(42)
    task = parser.get_random_task(key)

    operations = [
        ("Fill", 1),
        ("Flood Fill", 10),
        ("Copy Input", 31),
        ("Reset", 32),
    ]

    print("MiniARC Operation Profiling")
    print("=" * 30)

    for op_name, op_id in operations:
        state, obs = arc_reset(key, config, task)

        action = {
            "selection": jnp.ones((2, 2), dtype=jnp.bool_),
            "operation": jnp.array(op_id, dtype=jnp.int32),
        }

        # Time the operation
        start = time.time()
        for _ in range(1000):  # Multiple runs for accuracy
            state, obs, reward, done, info = arc_step(state, action, config)
        op_time = (time.time() - start) / 1000

        print(f"{op_name}: {op_time*1000:.3f}ms per operation")


# Run profiling
profile_miniarc_operations()
```

## Integration with Training Frameworks

### JAX Training Loop

```python
def miniarc_training_loop(num_episodes=1000):
    """Example training loop optimized for MiniARC."""
    config = create_miniarc_config()

    # Training metrics
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        key = jax.random.PRNGKey(episode)
        task = parser.get_random_task(key)

        # Episode
        state, obs = arc_reset(key, config, task)
        episode_reward = 0.0
        episode_length = 0

        for step in range(config.max_episode_steps):
            # Simple policy (replace with actual learning)
            action = {
                "selection": jnp.ones((1, 1), dtype=jnp.bool_),
                "operation": jnp.array((step % 5) + 1, dtype=jnp.int32),
            }

            state, obs, reward, done, info = arc_step(state, action, config)
            episode_reward += reward
            episode_length += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Log progress
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            avg_reward = jnp.mean(jnp.array(recent_rewards))
            avg_length = jnp.mean(jnp.array(episode_lengths[-100:]))
            print(
                f"Episode {episode+1}: Avg reward = {avg_reward:.3f}, Avg length = {avg_length:.1f}"
            )

    return episode_rewards, episode_lengths


# Run training loop
# rewards, lengths = miniarc_training_loop(500)
```

## Next Steps

- **[Advanced Patterns](advanced-patterns.md)**: Scale up with JAX
  transformations
- **[Basic Usage](basic-usage.md)**: Learn fundamental JaxARC patterns
- **[Configuration Guide](../configuration.md)**: Optimize configurations for
  your use case
- **[Datasets Guide](../datasets.md)**: Compare MiniARC with other datasets
