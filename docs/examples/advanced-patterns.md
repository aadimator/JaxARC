# Advanced Patterns

Advanced usage patterns featuring JAX transformations, batch processing, and integration with training frameworks.

## JAX Transformations

### JIT Compilation for Maximum Performance

```python
import jax
import jax.numpy as jnp
from jaxarc.envs import arc_reset, arc_step, create_standard_config
from jaxarc.parsers import MiniArcParser
from omegaconf import DictConfig
import time

# Setup
parser_config = DictConfig({
    "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
    "grid": {"max_grid_height": 5, "max_grid_width": 5},
})
parser = MiniArcParser(parser_config)
config = create_standard_config(max_episode_steps=50)

# JIT compile environment functions
@jax.jit
def jitted_reset(key, config, task):
    return arc_reset(key, config, task)

@jax.jit  
def jitted_step(state, action, config):
    return arc_step(state, action, config)

# Performance comparison
def compare_jit_performance():
    key = jax.random.PRNGKey(42)
    task = parser.get_random_task(key)
    
    action = {
        "selection": jnp.ones((2, 2), dtype=jnp.bool_),
        "operation": jnp.array(1, dtype=jnp.int32),
    }
    
    # Regular functions
    start = time.time()
    for _ in range(1000):
        state, obs = arc_reset(key, config, task)
        state, obs, reward, done, info = arc_step(state, action, config)
    regular_time = time.time() - start
    
    # JIT compiled functions (after warmup)
    state, obs = jitted_reset(key, config, task)  # Warmup
    jitted_step(state, action, config)  # Warmup
    
    start = time.time()
    for _ in range(1000):
        state, obs = jitted_reset(key, config, task)
        state, obs, reward, done, info = jitted_step(state, action, config)
    jit_time = time.time() - start
    
    print(f"Regular functions: {regular_time:.3f}s")
    print(f"JIT compiled: {jit_time:.3f}s")
    print(f"Speedup: {regular_time/jit_time:.1f}x")

compare_jit_performance()
```

### Vectorized Batch Processing with vmap

```python
def batch_episode_processing():
    """Process multiple episodes in parallel using vmap."""
    
    def single_episode(episode_key, task):
        """Run a single episode and return metrics."""
        state, obs = jitted_reset(episode_key, config, task)
        
        total_reward = 0.0
        steps_taken = 0
        final_similarity = 0.0
        
        for step in range(config.max_episode_steps):
            # Simple policy
            action = {
                "selection": jnp.ones((1, 1), dtype=jnp.bool_),
                "operation": jnp.array((step % 5) + 1, dtype=jnp.int32),
            }
            
            state, obs, reward, done, info = jitted_step(state, action, config)
            total_reward += reward
            steps_taken += 1
            final_similarity = info['similarity']
            
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'steps_taken': steps_taken,
            'final_similarity': final_similarity,
            'success': final_similarity > 0.9,
        }
    
    # Create batch of tasks and keys
    batch_size = 100
    base_key = jax.random.PRNGKey(42)
    keys = jax.random.split(base_key, batch_size)
    
    # Load batch of tasks
    tasks = []
    for i in range(batch_size):
        task_key = jax.random.fold_in(base_key, i + 1000)
        task = parser.get_random_task(task_key)
        tasks.append(task)
    
    # Vectorize the episode function
    batch_episode_fn = jax.vmap(single_episode, in_axes=(0, 0))
    
    # Process batch
    start = time.time()
    # Note: This is conceptual - actual vmap with tasks requires careful handling
    # of the task structure. In practice, you'd need to stack task arrays properly.
    batch_time = time.time() - start
    
    print(f"Processed {batch_size} episodes in batch")
    # print(f"Time: {batch_time:.3f}s")
    # print(f"Rate: {batch_size/batch_time:.0f} episodes/second")

batch_episode_processing()
```

### Parallel Multi-Device Processing with pmap

```python
def multi_device_processing():
    """Demonstrate multi-device processing with pmap."""
    
    # Check available devices
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    
    if len(devices) < 2:
        print("Multi-device demo requires multiple devices")
        return
    
    @jax.pmap
    def parallel_episode(keys, tasks_batch):
        """Run episodes in parallel across devices."""
        def device_episode(key, task):
            state, obs = arc_reset(key, config, task)
            total_reward = 0.0
            
            for step in range(10):  # Short episodes
                action = {
                    "selection": jnp.ones((1, 1), dtype=jnp.bool_),
                    "operation": jnp.array(1, dtype=jnp.int32),
                }
                state, obs, reward, done, info = arc_step(state, action, config)
                total_reward += reward
                if done:
                    break
            
            return total_reward
        
        return jax.vmap(device_episode)(keys, tasks_batch)
    
    # Prepare data for multiple devices
    num_devices = len(devices)
    episodes_per_device = 10
    total_episodes = num_devices * episodes_per_device
    
    # Create keys for each device
    base_key = jax.random.PRNGKey(42)
    device_keys = jax.random.split(base_key, num_devices)
    episode_keys = jnp.array([
        jax.random.split(device_key, episodes_per_device) 
        for device_key in device_keys
    ])
    
    print(f"Running {total_episodes} episodes across {num_devices} devices")
    # Note: Actual pmap execution would require proper task batching
    
multi_device_processing()
```

## Custom Configuration Patterns

### Dynamic Configuration

```python
def create_adaptive_config(task_difficulty="medium"):
    """Create configuration that adapts to task difficulty."""
    
    difficulty_settings = {
        "easy": {
            "max_episode_steps": 25,
            "success_bonus": 5.0,
            "step_penalty": -0.001,
        },
        "medium": {
            "max_episode_steps": 50,
            "success_bonus": 10.0,
            "step_penalty": -0.01,
        },
        "hard": {
            "max_episode_steps": 100,
            "success_bonus": 20.0,
            "step_penalty": -0.02,
        }
    }
    
    settings = difficulty_settings.get(task_difficulty, difficulty_settings["medium"])
    
    return create_standard_config(**settings)

# Example usage
easy_config = create_adaptive_config("easy")
hard_config = create_adaptive_config("hard")

print(f"Easy config: {easy_config.max_episode_steps} steps, {easy_config.success_bonus} bonus")
print(f"Hard config: {hard_config.max_episode_steps} steps, {hard_config.success_bonus} bonus")
```

### Configuration Composition

```python
from jaxarc.envs.config import ArcEnvConfig, ActionHandlerConfig
from jaxarc.envs.actions import SelectionActionHandler
import chex

def create_custom_config():
    """Create a custom configuration by composing components."""
    
    # Custom action handler configuration
    action_config = ActionHandlerConfig(
        operations=[0, 1, 2, 3, 10, 11, 31, 34],  # Specific operations only
        max_selection_size=10,
        require_valid_selection=True,
    )
    
    # Custom action handler
    action_handler = SelectionActionHandler(action_config)
    
    # Custom environment configuration
    env_config = ArcEnvConfig(
        max_episode_steps=75,
        success_bonus=15.0,
        step_penalty=-0.005,
        similarity_threshold=0.95,
        action_handler=action_handler,
        log_operations=True,
    )
    
    return env_config

custom_config = create_custom_config()
print(f"Custom config operations: {custom_config.action_handler.operations}")
```

## Advanced Training Patterns

### Curriculum Learning Implementation

```python
def curriculum_learning_demo():
    """Implement curriculum learning with increasing difficulty."""
    
    class CurriculumManager:
        def __init__(self):
            self.current_level = 0
            self.levels = [
                {"name": "basic", "max_steps": 25, "bonus": 5.0},
                {"name": "intermediate", "max_steps": 50, "bonus": 10.0},
                {"name": "advanced", "max_steps": 100, "bonus": 20.0},
            ]
            self.level_performance = []
        
        def get_current_config(self):
            level = self.levels[self.current_level]
            return create_standard_config(
                max_episode_steps=level["max_steps"],
                success_bonus=level["bonus"],
            )
        
        def update_performance(self, success_rate):
            self.level_performance.append(success_rate)
            
            # Advance if performance is good enough
            if len(self.level_performance) >= 10:  # Check last 10 episodes
                recent_performance = sum(self.level_performance[-10:]) / 10
                if recent_performance > 0.7 and self.current_level < len(self.levels) - 1:
                    self.current_level += 1
                    self.level_performance = []  # Reset for new level
                    print(f"Advanced to level {self.current_level}: {self.levels[self.current_level]['name']}")
    
    # Training with curriculum
    curriculum = CurriculumManager()
    
    for episode in range(100):
        config = curriculum.get_current_config()
        key = jax.random.PRNGKey(episode)
        task = parser.get_random_task(key)
        
        # Run episode
        state, obs = jitted_reset(key, config, task)
        success = False
        
        for step in range(config.max_episode_steps):
            action = {
                "selection": jnp.ones((1, 1), dtype=jnp.bool_),
                "operation": jnp.array((step % 3) + 1, dtype=jnp.int32),
            }
            state, obs, reward, done, info = jitted_step(state, action, config)
            
            if info['similarity'] > 0.9:
                success = True
                break
            if done:
                break
        
        curriculum.update_performance(1.0 if success else 0.0)
        
        if episode % 20 == 0:
            level_name = curriculum.levels[curriculum.current_level]["name"]
            print(f"Episode {episode}: Level {level_name}")

curriculum_learning_demo()
```

### Multi-Task Learning

```python
def multi_task_learning_demo():
    """Demonstrate multi-task learning across different parsers."""
    
    # Setup multiple parsers
    parsers = {
        "miniarc": MiniArcParser(DictConfig({
            "tasks": {"path": "data/raw/MiniARC/data/MiniARC"},
            "grid": {"max_grid_height": 5, "max_grid_width": 5},
        })),
    }
    
    # Add ConceptARC if available
    try:
        from jaxarc.parsers import ConceptArcParser
        parsers["conceptarc"] = ConceptArcParser(DictConfig({
            "corpus": {"path": "data/raw/ConceptARC/corpus"},
            "grid": {"max_grid_height": 30, "max_grid_width": 30},
        }))
    except:
        print("ConceptARC not available, using MiniARC only")
    
    def sample_task_from_distribution():
        """Sample task from mixed distribution."""
        parser_name = jax.random.choice(
            jax.random.PRNGKey(42), 
            jnp.array(list(parsers.keys())), 
            shape=()
        )
        # In practice, you'd implement proper sampling
        return "miniarc", parsers["miniarc"]
    
    # Multi-task training loop
    task_performance = {name: [] for name in parsers.keys()}
    
    for episode in range(50):
        # Sample task type
        task_type, parser = sample_task_from_distribution()
        
        key = jax.random.PRNGKey(episode)
        task = parser.get_random_task(key)
        
        # Adapt configuration based on task type
        if task_type == "miniarc":
            config = create_standard_config(max_episode_steps=25)
        else:
            config = create_standard_config(max_episode_steps=100)
        
        # Run episode
        state, obs = jitted_reset(key, config, task)
        episode_reward = 0.0
        
        for step in range(10):  # Short episodes for demo
            action = {
                "selection": jnp.ones((1, 1), dtype=jnp.bool_),
                "operation": jnp.array(1, dtype=jnp.int32),
            }
            state, obs, reward, done, info = jitted_step(state, action, config)
            episode_reward += reward
            if done:
                break
        
        task_performance[task_type].append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Task type {task_type}, Reward {episode_reward:.3f}")
    
    # Analyze multi-task performance
    for task_type, rewards in task_performance.items():
        if rewards:
            avg_reward = jnp.mean(jnp.array(rewards))
            print(f"{task_type}: Average reward {avg_reward:.3f}")

multi_task_learning_demo()
```

## Integration with Training Frameworks

### JAX-based RL Integration

```python
def jax_rl_integration_example():
    """Example integration with JAX-based RL frameworks."""
    
    # Define policy network (simplified)
    def init_policy_params(key, input_dim, hidden_dim, output_dim):
        """Initialize simple policy network parameters."""
        k1, k2 = jax.random.split(key)
        return {
            'w1': jax.random.normal(k1, (input_dim, hidden_dim)) * 0.1,
            'b1': jnp.zeros(hidden_dim),
            'w2': jax.random.normal(k2, (hidden_dim, output_dim)) * 0.1,
            'b2': jnp.zeros(output_dim),
        }
    
    def policy_forward(params, observation):
        """Simple policy forward pass."""
        # Flatten observation (simplified)
        x = observation['working_grid'].flatten()
        
        # Two-layer network
        h = jnp.tanh(jnp.dot(x, params['w1']) + params['b1'])
        logits = jnp.dot(h, params['w2']) + params['b2']
        
        return logits
    
    def sample_action(key, params, observation):
        """Sample action from policy."""
        logits = policy_forward(params, observation)
        
        # Sample operation
        op_key, sel_key = jax.random.split(key)
        operation = jax.random.categorical(op_key, logits[:10])  # First 10 for operations
        
        # Simple selection strategy
        grid_shape = observation['working_grid'].shape
        selection = jax.random.bernoulli(sel_key, 0.1, grid_shape)
        
        return {
            'selection': selection,
            'operation': operation,
        }
    
    # Training setup
    key = jax.random.PRNGKey(42)
    input_dim = 25  # 5x5 grid flattened
    hidden_dim = 64
    output_dim = 35  # Number of operations
    
    params = init_policy_params(key, input_dim, hidden_dim, output_dim)
    
    # Training episode with policy
    task = parser.get_random_task(key)
    config = create_standard_config(max_episode_steps=20)
    
    state, obs = jitted_reset(key, config, task)
    
    for step in range(10):
        action_key = jax.random.fold_in(key, step)
        action = sample_action(action_key, params, obs)
        
        state, obs, reward, done, info = jitted_step(state, action, config)
        
        print(f"Step {step}: Reward {reward:.3f}, Similarity {info['similarity']:.3f}")
        
        if done:
            break
    
    print("RL integration example completed")

jax_rl_integration_example()
```

### Distributed Training Setup

```python
def distributed_training_setup():
    """Setup for distributed training across multiple processes."""
    
    # Configuration for distributed training
    training_config = {
        'num_workers': 4,
        'episodes_per_worker': 100,
        'sync_frequency': 10,
    }
    
    def worker_training_loop(worker_id, episodes):
        """Training loop for a single worker."""
        worker_key = jax.random.PRNGKey(worker_id)
        config = create_standard_config(max_episode_steps=30)
        
        worker_rewards = []
        
        for episode in range(episodes):
            episode_key = jax.random.fold_in(worker_key, episode)
            task = parser.get_random_task(episode_key)
            
            state, obs = jitted_reset(episode_key, config, task)
            episode_reward = 0.0
            
            for step in range(config.max_episode_steps):
                action = {
                    "selection": jnp.ones((1, 1), dtype=jnp.bool_),
                    "operation": jnp.array((step % 5) + 1, dtype=jnp.int32),
                }
                state, obs, reward, done, info = jitted_step(state, action, config)
                episode_reward += reward
                if done:
                    break
            
            worker_rewards.append(episode_reward)
            
            if episode % 20 == 0:
                avg_reward = jnp.mean(jnp.array(worker_rewards[-20:]))
                print(f"Worker {worker_id}, Episode {episode}: Avg reward {avg_reward:.3f}")
        
        return worker_rewards
    
    # Simulate distributed training
    print("Distributed training simulation:")
    all_rewards = []
    
    for worker_id in range(training_config['num_workers']):
        worker_rewards = worker_training_loop(
            worker_id, 
            training_config['episodes_per_worker']
        )
        all_rewards.extend(worker_rewards)
    
    overall_performance = jnp.mean(jnp.array(all_rewards))
    print(f"Overall training performance: {overall_performance:.3f}")

distributed_training_setup()
```

## Performance Optimization

### Memory-Efficient Batch Processing

```python
def memory_efficient_processing():
    """Demonstrate memory-efficient batch processing techniques."""
    
    def process_batch_chunked(tasks, chunk_size=10):
        """Process large batches in chunks to manage memory."""
        total_rewards = []
        config = create_standard_config(max_episode_steps=20)
        
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            chunk_rewards = []
            
            for j, task in enumerate(chunk):
                key = jax.random.PRNGKey(i + j)
                state, obs = jitted_reset(key, config, task)
                
                # Short episode
                total_reward = 0.0
                for step in range(5):
                    action = {
                        "selection": jnp.ones((1, 1), dtype=jnp.bool_),
                        "operation": jnp.array(1, dtype=jnp.int32),
                    }
                    state, obs, reward, done, info = jitted_step(state, action, config)
                    total_reward += reward
                    if done:
                        break
                
                chunk_rewards.append(total_reward)
            
            total_rewards.extend(chunk_rewards)
            print(f"Processed chunk {i//chunk_size + 1}, avg reward: {jnp.mean(jnp.array(chunk_rewards)):.3f}")
        
        return total_rewards
    
    # Create batch of tasks
    batch_size = 50
    tasks = []
    for i in range(batch_size):
        key = jax.random.PRNGKey(i)
        task = parser.get_random_task(key)
        tasks.append(task)
    
    # Process in chunks
    rewards = process_batch_chunked(tasks, chunk_size=10)
    print(f"Total batch performance: {jnp.mean(jnp.array(rewards)):.3f}")

memory_efficient_processing()
```

## Next Steps

- **[Basic Usage](basic-usage.md)**: Master the fundamentals
- **[Configuration Guide](../configuration.md)**: Deep dive into configuration options
- **[API Reference](../api_reference.md)**: Complete API documentation
- **JAX Documentation**: Learn more about JAX transformations and optimization