# ConceptARC Examples

ConceptARC-specific usage patterns for systematic evaluation and concept-based analysis.

## Concept Group Exploration

### Available Concepts

```python
from jaxarc.parsers import ConceptArcParser
from omegaconf import DictConfig

# Create ConceptARC configuration
config = DictConfig({
    "corpus": {"path": "data/raw/ConceptARC/corpus"},
    "grid": {"max_grid_height": 30, "max_grid_width": 30},
    "max_train_pairs": 4,
    "max_test_pairs": 3,
})

parser = ConceptArcParser(config)
concepts = parser.get_concept_groups()
print(f"Available concepts: {concepts}")
# Output: ['AboveBelow', 'Center', 'CleanUp', 'CompleteShape', ...]
```

### Concept-Specific Task Loading

```python
import jax

key = jax.random.PRNGKey(42)

# Load task from specific concept
task = parser.get_random_task_from_concept("Center", key)
print(f"Center concept task: {task.num_train_pairs} training pairs")

# Load multiple tasks from same concept
tasks = []
for i in range(5):
    key, subkey = jax.random.split(key)
    task = parser.get_random_task_from_concept("Center", subkey)
    tasks.append(task)

print(f"Loaded {len(tasks)} tasks from Center concept")
```

## Systematic Evaluation Workflows

### Evaluate Across All Concepts

```python
from jaxarc.envs import create_standard_config, arc_reset, arc_step
import jax.numpy as jnp

def evaluate_concept(concept_name, num_tasks=5):
    """Evaluate performance on a specific concept."""
    config = create_standard_config(max_episode_steps=50)
    rewards = []
    
    for i in range(num_tasks):
        key = jax.random.PRNGKey(i)
        task = parser.get_random_task_from_concept(concept_name, key)
        
        # Simple evaluation episode
        state, obs = arc_reset(key, config, task)
        total_reward = 0.0
        
        for step in range(10):  # Short evaluation
            action = {
                "selection": jnp.ones((2, 2), dtype=jnp.bool_),
                "operation": jnp.array(1, dtype=jnp.int32),
            }
            state, obs, reward, done, info = arc_step(state, action, config)
            total_reward += reward
            if done:
                break
        
        rewards.append(total_reward)
    
    return jnp.array(rewards)

# Evaluate all concepts
concept_results = {}
for concept in concepts[:3]:  # First 3 concepts for demo
    rewards = evaluate_concept(concept)
    concept_results[concept] = {
        'mean_reward': jnp.mean(rewards),
        'std_reward': jnp.std(rewards),
    }
    print(f"{concept}: {jnp.mean(rewards):.3f} ± {jnp.std(rewards):.3f}")
```

### Concept Difficulty Analysis

```python
def analyze_concept_difficulty():
    """Analyze relative difficulty across concepts."""
    difficulty_metrics = {}
    
    for concept in concepts:
        # Sample tasks from concept
        tasks = []
        for i in range(10):
            key = jax.random.PRNGKey(i)
            task = parser.get_random_task_from_concept(concept, key)
            tasks.append(task)
        
        # Calculate difficulty metrics
        grid_sizes = [task.train_input_grids.shape[-2:] for task in tasks]
        avg_grid_size = jnp.mean(jnp.array([h*w for h, w in grid_sizes]))
        avg_train_pairs = jnp.mean(jnp.array([task.num_train_pairs for task in tasks]))
        
        difficulty_metrics[concept] = {
            'avg_grid_size': avg_grid_size,
            'avg_train_pairs': avg_train_pairs,
        }
    
    # Sort by difficulty (larger grids = harder)
    sorted_concepts = sorted(
        difficulty_metrics.items(), 
        key=lambda x: x[1]['avg_grid_size']
    )
    
    print("Concepts by difficulty (grid size):")
    for concept, metrics in sorted_concepts:
        print(f"{concept}: {metrics['avg_grid_size']:.1f} avg cells")

analyze_concept_difficulty()
```

## Concept-Specific Training Patterns

### Curriculum Learning by Concept

```python
def create_concept_curriculum():
    """Create a curriculum ordered by concept difficulty."""
    # Define concept progression (easy to hard)
    curriculum = [
        "Copy",           # Simple copying tasks
        "Center",         # Spatial reasoning
        "SameDifferent",  # Pattern comparison
        "CompleteShape",  # Shape completion
        "ExtractObjects", # Object manipulation
    ]
    
    return curriculum

def train_with_curriculum(curriculum, steps_per_concept=100):
    """Train using concept-based curriculum."""
    config = create_standard_config(max_episode_steps=50)
    
    for concept in curriculum:
        print(f"Training on concept: {concept}")
        
        for step in range(steps_per_concept):
            key = jax.random.PRNGKey(step)
            task = parser.get_random_task_from_concept(concept, key)
            
            # Training episode
            state, obs = arc_reset(key, config, task)
            # ... training logic here
            
        print(f"Completed {steps_per_concept} steps on {concept}")

curriculum = create_concept_curriculum()
# train_with_curriculum(curriculum)  # Uncomment to run
```

### Concept-Specific Action Strategies

```python
def get_concept_strategy(concept_name):
    """Get recommended action strategy for specific concepts."""
    strategies = {
        "Center": {
            "description": "Focus on central regions",
            "preferred_operations": [1, 2, 3],  # Fill colors
            "selection_pattern": "center_focused",
        },
        "Copy": {
            "description": "Direct copying operations", 
            "preferred_operations": [29, 30],  # Clipboard operations
            "selection_pattern": "full_grid",
        },
        "CompleteShape": {
            "description": "Shape completion and filling",
            "preferred_operations": [10, 11, 12],  # Flood fill
            "selection_pattern": "shape_boundary",
        },
    }
    
    return strategies.get(concept_name, {
        "description": "General strategy",
        "preferred_operations": [1, 2, 3, 10, 11],
        "selection_pattern": "adaptive",
    })

# Example usage
strategy = get_concept_strategy("Center")
print(f"Center strategy: {strategy['description']}")
print(f"Preferred operations: {strategy['preferred_operations']}")
```

## Performance Analysis

### Concept Performance Comparison

```python
def compare_concept_performance():
    """Compare performance across different concepts."""
    import matplotlib.pyplot as plt
    
    results = {}
    test_concepts = ["Center", "Copy", "SameDifferent", "CompleteShape"]
    
    for concept in test_concepts:
        rewards = evaluate_concept(concept, num_tasks=10)
        results[concept] = rewards
    
    # Statistical comparison
    for concept, rewards in results.items():
        print(f"{concept}:")
        print(f"  Mean: {jnp.mean(rewards):.3f}")
        print(f"  Std:  {jnp.std(rewards):.3f}")
        print(f"  Min:  {jnp.min(rewards):.3f}")
        print(f"  Max:  {jnp.max(rewards):.3f}")
        print()

# compare_concept_performance()  # Uncomment to run
```

### Concept Learning Curves

```python
def track_concept_learning(concept_name, num_episodes=100):
    """Track learning progress on a specific concept."""
    config = create_standard_config(max_episode_steps=30)
    rewards = []
    
    for episode in range(num_episodes):
        key = jax.random.PRNGKey(episode)
        task = parser.get_random_task_from_concept(concept_name, key)
        
        state, obs = arc_reset(key, config, task)
        episode_reward = 0.0
        
        # Simple policy (can be replaced with actual learning)
        for step in range(10):
            action = {
                "selection": jnp.ones((1, 1), dtype=jnp.bool_),
                "operation": jnp.array((step % 5) + 1, dtype=jnp.int32),
            }
            state, obs, reward, done, info = arc_step(state, action, config)
            episode_reward += reward
            if done:
                break
        
        rewards.append(episode_reward)
        
        # Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            recent_avg = jnp.mean(jnp.array(rewards[-20:]))
            print(f"Episode {episode+1}: Recent avg reward = {recent_avg:.3f}")
    
    return jnp.array(rewards)

# Track learning on Center concept
# learning_curve = track_concept_learning("Center", 50)
```

## Advanced ConceptARC Usage

### Multi-Concept Task Sampling

```python
def sample_mixed_concepts(concepts, num_tasks_per_concept=5):
    """Sample tasks from multiple concepts for diverse training."""
    all_tasks = []
    
    for concept in concepts:
        for i in range(num_tasks_per_concept):
            key = jax.random.PRNGKey(hash(concept) + i)
            task = parser.get_random_task_from_concept(concept, key)
            all_tasks.append((concept, task))
    
    # Shuffle for mixed training
    import random
    random.shuffle(all_tasks)
    
    return all_tasks

# Sample from multiple concepts
mixed_tasks = sample_mixed_concepts(["Center", "Copy", "SameDifferent"], 3)
print(f"Created mixed dataset with {len(mixed_tasks)} tasks")
for i, (concept, task) in enumerate(mixed_tasks[:5]):
    print(f"Task {i}: {concept} concept, {task.num_train_pairs} pairs")
```

### Concept Transfer Learning

```python
def evaluate_concept_transfer(source_concept, target_concept):
    """Evaluate transfer learning between concepts."""
    print(f"Evaluating transfer from {source_concept} to {target_concept}")
    
    # Train on source concept
    source_performance = evaluate_concept(source_concept, num_tasks=10)
    print(f"Source performance: {jnp.mean(source_performance):.3f}")
    
    # Test on target concept  
    target_performance = evaluate_concept(target_concept, num_tasks=10)
    print(f"Target performance: {jnp.mean(target_performance):.3f}")
    
    # Calculate transfer score (simplified)
    transfer_score = jnp.mean(target_performance) / jnp.mean(source_performance)
    print(f"Transfer score: {transfer_score:.3f}")
    
    return transfer_score

# Example transfer evaluation
# transfer_score = evaluate_concept_transfer("Copy", "Center")
```

## Next Steps

- **[MiniARC Examples](miniarc-examples.md)**: Learn rapid prototyping with smaller grids
- **[Advanced Patterns](advanced-patterns.md)**: Master batch processing and JAX transformations
- **[Configuration Guide](../configuration.md)**: Optimize configurations for ConceptARC
- **[Datasets Guide](../datasets.md)**: Complete ConceptARC dataset documentation