# Stoix Integration

Train RL agents on ARC puzzles using [Stoix](https://github.com/EdanToledo/Stoix),
a JAX-based RL framework.

## Factory Function

JaxARC provides `make_jaxarc_env()` which creates Stoix-compatible environments
with the correct wrapper stack:

```python
from omegaconf import OmegaConf
from jaxarc.stoix_adapter import make_jaxarc_env

config = OmegaConf.create({
    "env": {
        "scenario": {"name": "Mini"},
        "action": {"mode": "point"},
        "observation_wrappers": {
            "answer_grid": True,
            "input_grid": True,
            "contextual": True,
        },
    }
})

train_env, eval_env = make_jaxarc_env(config)
```

The factory:
1. Creates JaxARC environments via `jaxarc.make()`
2. Applies action wrappers (`PointActionWrapper` / `BboxActionWrapper` + `FlattenActionWrapper`)
3. Applies observation wrappers (answer grid, input grid, contextual)

## ExtendedMetrics

JaxARC's `ExtendedMetrics` wrapper tracks domain-specific episode statistics:

| Metric | Description |
|---|---|
| `best_similarity` | Highest grid similarity achieved during the episode |
| `solved` | Whether the puzzle was solved (similarity = 1.0) |
| `steps_to_solve` | Number of steps taken to solve (0 if unsolved) |
| `final_similarity` | Grid similarity at episode end |
| `was_truncated` | Whether the episode was truncated (hit max steps) |

When used with Stoix, `ExtendedMetrics` should be applied **after**
`RecordEpisodeMetrics` so its fields merge into the episode metrics dict:

```python
from stoa.core_wrappers.wrapper import AddRNGKey
from stoa.core_wrappers.episode_metrics import RecordEpisodeMetrics
from stoa.core_wrappers.auto_reset import AutoResetWrapper
from jaxarc.wrappers import ExtendedMetrics

env = AddRNGKey(env)
env = RecordEpisodeMetrics(env)
env = ExtendedMetrics(env)        # After REM, not before!
env = AutoResetWrapper(env, next_obs_in_extras=True)
```

## Custom Metrics Processing

`jaxarc_custom_metrics()` computes derived statistics from raw episode metrics:

```python
from jaxarc.stoix_adapter import jaxarc_custom_metrics

# raw_metrics comes from timestep.extras["episode_metrics"]
processed = jaxarc_custom_metrics(raw_metrics)
# Adds: success_rate, avg_steps_to_solve, truncation_rate,
#        best_similarity_mean, final_similarity_mean
```

## Using with jaxarc-baselines

The [jaxarc-baselines](https://github.com/aadimator/jaxarc-baselines) repository
provides ready-to-use experiment configs:

```bash
git clone --recurse-submodules https://github.com/aadimator/jaxarc-baselines
cd jaxarc-baselines
pixi install
pixi run python run_experiment.py
```

Override configuration from the command line:

```bash
pixi run python run_experiment.py \
    env.scenario.name=Mini-easy \
    env.action.mode=point \
    arch.total_num_envs=256
```
