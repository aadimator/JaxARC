# JaxARC Benchmarking System

This directory contains benchmarking tools for measuring and comparing JaxARC performance across different optimization phases.

## Files

- `baseline_benchmark.py` - Main benchmark script with configurable descriptions and tags
- `compare_benchmarks.py` - Utility for comparing benchmark results across runs
- `results/` - Directory containing benchmark result JSON files

## Usage

### Running Benchmarks

```bash
# Basic benchmark
pixi run python benchmarks/baseline_benchmark.py

# Benchmark with description and tag
pixi run python benchmarks/baseline_benchmark.py \
  --description "Phase 1: Removed callbacks, added StepInfo" \
  --tag "phase1"

# Short form
pixi run python benchmarks/baseline_benchmark.py \
  -d "Phase2-JAX-Native-Error-Handling" \
  -t "phase2"
```

### Comparing Results

```bash
# Compare all benchmark results
pixi run python benchmarks/compare_benchmarks.py

# Compare specific batch size
pixi run python benchmarks/compare_benchmarks.py --batch-size 500

# Show only latest batch scaling
pixi run python benchmarks/compare_benchmarks.py --scaling-only
```

## Benchmark Output

### Single Environment Metrics
- **Steps Per Second (SPS)**: Number of environment steps processed per second
- **Reset Time**: Average time for environment reset operations

### Batch Environment Metrics
- **Batch SPS**: Total steps per second across all environments in batch
- **Scaling Efficiency**: How well performance scales with batch size
- **Memory Usage**: Memory consumption patterns

### System Information
- Platform details (OS, architecture)
- JAX version and device information
- Hardware specifications

## Result Files

Benchmark results are saved as JSON files in `benchmarks/results/` with the format:
- `benchmark_YYYYMMDD_HHMMSS.json` (basic)
- `benchmark_TAG_YYYYMMDD_HHMMSS.json` (with tag)

Each result file contains:
```json
{
  "timestamp": 1691234567.89,
  "description": "Phase 1: Removed callbacks, added StepInfo",
  "tag": "phase1",
  "system_info": { ... },
  "config_summary": { ... },
  "single_environment": {
    "steps_per_second": 415,
    "avg_reset_time": 0.002
  },
  "batch_scaling": {
    "1": {"steps_per_second": 30, "batch_size": 1},
    "10": {"steps_per_second": 293, "batch_size": 10},
    ...
  }
}
```

## Performance Tracking

Use descriptions and tags to track optimization phases:

- **Baseline**: `--tag "baseline" --description "Initial implementation"`
- **Phase 1**: `--tag "phase1" --description "Removed callbacks, added StepInfo"`
- **Phase 2**: `--tag "phase2" --description "JAX-native error handling"`
- **Phase 3**: `--tag "phase3" --description "Memory optimizations"`

## Expected Performance Targets

Based on the implementation plan:

| Metric | Current | Target | Phase |
|--------|---------|--------|-------|
| Single Environment SPS | 400-600 | 50K-100K | Phase 2-3 |
| Batch 1000 SPS | 15K-20K | 1M-5M | Phase 2-3 |
| Memory Efficiency | Baseline | 2-5x better | Phase 3 |

## Tips

1. **Consistent Environment**: Run benchmarks on the same machine with similar system load
2. **Multiple Runs**: Performance can vary; consider running multiple benchmarks for averaging
3. **Tag Everything**: Use descriptive tags and descriptions for easy comparison
4. **Track Changes**: Document what changed between benchmark runs
5. **System Load**: Close other applications during benchmarking for consistent results