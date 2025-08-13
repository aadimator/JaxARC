# JaxARC Baseline Performance Results

**Date**: August 12, 2025  
**System**: macOS 15.5 (ARM64), Python 3.13.5, JAX 0.6.2  
**Hardware**: Apple Silicon (CPU only)

## Summary

This document captures the baseline performance metrics for the current JaxARC implementation before applying the optimizations outlined in the performance implementation plan.

## Key Findings

### 1. Single Environment Performance
- **366 SPS** (Steps Per Second)
- Average reset time: 0.52ms
- Total time for 1000 steps: 2.73 seconds

### 2. Batch Environment Scaling
| Batch Size | Steps Per Second | Speedup vs Single | Efficiency |
|------------|------------------|-------------------|------------|
| 1          | 26 SPS          | 0.07x             | 7%         |
| 10         | 92 SPS          | 0.25x             | 2.5%       |
| 50         | 868 SPS         | 2.37x             | 4.7%       |
| 100        | 1,532 SPS       | 4.18x             | 4.2%       |
| 500        | 2,665 SPS       | 7.28x             | 1.5%       |
| 1000       | 2,542 SPS       | 6.94x             | 0.7%       |

### 3. Performance Bottlenecks Identified

1. **Logging Overhead**: Even with `log_operations=False`, there are still `jax.debug.callback` calls in the hot path
2. **Batch Inefficiency**: Single environment (366 SPS) is much faster than batch size 1 (26 SPS), indicating overhead in batch processing
3. **Scaling Issues**: Efficiency drops significantly with larger batch sizes, suggesting memory or compilation bottlenecks

## Configuration Used

- Grid size: 30x30
- Max train pairs: 3
- Selection format: point
- Logging: disabled
- History: disabled

## Comparison with Planning Document Targets

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Single Environment SPS | 366 | 50K-100K | 136-273x improvement needed |
| Batch SPS (1000 envs) | 2,542 | 1M-5M | 393-1,967x improvement needed |

## Next Steps

Based on these baseline results, the optimization plan should focus on:

1. **Remove `jax.debug.callback` calls** - This is the #1 performance bottleneck
2. **Fix batch processing overhead** - Single env is 14x faster than batch size 1
3. **Improve memory efficiency** - Batch scaling efficiency drops significantly
4. **JAX-native error handling** - Replace try/except blocks with `jax.lax.cond`

The current results confirm the bottlenecks identified in the planning document and provide a solid baseline for measuring optimization improvements.