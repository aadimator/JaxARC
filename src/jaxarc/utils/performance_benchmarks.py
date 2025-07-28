"""
Performance benchmarking suite for JaxARC (Task 6.1).

This module provides comprehensive performance benchmarking capabilities including:
- JIT compilation timing tests
- Step execution performance measurement
- Batch processing scalability analysis
- Memory usage profiling for different configurations
- Automated performance regression testing

The benchmarking suite is designed to validate the performance improvements
achieved through JAX compatibility fixes and provide ongoing performance monitoring.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import psutil

from ..envs.config import JaxArcConfig
from ..envs.functional import arc_reset, arc_step, batch_reset, batch_step
from ..envs.structured_actions import PointAction, BboxAction, MaskAction
from ..state import ArcEnvState
from ..types import JaxArcTask


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    memory_usage_mb: float
    throughput: float  # operations per second
    metadata: Dict[str, Any]


@dataclass
class MemoryProfile:
    """Container for memory profiling results."""
    peak_memory_mb: float
    current_memory_mb: float
    memory_delta_mb: float
    gc_collections: int
    metadata: Dict[str, Any]


class PerformanceBenchmarks:
    """Comprehensive performance benchmarking suite for JaxARC.
    
    This class provides methods to benchmark various aspects of JaxARC performance:
    - JIT compilation timing
    - Step execution performance
    - Batch processing scalability
    - Memory usage profiling
    - Performance regression testing
    
    Examples:
        ```python
        # Create benchmarking suite
        benchmarks = PerformanceBenchmarks(config, task_data)
        
        # Run JIT compilation benchmarks
        jit_results = benchmarks.benchmark_jit_compilation()
        
        # Run batch processing benchmarks
        batch_results = benchmarks.benchmark_batch_processing()
        
        # Run comprehensive performance suite
        all_results = benchmarks.run_comprehensive_benchmarks()
        
        # Generate performance report
        benchmarks.generate_performance_report(all_results, "performance_report.json")
        ```
    """
    
    def __init__(
        self,
        config: JaxArcConfig,
        task_data: JaxArcTask,
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        memory_profiling: bool = True
    ):
        """Initialize performance benchmarking suite.
        
        Args:
            config: JaxARC configuration for benchmarking
            task_data: Task data to use for benchmarking
            warmup_iterations: Number of warmup iterations before timing
            benchmark_iterations: Number of iterations for timing measurements
            memory_profiling: Whether to enable memory profiling
        """
        self.config = config
        self.task_data = task_data
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.memory_profiling = memory_profiling
        
        # Initialize process for memory monitoring
        self.process = psutil.Process()
        
        # Pre-compile functions for accurate timing
        self._warmup_functions()
    
    def _warmup_functions(self) -> None:
        """Warm up JAX functions to ensure JIT compilation is complete."""
        key = jrandom.PRNGKey(42)
        
        # Warm up single environment functions
        state, obs = arc_reset(key, self.config, self.task_data)
        action = PointAction(
            operation=jnp.array(0, dtype=jnp.int32),
            row=jnp.array(2, dtype=jnp.int32),
            col=jnp.array(2, dtype=jnp.int32)
        )
        arc_step(state, action, self.config)
        
        # Warm up batch functions
        keys = jrandom.split(key, 4)
        batch_states, batch_obs = batch_reset(keys, self.config, self.task_data)
        batch_actions = PointAction(
            operation=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
            row=jnp.array([2, 2, 2, 2], dtype=jnp.int32),
            col=jnp.array([2, 2, 2, 2], dtype=jnp.int32)
        )
        batch_step(batch_states, batch_actions, self.config)
    
    def _start_memory_profiling(self) -> Tuple[float, int]:
        """Start memory profiling and return baseline measurements."""
        if self.memory_profiling:
            tracemalloc.start()
            gc.collect()  # Clean up before measurement
        
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        baseline_gc = len(gc.get_stats())
        
        return baseline_memory, baseline_gc
    
    def _stop_memory_profiling(self, baseline_memory: float, baseline_gc: int) -> MemoryProfile:
        """Stop memory profiling and return memory usage statistics."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_gc = len(gc.get_stats())
        
        peak_memory = current_memory
        if self.memory_profiling and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = max(peak_memory, peak / 1024 / 1024)  # MB
            tracemalloc.stop()
        
        return MemoryProfile(
            peak_memory_mb=peak_memory,
            current_memory_mb=current_memory,
            memory_delta_mb=current_memory - baseline_memory,
            gc_collections=current_gc - baseline_gc,
            metadata={}
        )
    
    def benchmark_jit_compilation(self) -> Dict[str, BenchmarkResult]:
        """Benchmark JIT compilation performance for core functions.
        
        Tests the time required for JIT compilation of arc_reset, arc_step,
        and batch processing functions. This validates that JIT compilation
        is working correctly and measures compilation overhead.
        
        Returns:
            Dictionary mapping function names to BenchmarkResult objects
        """
        results = {}
        
        # Benchmark arc_reset JIT compilation
        baseline_memory, baseline_gc = self._start_memory_profiling()
        
        compilation_times = []
        for i in range(self.benchmark_iterations):
            key = jrandom.PRNGKey(42 + i)
            
            # Clear JIT cache to force recompilation
            jax.clear_caches()
            
            start_time = time.perf_counter()
            state, obs = arc_reset(key, self.config, self.task_data)
            jax.block_until_ready(state.working_grid)  # Ensure computation is complete
            end_time = time.perf_counter()
            
            compilation_times.append(end_time - start_time)
        
        memory_profile = self._stop_memory_profiling(baseline_memory, baseline_gc)
        
        results['arc_reset_jit'] = BenchmarkResult(
            name='arc_reset_jit_compilation',
            mean_time=float(jnp.mean(jnp.array(compilation_times))),
            std_time=float(jnp.std(jnp.array(compilation_times))),
            min_time=float(jnp.min(jnp.array(compilation_times))),
            max_time=float(jnp.max(jnp.array(compilation_times))),
            iterations=self.benchmark_iterations,
            memory_usage_mb=memory_profile.peak_memory_mb,
            throughput=1.0 / float(jnp.mean(jnp.array(compilation_times))),
            metadata={'memory_profile': memory_profile}
        )
        
        # Benchmark arc_step JIT compilation
        baseline_memory, baseline_gc = self._start_memory_profiling()
        
        compilation_times = []
        for i in range(self.benchmark_iterations):
            key = jrandom.PRNGKey(42 + i)
            state, obs = arc_reset(key, self.config, self.task_data)
            action = PointAction(
                operation=jnp.array(0, dtype=jnp.int32),
                row=jnp.array(2, dtype=jnp.int32),
                col=jnp.array(2, dtype=jnp.int32)
            )
            
            # Clear JIT cache to force recompilation
            jax.clear_caches()
            
            start_time = time.perf_counter()
            new_state, new_obs, reward, done, info = arc_step(state, action, self.config)
            jax.block_until_ready(new_state.working_grid)
            end_time = time.perf_counter()
            
            compilation_times.append(end_time - start_time)
        
        memory_profile = self._stop_memory_profiling(baseline_memory, baseline_gc)
        
        results['arc_step_jit'] = BenchmarkResult(
            name='arc_step_jit_compilation',
            mean_time=float(jnp.mean(jnp.array(compilation_times))),
            std_time=float(jnp.std(jnp.array(compilation_times))),
            min_time=float(jnp.min(jnp.array(compilation_times))),
            max_time=float(jnp.max(jnp.array(compilation_times))),
            iterations=self.benchmark_iterations,
            memory_usage_mb=memory_profile.peak_memory_mb,
            throughput=1.0 / float(jnp.mean(jnp.array(compilation_times))),
            metadata={'memory_profile': memory_profile}
        )
        
        return results
    
    def benchmark_step_execution(self) -> Dict[str, BenchmarkResult]:
        """Benchmark step execution performance after JIT compilation.
        
        Measures the execution time of arc_reset and arc_step after JIT
        compilation is complete. This validates the 100x+ performance
        improvement target from the requirements.
        
        Returns:
            Dictionary mapping function names to BenchmarkResult objects
        """
        results = {}
        
        # Benchmark arc_reset execution (after JIT)
        baseline_memory, baseline_gc = self._start_memory_profiling()
        
        # Warmup
        for _ in range(self.warmup_iterations):
            key = jrandom.PRNGKey(42)
            arc_reset(key, self.config, self.task_data)
        
        execution_times = []
        for i in range(self.benchmark_iterations):
            key = jrandom.PRNGKey(42 + i)
            
            start_time = time.perf_counter()
            state, obs = arc_reset(key, self.config, self.task_data)
            jax.block_until_ready(state.working_grid)
            end_time = time.perf_counter()
            
            execution_times.append(end_time - start_time)
        
        memory_profile = self._stop_memory_profiling(baseline_memory, baseline_gc)
        
        results['arc_reset_execution'] = BenchmarkResult(
            name='arc_reset_execution',
            mean_time=float(jnp.mean(jnp.array(execution_times))),
            std_time=float(jnp.std(jnp.array(execution_times))),
            min_time=float(jnp.min(jnp.array(execution_times))),
            max_time=float(jnp.max(jnp.array(execution_times))),
            iterations=self.benchmark_iterations,
            memory_usage_mb=memory_profile.peak_memory_mb,
            throughput=1.0 / float(jnp.mean(jnp.array(execution_times))),
            metadata={'memory_profile': memory_profile}
        )
        
        # Benchmark arc_step execution (after JIT)
        baseline_memory, baseline_gc = self._start_memory_profiling()
        
        # Prepare state and action for step benchmarking
        key = jrandom.PRNGKey(42)
        state, obs = arc_reset(key, self.config, self.task_data)
        action = PointAction(
            operation=jnp.array(0, dtype=jnp.int32),
            row=jnp.array(2, dtype=jnp.int32),
            col=jnp.array(2, dtype=jnp.int32)
        )
        
        # Warmup
        for _ in range(self.warmup_iterations):
            arc_step(state, action, self.config)
        
        execution_times = []
        for i in range(self.benchmark_iterations):
            start_time = time.perf_counter()
            new_state, new_obs, reward, done, info = arc_step(state, action, self.config)
            jax.block_until_ready(new_state.working_grid)
            end_time = time.perf_counter()
            
            execution_times.append(end_time - start_time)
        
        memory_profile = self._stop_memory_profiling(baseline_memory, baseline_gc)
        
        results['arc_step_execution'] = BenchmarkResult(
            name='arc_step_execution',
            mean_time=float(jnp.mean(jnp.array(execution_times))),
            std_time=float(jnp.std(jnp.array(execution_times))),
            min_time=float(jnp.min(jnp.array(execution_times))),
            max_time=float(jnp.max(jnp.array(execution_times))),
            iterations=self.benchmark_iterations,
            memory_usage_mb=memory_profile.peak_memory_mb,
            throughput=1.0 / float(jnp.mean(jnp.array(execution_times))),
            metadata={'memory_profile': memory_profile}
        )
        
        return results
    
    def benchmark_batch_processing(self, batch_sizes: Optional[List[int]] = None) -> Dict[str, BenchmarkResult]:
        """Benchmark batch processing scalability.
        
        Tests batch processing performance across different batch sizes to
        validate linear scaling and 10,000+ steps/second throughput capability.
        
        Args:
            batch_sizes: List of batch sizes to test. Defaults to [1, 8, 32, 128, 512]
        
        Returns:
            Dictionary mapping batch size descriptions to BenchmarkResult objects
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 128, 512]
        
        results = {}
        
        for batch_size in batch_sizes:
            # Benchmark batch_reset
            baseline_memory, baseline_gc = self._start_memory_profiling()
            
            # Warmup
            for _ in range(self.warmup_iterations):
                keys = jrandom.split(jrandom.PRNGKey(42), batch_size)
                batch_reset(keys, self.config, self.task_data)
            
            execution_times = []
            for i in range(self.benchmark_iterations):
                keys = jrandom.split(jrandom.PRNGKey(42 + i), batch_size)
                
                start_time = time.perf_counter()
                batch_states, batch_obs = batch_reset(keys, self.config, self.task_data)
                jax.block_until_ready(batch_states.working_grid)
                end_time = time.perf_counter()
                
                execution_times.append(end_time - start_time)
            
            memory_profile = self._stop_memory_profiling(baseline_memory, baseline_gc)
            
            mean_time = float(jnp.mean(jnp.array(execution_times)))
            per_env_time = mean_time / batch_size
            throughput = batch_size / mean_time
            
            results[f'batch_reset_{batch_size}'] = BenchmarkResult(
                name=f'batch_reset_batch_size_{batch_size}',
                mean_time=mean_time,
                std_time=float(jnp.std(jnp.array(execution_times))),
                min_time=float(jnp.min(jnp.array(execution_times))),
                max_time=float(jnp.max(jnp.array(execution_times))),
                iterations=self.benchmark_iterations,
                memory_usage_mb=memory_profile.peak_memory_mb,
                throughput=throughput,
                metadata={
                    'batch_size': batch_size,
                    'per_env_time': per_env_time,
                    'memory_profile': memory_profile
                }
            )
            
            # Benchmark batch_step
            baseline_memory, baseline_gc = self._start_memory_profiling()
            
            # Prepare batch state and actions
            keys = jrandom.split(jrandom.PRNGKey(42), batch_size)
            batch_states, _ = batch_reset(keys, self.config, self.task_data)
            batch_actions = PointAction(
                operation=jnp.zeros(batch_size, dtype=jnp.int32),
                row=jnp.full(batch_size, 2, dtype=jnp.int32),
                col=jnp.full(batch_size, 2, dtype=jnp.int32)
            )
            
            # Warmup
            for _ in range(self.warmup_iterations):
                batch_step(batch_states, batch_actions, self.config)
            
            execution_times = []
            for i in range(self.benchmark_iterations):
                start_time = time.perf_counter()
                new_states, new_obs, rewards, dones, infos = batch_step(
                    batch_states, batch_actions, self.config
                )
                jax.block_until_ready(new_states.working_grid)
                end_time = time.perf_counter()
                
                execution_times.append(end_time - start_time)
            
            memory_profile = self._stop_memory_profiling(baseline_memory, baseline_gc)
            
            mean_time = float(jnp.mean(jnp.array(execution_times)))
            per_env_time = mean_time / batch_size
            throughput = batch_size / mean_time
            
            results[f'batch_step_{batch_size}'] = BenchmarkResult(
                name=f'batch_step_batch_size_{batch_size}',
                mean_time=mean_time,
                std_time=float(jnp.std(jnp.array(execution_times))),
                min_time=float(jnp.min(jnp.array(execution_times))),
                max_time=float(jnp.max(jnp.array(execution_times))),
                iterations=self.benchmark_iterations,
                memory_usage_mb=memory_profile.peak_memory_mb,
                throughput=throughput,
                metadata={
                    'batch_size': batch_size,
                    'per_env_time': per_env_time,
                    'memory_profile': memory_profile
                }
            )
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, BenchmarkResult]:
        """Benchmark memory usage for different action formats.
        
        Tests memory usage for point, bbox, and mask action formats to
        validate the 85%+ memory reduction for point/bbox actions.
        
        Returns:
            Dictionary mapping action format names to BenchmarkResult objects
        """
        results = {}
        
        # Test different action formats by creating configs with different selection formats
        action_formats = [
            ('point', 'point', PointAction(
                operation=jnp.array(0, dtype=jnp.int32),
                row=jnp.array(2, dtype=jnp.int32),
                col=jnp.array(2, dtype=jnp.int32)
            )),
            ('bbox', 'bbox', BboxAction(
                operation=jnp.array(1, dtype=jnp.int32),
                r1=jnp.array(1, dtype=jnp.int32),
                c1=jnp.array(1, dtype=jnp.int32),
                r2=jnp.array(3, dtype=jnp.int32),
                c2=jnp.array(3, dtype=jnp.int32)
            )),
            ('mask', 'mask', MaskAction(
                operation=jnp.array(2, dtype=jnp.int32),
                selection=jnp.ones((self.config.dataset.max_grid_height, 
                                  self.config.dataset.max_grid_width), dtype=jnp.bool_)
            ))
        ]
        
        for format_name, selection_format, action in action_formats:
            baseline_memory, baseline_gc = self._start_memory_profiling()
            
            # Create config with specific selection format using eqx.tree_at
            import equinox as eqx
            format_config = eqx.tree_at(
                lambda c: c.action.selection_format,
                self.config,
                selection_format
            )
            
            # Create state with format-specific action history sizing
            key = jrandom.PRNGKey(42)
            state, obs = arc_reset(key, format_config, self.task_data)
            
            # Perform multiple steps to accumulate action history
            current_state = state
            execution_times = []
            
            for i in range(self.benchmark_iterations):
                start_time = time.perf_counter()
                new_state, new_obs, reward, done, info = arc_step(current_state, action, format_config)
                jax.block_until_ready(new_state.working_grid)
                end_time = time.perf_counter()
                
                execution_times.append(end_time - start_time)
                current_state = new_state
            
            memory_profile = self._stop_memory_profiling(baseline_memory, baseline_gc)
            
            # Calculate action history memory usage
            action_history_memory = current_state.action_history.nbytes / 1024 / 1024  # MB
            
            # Calculate action record fields for this format
            from ..utils.jax_types import get_action_record_fields
            record_fields = get_action_record_fields(
                selection_format,
                format_config.dataset.max_grid_height,
                format_config.dataset.max_grid_width
            )
            
            results[f'memory_{format_name}'] = BenchmarkResult(
                name=f'memory_usage_{format_name}_actions',
                mean_time=float(jnp.mean(jnp.array(execution_times))),
                std_time=float(jnp.std(jnp.array(execution_times))),
                min_time=float(jnp.min(jnp.array(execution_times))),
                max_time=float(jnp.max(jnp.array(execution_times))),
                iterations=self.benchmark_iterations,
                memory_usage_mb=memory_profile.peak_memory_mb,
                throughput=1.0 / float(jnp.mean(jnp.array(execution_times))),
                metadata={
                    'action_format': format_name,
                    'selection_format': selection_format,
                    'action_history_memory_mb': action_history_memory,
                    'action_record_fields': record_fields,
                    'memory_profile': memory_profile
                }
            )
        
        return results
    
    def run_comprehensive_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run comprehensive performance benchmarks.
        
        Executes all benchmark categories and returns combined results.
        This is the main entry point for complete performance evaluation.
        
        Returns:
            Dictionary containing all benchmark results
        """
        all_results = {}
        
        print("Running JIT compilation benchmarks...")
        jit_results = self.benchmark_jit_compilation()
        all_results.update(jit_results)
        
        print("Running step execution benchmarks...")
        execution_results = self.benchmark_step_execution()
        all_results.update(execution_results)
        
        print("Running batch processing benchmarks...")
        batch_results = self.benchmark_batch_processing()
        all_results.update(batch_results)
        
        print("Running memory usage benchmarks...")
        memory_results = self.benchmark_memory_usage()
        all_results.update(memory_results)
        
        return all_results
    
    def generate_performance_report(
        self, 
        results: Dict[str, BenchmarkResult], 
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Creates a detailed performance report with analysis and recommendations.
        
        Args:
            results: Benchmark results from run_comprehensive_benchmarks()
            output_path: Optional path to save JSON report
        
        Returns:
            Dictionary containing the performance report
        """
        import json
        from datetime import datetime
        
        # Analyze results
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_grid_height': self.config.dataset.max_grid_height,
                'max_grid_width': self.config.dataset.max_grid_width,
                'max_episode_steps': self.config.environment.max_episode_steps,
            },
            'summary': {},
            'detailed_results': {},
            'performance_analysis': {},
            'recommendations': []
        }
        
        # Process results by category
        jit_results = {k: v for k, v in results.items() if 'jit' in k}
        execution_results = {k: v for k, v in results.items() if 'execution' in k}
        batch_results = {k: v for k, v in results.items() if 'batch' in k}
        memory_results = {k: v for k, v in results.items() if 'memory' in k}
        
        # Summary statistics
        if execution_results:
            step_times = [r.mean_time for r in execution_results.values()]
            report['summary']['mean_step_time_ms'] = float(jnp.mean(jnp.array(step_times)) * 1000)
            report['summary']['step_throughput'] = float(1.0 / jnp.mean(jnp.array(step_times)))
        
        if batch_results:
            batch_throughputs = [r.throughput for r in batch_results.values() if 'step' in r.name]
            if batch_throughputs:
                report['summary']['max_batch_throughput'] = float(jnp.max(jnp.array(batch_throughputs)))
        
        if memory_results:
            memory_usages = [r.metadata.get('action_history_memory_mb', 0) for r in memory_results.values()]
            report['summary']['memory_usage_range_mb'] = [float(jnp.min(jnp.array(memory_usages))), 
                                                         float(jnp.max(jnp.array(memory_usages)))]
        
        # Detailed results
        for name, result in results.items():
            report['detailed_results'][name] = {
                'mean_time_ms': result.mean_time * 1000,
                'std_time_ms': result.std_time * 1000,
                'throughput': result.throughput,
                'memory_usage_mb': result.memory_usage_mb,
                'iterations': result.iterations
            }
        
        # Performance analysis
        report['performance_analysis'] = self._analyze_performance(results)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(results)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _analyze_performance(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance results and identify key metrics."""
        analysis = {}
        
        # Check if 100x improvement target is met (assuming baseline of 10ms)
        execution_results = {k: v for k, v in results.items() if 'execution' in k}
        if execution_results:
            step_times = [r.mean_time for r in execution_results.values()]
            mean_step_time = float(jnp.mean(jnp.array(step_times)))
            analysis['step_performance'] = {
                'mean_time_ms': mean_step_time * 1000,
                'meets_100x_target': mean_step_time < 0.0001,  # <0.1ms for 100x improvement from 10ms
                'improvement_factor': 0.01 / mean_step_time if mean_step_time > 0 else float('inf')
            }
        
        # Check if 10,000+ steps/second target is met
        batch_results = {k: v for k, v in results.items() if 'batch' in k and 'step' in k}
        if batch_results:
            max_throughput = max(r.throughput for r in batch_results.values())
            analysis['batch_performance'] = {
                'max_throughput': max_throughput,
                'meets_10k_target': max_throughput >= 10000,
                'scaling_efficiency': self._calculate_scaling_efficiency(batch_results)
            }
        
        # Check if 85% memory reduction target is met
        memory_results = {k: v for k, v in results.items() if 'memory' in k}
        if memory_results:
            memory_usages = {
                k.split('_')[-1]: v.metadata.get('action_history_memory_mb', 0) 
                for k, v in memory_results.items()
            }
            if 'mask' in memory_usages and 'point' in memory_usages:
                point_reduction = (memory_usages['mask'] - memory_usages['point']) / memory_usages['mask']
                analysis['memory_performance'] = {
                    'point_memory_mb': memory_usages.get('point', 0),
                    'mask_memory_mb': memory_usages.get('mask', 0),
                    'point_reduction_percent': point_reduction * 100,
                    'meets_85_reduction_target': point_reduction >= 0.85
                }
        
        return analysis
    
    def _calculate_scaling_efficiency(self, batch_results: Dict[str, BenchmarkResult]) -> float:
        """Calculate batch processing scaling efficiency."""
        # Extract batch sizes and throughputs
        batch_data = []
        for name, result in batch_results.items():
            if 'batch_size' in result.metadata:
                batch_data.append((result.metadata['batch_size'], result.throughput))
        
        if len(batch_data) < 2:
            return 1.0
        
        # Sort by batch size
        batch_data.sort(key=lambda x: x[0])
        
        # Calculate efficiency as throughput ratio vs batch size ratio
        base_batch_size, base_throughput = batch_data[0]
        max_batch_size, max_throughput = batch_data[-1]
        
        expected_throughput = base_throughput * (max_batch_size / base_batch_size)
        efficiency = max_throughput / expected_throughput
        
        return float(efficiency)
    
    def _generate_recommendations(self, results: Dict[str, BenchmarkResult]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze step execution performance
        execution_results = {k: v for k, v in results.items() if 'execution' in k}
        if execution_results:
            step_times = [r.mean_time for r in execution_results.values()]
            mean_step_time = float(jnp.mean(jnp.array(step_times)))
            
            if mean_step_time > 0.001:  # >1ms
                recommendations.append(
                    "Step execution time is above 1ms. Consider optimizing grid operations "
                    "or reducing grid size for better performance."
                )
            
            if mean_step_time > 0.0001:  # >0.1ms
                recommendations.append(
                    "Step execution has not achieved 100x improvement target. "
                    "Verify JIT compilation is working correctly."
                )
        
        # Analyze batch processing performance
        batch_results = {k: v for k, v in results.items() if 'batch' in k and 'step' in k}
        if batch_results:
            max_throughput = max(r.throughput for r in batch_results.values())
            
            if max_throughput < 10000:
                recommendations.append(
                    f"Batch throughput ({max_throughput:.0f} steps/sec) is below 10,000 target. "
                    "Consider increasing batch size or optimizing batch operations."
                )
            
            scaling_efficiency = self._calculate_scaling_efficiency(batch_results)
            if scaling_efficiency < 0.8:
                recommendations.append(
                    f"Batch scaling efficiency ({scaling_efficiency:.2f}) is below 80%. "
                    "Check for memory bottlenecks or suboptimal vectorization."
                )
        
        # Analyze memory usage
        memory_results = {k: v for k, v in results.items() if 'memory' in k}
        if memory_results:
            memory_usages = {
                k.split('_')[-1]: v.metadata.get('action_history_memory_mb', 0) 
                for k, v in memory_results.items()
            }
            
            if 'mask' in memory_usages and memory_usages['mask'] > 10:
                recommendations.append(
                    f"Mask action memory usage ({memory_usages['mask']:.1f} MB) is high. "
                    "Consider using point or bbox actions when possible."
                )
            
            if 'point' in memory_usages and 'mask' in memory_usages:
                reduction = (memory_usages['mask'] - memory_usages['point']) / memory_usages['mask']
                if reduction < 0.85:
                    recommendations.append(
                        f"Point action memory reduction ({reduction*100:.1f}%) is below 85% target. "
                        "Verify format-specific action history is implemented correctly."
                    )
        
        if not recommendations:
            recommendations.append("Performance is meeting all targets. No optimizations needed.")
        
        return recommendations


def create_benchmark_config() -> JaxArcConfig:
    """Create optimized configuration for benchmarking."""
    from ..envs.config import (
        EnvironmentConfig, DatasetConfig, ActionConfig, RewardConfig,
        VisualizationConfig, StorageConfig, LoggingConfig, WandbConfig, JaxArcConfig
    )
    
    return JaxArcConfig(
        environment=EnvironmentConfig(
            max_episode_steps=50,
            auto_reset=True,
            strict_validation=False,  # Disable for performance
            debug_level="off"
        ),
        dataset=DatasetConfig(
            max_grid_height=20,
            max_grid_width=20,
            max_colors=10,
            background_color=0
        ),
        action=ActionConfig(
            validate_actions=False,  # Disable for performance
            allow_invalid_actions=True
        ),
        reward=RewardConfig(),
        visualization=VisualizationConfig(enabled=False),  # Disable for performance
        storage=StorageConfig(policy="none"),  # Disable for performance
        logging=LoggingConfig(
            log_operations=False,  # Disable for performance
            log_level="ERROR"
        ),
        wandb=WandbConfig(enabled=False)
    )


def create_benchmark_task(config: JaxArcConfig) -> JaxArcTask:
    """Create optimized task for benchmarking."""
    grid_height = min(10, config.dataset.max_grid_height)
    grid_width = min(10, config.dataset.max_grid_width)
    grid_shape = (grid_height, grid_width)

    # Create simple input/target grids for consistent benchmarking
    input_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
    input_grid = input_grid.at[2:5, 2:5].set(1)  # 3x3 pattern

    target_grid = jnp.zeros(grid_shape, dtype=jnp.int32)
    target_grid = target_grid.at[5:8, 5:8].set(2)  # Different pattern

    mask = jnp.ones(grid_shape, dtype=jnp.bool_)

    # Pad to max size
    max_shape = (config.dataset.max_grid_height, config.dataset.max_grid_width)
    padded_input = jnp.full(max_shape, -1, dtype=jnp.int32)
    padded_target = jnp.full(max_shape, -1, dtype=jnp.int32)
    padded_mask = jnp.zeros(max_shape, dtype=jnp.bool_)

    padded_input = padded_input.at[:grid_shape[0], :grid_shape[1]].set(input_grid)
    padded_target = padded_target.at[:grid_shape[0], :grid_shape[1]].set(target_grid)
    padded_mask = padded_mask.at[:grid_shape[0], :grid_shape[1]].set(mask)

    return JaxArcTask(
        input_grids_examples=jnp.expand_dims(padded_input, 0),
        output_grids_examples=jnp.expand_dims(padded_target, 0),
        input_masks_examples=jnp.expand_dims(padded_mask, 0),
        output_masks_examples=jnp.expand_dims(padded_mask, 0),
        num_train_pairs=1,
        test_input_grids=jnp.expand_dims(padded_input, 0),
        test_input_masks=jnp.expand_dims(padded_mask, 0),
        true_test_output_grids=jnp.expand_dims(padded_target, 0),
        true_test_output_masks=jnp.expand_dims(padded_mask, 0),
        num_test_pairs=1,
        task_index=jnp.array(0, dtype=jnp.int32),
    )