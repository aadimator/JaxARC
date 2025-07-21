"""Performance monitoring for visualization impact measurement.

This module provides performance monitoring capabilities to measure the impact
of visualization and logging on JAX computation performance.
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, TypeVar

import chex
from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


@chex.dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""

    enabled: bool = True
    max_samples: int = 1000  # Maximum samples to keep in memory
    alert_threshold: float = 0.05  # Alert if overhead > 5%
    adaptive_logging: bool = True  # Reduce logging if performance impact is high
    measurement_window: int = 100  # Number of samples for moving averages
    min_execution_time: float = 0.001  # Minimum time to consider for measurement (1ms)


@chex.dataclass
class PerformanceSample:
    """Single performance measurement sample."""

    timestamp: float
    function_name: str
    execution_time: float
    visualization_time: float
    overhead_ratio: float
    memory_usage: Optional[int] = None  # Memory usage in bytes


class PerformanceMonitor:
    """Monitor visualization performance impact.

    This class tracks the performance impact of visualization operations
    on the main computation and provides adaptive logging capabilities.
    """

    def __init__(self, config: PerformanceConfig):
        """Initialize the performance monitor.

        Args:
            config: Configuration for performance monitoring
        """
        self.config = config

        # Performance data storage
        self.samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.max_samples)
        )
        self.function_stats: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Adaptive logging state
        self.logging_level = 1.0  # 1.0 = full logging, 0.0 = no logging
        self.last_adaptation = time.time()
        self.adaptation_interval = 30.0  # Adapt every 30 seconds

        # Alert tracking
        self.alerts_sent: Dict[str, float] = {}  # function_name -> last_alert_time
        self.alert_cooldown = 300.0  # 5 minutes between alerts for same function

        logger.info(f"PerformanceMonitor initialized with config: {config}")

    def measure_function_impact(
        self, func: F, function_name: Optional[str] = None
    ) -> F:
        """Decorator to measure visualization impact on function performance.

        Args:
            func: Function to measure
            function_name: Optional name for the function (defaults to func.__name__)

        Returns:
            Decorated function with performance measurement
        """
        if not self.config.enabled:
            return func

        name = function_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip measurement for very fast functions
            start_time = time.perf_counter()

            # Execute function
            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Only measure if execution time is significant
            if execution_time >= self.config.min_execution_time:
                # For now, assume visualization time is 0 since this is just function timing
                # In practice, this would be used with visualization functions
                self._record_sample(name, execution_time, 0.0)

            return result

        return wrapper

    def measure_visualization_impact(
        self,
        main_func: Callable,
        viz_func: Callable,
        function_name: Optional[str] = None,
    ) -> Callable:
        """Measure the impact of visualization on main computation.

        Args:
            main_func: Main computation function
            viz_func: Visualization function
            function_name: Optional name for tracking

        Returns:
            Combined function with performance measurement
        """
        if not self.config.enabled:

            def combined(*args, **kwargs):
                result = main_func(*args, **kwargs)
                if self.should_log():
                    viz_func(*args, **kwargs)
                return result

            return combined

        name = function_name or f"{main_func.__name__}_with_viz"

        def combined(*args, **kwargs):
            # Measure main computation
            start_main = time.perf_counter()
            result = main_func(*args, **kwargs)
            end_main = time.perf_counter()
            main_time = end_main - start_main

            # Measure visualization if we should log
            viz_time = 0.0
            if self.should_log():
                start_viz = time.perf_counter()
                try:
                    viz_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
                end_viz = time.perf_counter()
                viz_time = end_viz - start_viz

            # Record performance sample
            if main_time >= self.config.min_execution_time:
                self._record_sample(name, main_time, viz_time)

            return result

        return combined

    def _record_sample(
        self, function_name: str, execution_time: float, visualization_time: float
    ) -> None:
        """Record a performance sample.

        Args:
            function_name: Name of the function
            execution_time: Time for main execution
            visualization_time: Time for visualization
        """
        overhead_ratio = (
            visualization_time / execution_time if execution_time > 0 else 0.0
        )

        sample = PerformanceSample(
            timestamp=time.time(),
            function_name=function_name,
            execution_time=execution_time,
            visualization_time=visualization_time,
            overhead_ratio=overhead_ratio,
        )

        # Store sample
        self.samples[function_name].append(sample)

        # Update function statistics
        self._update_function_stats(function_name)

        # Check for performance alerts
        self._check_performance_alert(function_name, overhead_ratio)

        # Adapt logging if needed
        if self.config.adaptive_logging:
            self._adapt_logging_level()

    def _update_function_stats(self, function_name: str) -> None:
        """Update statistics for a function.

        Args:
            function_name: Name of the function to update stats for
        """
        samples = list(self.samples[function_name])
        if not samples:
            return

        # Calculate statistics over the measurement window
        recent_samples = samples[-self.config.measurement_window :]

        execution_times = [s.execution_time for s in recent_samples]
        viz_times = [s.visualization_time for s in recent_samples]
        overhead_ratios = [s.overhead_ratio for s in recent_samples]

        stats = {
            "count": len(recent_samples),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "avg_viz_time": sum(viz_times) / len(viz_times),
            "avg_overhead_ratio": sum(overhead_ratios) / len(overhead_ratios),
            "max_overhead_ratio": max(overhead_ratios),
            "last_updated": time.time(),
        }

        self.function_stats[function_name] = stats

    def _check_performance_alert(
        self, function_name: str, overhead_ratio: float
    ) -> None:
        """Check if a performance alert should be sent.

        Args:
            function_name: Name of the function
            overhead_ratio: Current overhead ratio
        """
        if overhead_ratio <= self.config.alert_threshold:
            return

        # Check cooldown
        last_alert = self.alerts_sent.get(function_name, 0)
        if time.time() - last_alert < self.alert_cooldown:
            return

        # Send alert
        logger.warning(
            f"High visualization overhead detected for {function_name}: "
            f"{overhead_ratio:.1%} (threshold: {self.config.alert_threshold:.1%})"
        )

        self.alerts_sent[function_name] = time.time()

    def _adapt_logging_level(self) -> None:
        """Adapt logging level based on performance impact."""
        current_time = time.time()
        if current_time - self.last_adaptation < self.adaptation_interval:
            return

        # Calculate overall performance impact
        total_overhead = 0.0
        total_samples = 0

        for function_name, stats in self.function_stats.items():
            if "avg_overhead_ratio" in stats and stats["count"] > 0:
                total_overhead += stats["avg_overhead_ratio"] * stats["count"]
                total_samples += stats["count"]

        if total_samples == 0:
            return

        avg_overhead = total_overhead / total_samples

        # Adapt logging level
        if avg_overhead > self.config.alert_threshold * 2:
            # High overhead - reduce logging significantly
            self.logging_level = max(0.1, self.logging_level * 0.5)
            logger.info(
                f"Reducing logging level to {self.logging_level:.2f} due to high overhead ({avg_overhead:.1%})"
            )
        elif avg_overhead > self.config.alert_threshold:
            # Moderate overhead - reduce logging moderately
            self.logging_level = max(0.3, self.logging_level * 0.8)
            logger.info(
                f"Reducing logging level to {self.logging_level:.2f} due to moderate overhead ({avg_overhead:.1%})"
            )
        elif (
            avg_overhead < self.config.alert_threshold * 0.5
            and self.logging_level < 1.0
        ):
            # Low overhead - increase logging
            self.logging_level = min(1.0, self.logging_level * 1.2)
            logger.info(
                f"Increasing logging level to {self.logging_level:.2f} due to low overhead ({avg_overhead:.1%})"
            )

        self.last_adaptation = current_time

    def should_log(self) -> bool:
        """Determine if logging should occur based on adaptive level.

        Returns:
            True if logging should occur
        """
        if not self.config.enabled or not self.config.adaptive_logging:
            return True

        # Use logging level as probability
        import random

        return random.random() < self.logging_level

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.

        Returns:
            Dictionary with performance statistics
        """
        report = {
            "config": self.config,
            "logging_level": self.logging_level,
            "total_functions": len(self.function_stats),
            "functions": {},
        }

        for function_name, stats in self.function_stats.items():
            samples = list(self.samples[function_name])

            report["functions"][function_name] = {
                "stats": stats,
                "total_samples": len(samples),
                "recent_samples": len(samples[-self.config.measurement_window :]),
                "alerts_sent": self.alerts_sent.get(function_name, 0) > 0,
            }

        # Overall statistics
        if self.function_stats:
            all_overheads = [
                stats.get("avg_overhead_ratio", 0.0)
                for stats in self.function_stats.values()
            ]
            report["overall"] = {
                "avg_overhead": sum(all_overheads) / len(all_overheads),
                "max_overhead": max(all_overheads),
                "functions_over_threshold": sum(
                    1
                    for overhead in all_overheads
                    if overhead > self.config.alert_threshold
                ),
            }

        return report

    def get_function_stats(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific function.

        Args:
            function_name: Name of the function

        Returns:
            Function statistics or None if not found
        """
        if function_name not in self.function_stats:
            return None

        stats = self.function_stats[function_name].copy()
        samples = list(self.samples[function_name])

        stats["total_samples"] = len(samples)
        stats["recent_samples"] = len(samples[-self.config.measurement_window :])

        return stats

    def reset_stats(self, function_name: Optional[str] = None) -> None:
        """Reset performance statistics.

        Args:
            function_name: Specific function to reset, or None for all functions
        """
        if function_name:
            if function_name in self.samples:
                self.samples[function_name].clear()
            if function_name in self.function_stats:
                del self.function_stats[function_name]
            if function_name in self.alerts_sent:
                del self.alerts_sent[function_name]
            logger.info(f"Reset stats for function: {function_name}")
        else:
            self.samples.clear()
            self.function_stats.clear()
            self.alerts_sent.clear()
            self.logging_level = 1.0
            logger.info("Reset all performance statistics")

    def should_reduce_logging(self) -> bool:
        """Determine if logging should be reduced due to performance impact.

        Returns:
            True if logging should be reduced
        """
        return self.logging_level < 0.5

    def get_overhead_summary(self) -> Dict[str, float]:
        """Get a summary of overhead ratios by function.

        Returns:
            Dictionary mapping function names to average overhead ratios
        """
        summary = {}
        for function_name, stats in self.function_stats.items():
            summary[function_name] = stats.get("avg_overhead_ratio", 0.0)
        return summary

    def export_samples(
        self, function_name: str, max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Export performance samples for analysis.

        Args:
            function_name: Name of the function
            max_samples: Maximum number of samples to export

        Returns:
            List of sample dictionaries
        """
        if function_name not in self.samples:
            return []

        samples = list(self.samples[function_name])
        if max_samples:
            samples = samples[-max_samples:]

        return [
            {
                "timestamp": sample.timestamp,
                "function_name": sample.function_name,
                "execution_time": sample.execution_time,
                "visualization_time": sample.visualization_time,
                "overhead_ratio": sample.overhead_ratio,
                "memory_usage": sample.memory_usage,
            }
            for sample in samples
        ]


# Convenience decorators
def monitor_performance(
    monitor: PerformanceMonitor, function_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator for monitoring function performance.

    Args:
        monitor: Performance monitor instance
        function_name: Optional function name override

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        return monitor.measure_function_impact(func, function_name)

    return decorator


def monitor_visualization_impact(
    monitor: PerformanceMonitor, viz_func: Callable, function_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator for monitoring visualization impact.

    Args:
        monitor: Performance monitor instance
        viz_func: Visualization function to measure
        function_name: Optional function name override

    Returns:
        Decorator function
    """

    def decorator(main_func: F) -> F:
        return monitor.measure_visualization_impact(main_func, viz_func, function_name)

    return decorator
