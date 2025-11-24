"""
Profiling utilities for monitoring resource usage during pipeline execution.

This module provides utilities to snapshot memory and CPU usage, and track
resource consumption across pipeline stages.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import logging
import secrets
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """
    Snapshot of system resource usage at a point in time.
    """

    timestamp: float
    # Memory metrics (in bytes)
    rss: int  # Resident Set Size - physical memory
    vms: int  # Virtual Memory Size
    memory_percent: float  # Percentage of system memory
    # CPU metrics
    cpu_percent: float  # CPU usage percentage
    # Derived metrics
    memory_mb: float  # RSS in megabytes for convenience

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def get_resource_snapshot() -> ResourceSnapshot:
    """
    Capture current resource usage snapshot.

    Returns:
        ResourceSnapshot with current memory and CPU metrics
    """
    process = psutil.Process()

    # Get memory info
    mem_info = process.memory_info()
    mem_percent = process.memory_percent()

    # Get CPU info (interval=None for instant reading)
    cpu_percent = process.cpu_percent(interval=None)

    return ResourceSnapshot(
        timestamp=time.time(),
        rss=mem_info.rss,
        vms=mem_info.vms,
        memory_percent=mem_percent,
        cpu_percent=cpu_percent,
        memory_mb=mem_info.rss / (1024 * 1024),
    )


@dataclass
class ProfileResult:
    """
    Result of profiling a code block.
    """

    stage_name: str
    duration_ms: float
    before: ResourceSnapshot
    after: ResourceSnapshot
    memory_delta_mb: float
    peak_memory_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage_name": self.stage_name,
            "duration_ms": self.duration_ms,
            "memory_delta_mb": self.memory_delta_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


@contextmanager
def profile_context(stage_name: str, log_results: bool = True) -> Iterator[ProfileResult]:
    """
    Context manager for profiling a code block.

    Args:
        stage_name: Name of the stage being profiled
        log_results: Whether to log profiling results

    Yields:
        ProfileResult object that will be populated after the context exits
    """
    # Create result object (will be populated on exit)
    result = ProfileResult(
        stage_name=stage_name,
        duration_ms=0.0,
        before=get_resource_snapshot(),
        after=get_resource_snapshot(),  # Placeholder
        memory_delta_mb=0.0,
        peak_memory_mb=0.0,
    )

    start_time = time.perf_counter()
    start_memory_mb = result.before.memory_mb

    try:
        yield result
    finally:
        # Capture end metrics
        end_time = time.perf_counter()
        result.after = get_resource_snapshot()

        # Calculate metrics
        result.duration_ms = (end_time - start_time) * 1000
        result.memory_delta_mb = result.after.memory_mb - start_memory_mb
        result.peak_memory_mb = max(start_memory_mb, result.after.memory_mb)

        if log_results:
            logger.info(
                "Profile [%s]: duration=%.2fms, memory_delta=%.2fMB, peak_memory=%.2fMB",
                stage_name,
                result.duration_ms,
                result.memory_delta_mb,
                result.peak_memory_mb,
            )


class StageProfiler:
    """
    Utility class for tracking profiling data across multiple stages.
    """

    def __init__(self) -> None:
        """Initialize the profiler."""
        self.profiles: list[ProfileResult] = []

    def add(self, profile: ProfileResult) -> None:
        """Add a profile result."""
        self.profiles.append(profile)

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics across all profiled stages.

        Returns:
            Dictionary with total duration, memory statistics, and per-stage breakdown
        """
        if not self.profiles:
            return {"total_duration_ms": 0.0, "stages": []}

        total_duration = sum(p.duration_ms for p in self.profiles)
        peak_memory = max(p.peak_memory_mb for p in self.profiles)
        total_memory_delta = sum(p.memory_delta_mb for p in self.profiles)

        return {
            "total_duration_ms": total_duration,
            "peak_memory_mb": peak_memory,
            "total_memory_delta_mb": total_memory_delta,
            "stage_count": len(self.profiles),
            "stages": [p.to_dict() for p in self.profiles],
        }


class SampledStageProfiler:
    """
    Sampling wrapper around StageProfiler to limit profiling overhead.
    """

    def __init__(
        self,
        enabled: bool,
        sample_rate: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = enabled and sample_rate > 0
        self.sample_rate = max(0.0, min(sample_rate, 1.0))
        self.profiler = StageProfiler() if self._should_sample() else None

    def _should_sample(self) -> bool:
        return self.enabled and secrets.SystemRandom().random() <= self.sample_rate

    def is_active(self) -> bool:
        """Return whether profiling is active for the current request."""
        return self.profiler is not None

    @contextmanager
    def track(self, stage_name: str, log_results: bool = False) -> Iterator[ProfileResult | None]:
        """
        Profile a stage if sampling is active.
        """
        if not self.profiler:
            yield None
            return

        with profile_context(stage_name, log_results=log_results) as profile:
            yield profile
            self.profiler.add(profile)

    def summary(self) -> dict[str, Any] | None:
        """
        Return summary of collected samples if profiling was active.
        """
        if not self.profiler:
            return None

        summary = self.profiler.get_summary()
        self.logger.debug("Profiling summary collected: %s", summary)
        return summary

    def clear(self) -> None:
        """Clear all profiling data."""
        if self.profiler:
            self.profiler.profiles.clear()
