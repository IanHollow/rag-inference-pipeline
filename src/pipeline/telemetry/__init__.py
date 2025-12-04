"""
Telemetry and profiling utilities for the distributed ML pipeline.
"""

from .metrics import (
    batch_flush_counter,
    batch_size_histogram,
    cache_evictions_total,
    cache_hits_total,
    cache_misses_total,
    error_counter,
    latency_histogram,
    memory_gauge,
    queue_depth_gauge,
    request_counter,
    rpc_duration_histogram,
    stage_duration_gauge,
    worker_utilization_gauge,
)
from .profiling import (
    ProfileResult,
    ResourceSnapshot,
    SampledStageProfiler,
    StageProfiler,
    get_resource_snapshot,
    profile_context,
)
from .tracing import instrument_fastapi_app, setup_tracing


__all__ = [
    "ProfileResult",
    "ResourceSnapshot",
    "SampledStageProfiler",
    "StageProfiler",
    "batch_flush_counter",
    "batch_size_histogram",
    "cache_evictions_total",
    "cache_hits_total",
    "cache_misses_total",
    "error_counter",
    "get_resource_snapshot",
    "instrument_fastapi_app",
    "latency_histogram",
    "memory_gauge",
    "profile_context",
    "queue_depth_gauge",
    "request_counter",
    "rpc_duration_histogram",
    "setup_tracing",
    "stage_duration_gauge",
    "worker_utilization_gauge",
]
