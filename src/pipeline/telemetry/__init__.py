"""
Telemetry and profiling utilities for the distributed ML pipeline.
"""

from .metrics import (
    batch_size_histogram,
    error_counter,
    latency_histogram,
    memory_gauge,
    request_counter,
    rpc_duration_histogram,
    stage_duration_gauge,
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
    "batch_size_histogram",
    "error_counter",
    "get_resource_snapshot",
    "instrument_fastapi_app",
    "latency_histogram",
    "memory_gauge",
    "profile_context",
    "request_counter",
    "rpc_duration_histogram",
    "setup_tracing",
    "stage_duration_gauge",
]
