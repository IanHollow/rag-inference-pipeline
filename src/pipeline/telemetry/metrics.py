"""
Prometheus metrics for the distributed ML pipeline.

This module provides shared metrics that can be used by all services.
All metrics are exposed via the /metrics endpoint on each node.
"""

from collections.abc import Sequence
from typing import Any, TypeVar, cast

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

MetricType = Counter | Gauge | Histogram
T = TypeVar("T", bound=MetricType)


def get_metric(
    name: str,
    type_cls: type[T],
    documentation: str,
    labelnames: Sequence[str],
    buckets: Sequence[float] | None = None,
) -> T:
    """
    Get an existing metric or create a new one.
    This prevents 'Duplicated timeseries' errors when reloading modules or running tests.
    """
    if name in REGISTRY._names_to_collectors:
        return cast("T", REGISTRY._names_to_collectors[name])

    # Also check for the base name if _total is appended automatically
    # But here we are explicit with names usually.

    kwargs = {}
    if buckets and type_cls is Histogram:
        kwargs["buckets"] = buckets
    return cast("T", type_cls(name, documentation, labelnames, **cast("Any", kwargs)))


# === Request Metrics ===

request_counter = get_metric(
    "pipeline_requests_total",
    Counter,
    "Total number of requests received by the pipeline",
    [
        "run_id",
        "node",
        "service",
        "status",
    ],  # node=0/1/2, service=gateway/retrieval/generation, status=success/error
)

latency_histogram = get_metric(
    "pipeline_request_latency_seconds",
    Histogram,
    "End-to-end request latency distribution",
    ["run_id", "node", "service"],  # Which service measured this
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# === Batch Metrics ===

batch_size_histogram = get_metric(
    "pipeline_batch_size",
    Histogram,
    "Distribution of batch sizes across services",
    ["run_id", "node", "service"],
    buckets=[1, 2, 4, 8, 16, 32, 64],
)

# === RPC/Network Metrics ===

rpc_duration_histogram = get_metric(
    "pipeline_rpc_duration_seconds",
    Histogram,
    "Duration of RPC calls between services",
    ["run_id", "source_node", "target_service"],  # e.g., source_node=0, target_service=retrieval
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# === Stage Duration Metrics ===

stage_duration_gauge = get_metric(
    "pipeline_stage_duration_seconds",
    Gauge,
    "Duration of each pipeline stage (last measurement)",
    ["run_id", "node", "stage"],  # stage=embedding/retrieval/reranking/llm/sentiment/toxicity
)

# === Resource Metrics ===

memory_gauge = get_metric(
    "pipeline_memory_bytes",
    Gauge,
    "Current memory usage in bytes",
    ["run_id", "node", "service", "type"],  # type=rss/vms/percent
)

# === Error Metrics ===

error_counter = get_metric(
    "pipeline_errors_total",
    Counter,
    "Total number of errors by type and location",
    [
        "run_id",
        "node",
        "service",
        "error_type",
    ],  # error_type=rpc_error/timeout/validation/oom/unknown
)
