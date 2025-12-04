"""
Prometheus metrics for the gateway service.
"""

from collections.abc import Sequence
from typing import Any, TypeVar, cast

from prometheus_client import REGISTRY, Counter, Histogram


MetricType = Counter | Histogram
T = TypeVar("T", bound=MetricType)


def get_metric(
    name: str,
    type_cls: type[T],
    documentation: str,
    labelnames: Sequence[str] = (),
    buckets: Sequence[float] | None = None,
) -> T:
    if name in REGISTRY._names_to_collectors:
        return cast("T", REGISTRY._names_to_collectors[name])
    kwargs = {}
    if buckets and type_cls is Histogram:
        kwargs["buckets"] = buckets
    return cast("T", type_cls(name, documentation, labelnames, **cast("Any", kwargs)))


# Request counter
request_counter = get_metric(
    "gateway_requests_total",
    Counter,
    "Total number of requests received",
    ["status"],  # success, error, timeout
)

# Batch size histogram
batch_size_histogram = get_metric(
    "gateway_batch_size",
    Histogram,
    "Distribution of batch sizes",
    buckets=[1, 2, 4, 8, 16, 32],
)

# End-to-end latency histogram
latency_histogram = get_metric(
    "gateway_request_latency_seconds",
    Histogram,
    "End-to-end request latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# RPC call duration histogram
rpc_duration_histogram = get_metric(
    "gateway_rpc_duration_seconds",
    Histogram,
    "Duration of RPC calls to downstream services",
    ["service"],  # retrieval, generation
    buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# Error counter by type
error_counter = get_metric(
    "gateway_errors_total",
    Counter,
    "Total number of errors by type",
    ["error_type"],  # rpc_error, timeout, validation, unknown
)
