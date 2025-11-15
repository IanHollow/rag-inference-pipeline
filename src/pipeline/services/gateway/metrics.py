"""
Prometheus metrics for the gateway service.
"""

from prometheus_client import Counter, Histogram

# Request counter
request_counter = Counter(
    "gateway_requests_total",
    "Total number of requests received",
    ["status"],  # success, error, timeout
)

# Batch size histogram
batch_size_histogram = Histogram(
    "gateway_batch_size",
    "Distribution of batch sizes",
    buckets=[1, 2, 4, 8, 16, 32],
)

# End-to-end latency histogram
latency_histogram = Histogram(
    "gateway_request_latency_seconds",
    "End-to-end request latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# RPC call duration histogram
rpc_duration_histogram = Histogram(
    "gateway_rpc_duration_seconds",
    "Duration of RPC calls to downstream services",
    ["service"],  # retrieval, generation
    buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# Error counter by type
error_counter = Counter(
    "gateway_errors_total",
    "Total number of errors by type",
    ["error_type"],  # rpc_error, timeout, validation, unknown
)
