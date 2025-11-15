"""
Prometheus metrics for the distributed ML pipeline.

This module provides shared metrics that can be used by all services.
All metrics are exposed via the /metrics endpoint on each node.
"""

from prometheus_client import Counter, Gauge, Histogram

# === Request Metrics ===

request_counter = Counter(
    "pipeline_requests_total",
    "Total number of requests received by the pipeline",
    [
        "node",
        "service",
        "status",
    ],  # node=0/1/2, service=gateway/retrieval/generation, status=success/error
)

latency_histogram = Histogram(
    "pipeline_request_latency_seconds",
    "End-to-end request latency distribution",
    ["node", "service"],  # Which service measured this
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# === Batch Metrics ===

batch_size_histogram = Histogram(
    "pipeline_batch_size",
    "Distribution of batch sizes across services",
    ["node", "service"],
    buckets=[1, 2, 4, 8, 16, 32, 64],
)

# === RPC/Network Metrics ===

rpc_duration_histogram = Histogram(
    "pipeline_rpc_duration_seconds",
    "Duration of RPC calls between services",
    ["source_node", "target_service"],  # e.g., source_node=0, target_service=retrieval
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# === Stage Duration Metrics ===

stage_duration_gauge = Gauge(
    "pipeline_stage_duration_seconds",
    "Duration of each pipeline stage (last measurement)",
    ["node", "stage"],  # stage=embedding/retrieval/reranking/llm/sentiment/toxicity
)

# === Resource Metrics ===

memory_gauge = Gauge(
    "pipeline_memory_bytes",
    "Current memory usage in bytes",
    ["node", "service", "type"],  # type=rss/vms/percent
)

# === Error Metrics ===

error_counter = Counter(
    "pipeline_errors_total",
    "Total number of errors by type and location",
    ["node", "service", "error_type"],  # error_type=rpc_error/timeout/validation/oom/unknown
)
