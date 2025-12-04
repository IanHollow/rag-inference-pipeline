"""
Prometheus metrics for the generation service.
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


generation_requests_total = get_metric(
    "generation_requests_total",
    Counter,
    "Total number of generation requests",
)

generation_request_duration = get_metric(
    "generation_request_duration_seconds",
    Histogram,
    "Generation request duration in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

generation_batch_size = get_metric(
    "generation_batch_size",
    Histogram,
    "Number of items in generation batch",
    buckets=[1, 2, 4, 8, 16, 32],
)

rerank_duration = get_metric(
    "rerank_duration_seconds",
    Histogram,
    "Reranking duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)

llm_generation_duration = get_metric(
    "llm_generation_duration_seconds",
    Histogram,
    "LLM generation duration in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

sentiment_duration = get_metric(
    "sentiment_duration_seconds",
    Histogram,
    "Sentiment analysis duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)

toxicity_duration = get_metric(
    "toxicity_duration_seconds",
    Histogram,
    "Toxicity filtering duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)
