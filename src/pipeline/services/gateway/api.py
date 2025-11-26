"""
Gateway service API

Orchestrates the ML pipeline.
"""

import logging
import threading
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from opentelemetry import trace
from prometheus_client import generate_latest
from pydantic import ValidationError

from ...components.embedding import EmbeddingGenerator
from ...components.reranker import Reranker
from ...components.sentiment import SentimentAnalyzer
from ...components.toxicity import ToxicityFilter
from ...config import get_settings
from ...dependencies import get_component
from ...telemetry import (
    error_counter as pipeline_error_counter,
    get_resource_snapshot,
    memory_gauge,
    request_counter as pipeline_request_counter,
)
from .metrics import (
    error_counter as gateway_error_counter,
    latency_histogram,
    request_counter as gateway_request_counter,
)
from .orchestrator import Orchestrator
from .rpc_client import RPCError, RPCTimeoutError
from .schemas import QueryRequest, QueryResponse


class _MetricsCache:
    def __init__(self):
        self.content = None
        self.time = 0.0
        self.lock = threading.Lock()
        self.ttl = 0.5

    def get_content(self):
        now = time.monotonic()
        if self.content is not None and now - self.time < self.ttl:
            return self.content
        with self.lock:
            if self.content is not None and now - self.time < self.ttl:
                return self.content
            content = generate_latest()
            self.content = content
            self.time = now
            return content


_cache = _MetricsCache()

logger = logging.getLogger(__name__)
settings = get_settings()
tracer = trace.get_tracer(__name__)

router = APIRouter()


def _record_memory_usage() -> None:
    """Push current memory stats to Prometheus gauges."""
    snapshot = get_resource_snapshot()
    labels = {
        "run_id": settings.profiling_run_id,
        "node": str(settings.node_number),
        "service": "gateway",
    }
    memory_gauge.labels(**labels, type="rss").set(snapshot.rss)
    memory_gauge.labels(**labels, type="vms").set(snapshot.vms)
    memory_gauge.labels(**labels, type="percent").set(snapshot.memory_percent)


@router.post("/query")
async def query(
    request: Request,
    query_request: QueryRequest,
    orchestrator: Annotated[Orchestrator, Depends(get_component("orchestrator"))],
    embedding_generator: Annotated[
        EmbeddingGenerator | None, Depends(get_component("embedding_generator"))
    ] = None,
    reranker: Annotated[Reranker | None, Depends(get_component("reranker"))] = None,
    sentiment_analyzer: Annotated[
        SentimentAnalyzer | None, Depends(get_component("sentiment_analyzer"))
    ] = None,
    toxicity_filter: Annotated[
        ToxicityFilter | None, Depends(get_component("toxicity_filter"))
    ] = None,
) -> QueryResponse:
    """
    Main query endpoint - receives client requests and orchestrates pipeline.
    """
    start_time = time.time()
    _record_memory_usage()

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    # Inject components if available
    orchestrator.set_components(
        embedding_generator=embedding_generator,
        reranker=reranker,
        sentiment_analyzer=sentiment_analyzer,
        toxicity_filter=toxicity_filter,
    )

    try:
        # Validate request
        if not query_request.request_id:
            gateway_error_counter.labels(error_type="validation").inc()
            pipeline_error_counter.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                service="gateway",
                error_type="validation",
            ).inc()
            raise HTTPException(status_code=400, detail="request_id is required")

        if not query_request.query:
            gateway_error_counter.labels(error_type="validation").inc()
            pipeline_error_counter.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                service="gateway",
                error_type="validation",
            ).inc()
            raise HTTPException(status_code=400, detail="Query is required")

        gateway_request_counter.labels(status="received").inc()
        pipeline_request_counter.labels(
            run_id=settings.profiling_run_id,
            node=str(settings.node_number),
            service="gateway",
            status="received",
        ).inc()

        with tracer.start_as_current_span(
            "gateway.query",
            attributes={
                "pipeline.request_id": query_request.request_id,
                "pipeline.node": settings.node_number,
            },
        ):
            # Delegate to orchestrator
            response = await orchestrator.process_query(
                query_request.request_id, query_request.query
            )

        # Record metrics
        duration = time.time() - start_time
        latency_histogram.observe(duration)
        pipeline_request_counter.labels(
            run_id=settings.profiling_run_id,
            node=str(settings.node_number),
            service="gateway",
            status="success",
        ).inc()

        return response

    except (RPCError, RPCTimeoutError) as e:
        logger.error("RPC error processing query %s: %s", query_request.request_id, e)
        gateway_error_counter.labels(error_type="rpc_error").inc()
        pipeline_error_counter.labels(
            run_id=settings.profiling_run_id,
            node=str(settings.node_number),
            service="gateway",
            error_type="rpc_error",
        ).inc()
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e!s}") from e

    except ValidationError as e:
        logger.error("Validation error: %s", e)
        gateway_error_counter.labels(error_type="validation").inc()
        pipeline_error_counter.labels(
            run_id=settings.profiling_run_id,
            node=str(settings.node_number),
            service="gateway",
            error_type="validation",
        ).inc()
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        logger.exception("Unexpected error processing query %s: %s", query_request.request_id, e)
        gateway_error_counter.labels(error_type="unknown").inc()
        pipeline_error_counter.labels(
            run_id=settings.profiling_run_id,
            node=str(settings.node_number),
            service="gateway",
            error_type="unknown",
        ).inc()
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/clear_cache")
async def clear_cache(
    orchestrator: Annotated[Orchestrator, Depends(get_component("orchestrator"))],
) -> dict[str, Any]:
    """Clear the gateway response cache and downstream caches."""
    results = {}

    if orchestrator:
        # Clear Gateway Cache
        orchestrator.query_cache.clear()
        results["gateway"] = "cleared"
        logger.info("Gateway cache cleared")

        # Clear Downstream Caches
        if orchestrator.retrieval_client:
            success = await orchestrator.retrieval_client.clear_cache(
                endpoint="/retrieve/clear_cache"
            )
            results["retrieval"] = "cleared" if success else "failed"

        if orchestrator.generation_client:
            success = await orchestrator.generation_client.clear_cache(
                endpoint="/generate/clear_cache"
            )
            results["generation"] = "cleared" if success else "failed"
    else:
        results["gateway"] = "error: orchestrator not initialized"

    return results


@router.get("/metrics", response_class=Response)
@router.head("/metrics", response_class=Response)
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=_cache.get_content(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
