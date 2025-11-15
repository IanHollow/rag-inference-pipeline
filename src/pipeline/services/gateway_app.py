"""
Gateway service which Orchestrates the ML pipeline.

This service receives client requests and coordinates with retrieval and generation services.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from opentelemetry import trace
from prometheus_client import generate_latest
from pydantic import ValidationError

from ..config import get_settings
from ..telemetry import (
    error_counter as pipeline_error_counter,
    get_resource_snapshot,
    instrument_fastapi_app,
    memory_gauge,
    request_counter as pipeline_request_counter,
)
from .gateway.metrics import (
    error_counter as gateway_error_counter,
    latency_histogram,
    request_counter as gateway_request_counter,
)
from .gateway.orchestrator import Orchestrator
from .gateway.rpc_client import RPCError, RPCTimeoutError
from .gateway.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
settings = get_settings()
tracer = trace.get_tracer(__name__)

# Global orchestrator instance
orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown."""
    global orchestrator

    # Startup
    logger.info("Starting gateway service...")
    orchestrator = Orchestrator()
    await orchestrator.start()
    logger.info("Gateway service started")

    yield

    # Shutdown
    logger.info("Stopping gateway service...")
    if orchestrator:
        await orchestrator.stop()
    logger.info("Gateway service stopped")


app = FastAPI(
    title="ML Pipeline Gateway",
    description="Orchestrates distributed ML inference pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

instrument_fastapi_app(app)


def _record_memory_usage() -> None:
    """Push current memory stats to Prometheus gauges."""
    snapshot = get_resource_snapshot()
    labels = {"node": str(settings.node_number), "service": "gateway"}
    memory_gauge.labels(**labels, type="rss").set(snapshot.rss)
    memory_gauge.labels(**labels, type="vms").set(snapshot.vms)
    memory_gauge.labels(**labels, type="percent").set(snapshot.memory_percent)


@app.get("/health")
async def health() -> dict[str, str | int]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "node": settings.node_number,
        "role": settings.role.value,
        "total_nodes": settings.total_nodes,
    }


@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Main query endpoint - receives client requests and orchestrates pipeline.

    Args:
        request: QueryRequest with request_id and query

    Returns:
        QueryResponse with generated_response, sentiment, is_toxic

    Raises:
        HTTPException: On validation or processing errors
    """
    start_time = time.time()

    try:
        _record_memory_usage()

        # Validate request
        if not request.request_id:
            gateway_error_counter.labels(error_type="validation").inc()
            pipeline_error_counter.labels(
                node=str(settings.node_number),
                service="gateway",
                error_type="validation",
            ).inc()
            raise HTTPException(status_code=400, detail="request_id is required")

        if not request.query:
            gateway_error_counter.labels(error_type="validation").inc()
            pipeline_error_counter.labels(
                node=str(settings.node_number),
                service="gateway",
                error_type="validation",
            ).inc()
            raise HTTPException(status_code=400, detail="query is required")

        logger.info("Received query: request_id=%s", request.request_id)

        # Process through orchestrator
        if orchestrator is None:
            gateway_error_counter.labels(error_type="service_unavailable").inc()
            pipeline_error_counter.labels(
                node=str(settings.node_number),
                service="gateway",
                error_type="service_unavailable",
            ).inc()
            raise HTTPException(status_code=503, detail="Service not ready")

        with tracer.start_as_current_span(
            "gateway.query",
            attributes={
                "pipeline.request_id": request.request_id,
                "pipeline.service": "gateway",
                "pipeline.node": settings.node_number,
            },
        ):
            response = await orchestrator.process_query(
                request_id=request.request_id,
                query=request.query,
            )

        # Record metrics
        elapsed = time.time() - start_time
        latency_histogram.observe(elapsed)
        gateway_request_counter.labels(status="success").inc()
        pipeline_request_counter.labels(
            node=str(settings.node_number),
            service="gateway",
            status="success",
        ).inc()

        logger.info(
            "Query completed: request_id=%s, latency=%.3fs",
            request.request_id,
            elapsed,
        )

        return response

    except RPCTimeoutError as e:
        elapsed = time.time() - start_time
        latency_histogram.observe(elapsed)
        gateway_error_counter.labels(error_type="timeout").inc()
        pipeline_error_counter.labels(
            node=str(settings.node_number),
            service="gateway",
            error_type="timeout",
        ).inc()
        gateway_request_counter.labels(status="timeout").inc()
        pipeline_request_counter.labels(
            node=str(settings.node_number),
            service="gateway",
            status="error",
        ).inc()

        logger.error(
            "Query timeout: request_id=%s, latency=%.3fs",
            request.request_id,
            elapsed,
        )
        raise HTTPException(status_code=504, detail=str(e)) from e

    except RPCError as e:
        elapsed = time.time() - start_time
        latency_histogram.observe(elapsed)
        gateway_error_counter.labels(error_type="rpc_error").inc()
        pipeline_error_counter.labels(
            node=str(settings.node_number),
            service="gateway",
            error_type="rpc_error",
        ).inc()
        gateway_request_counter.labels(status="error").inc()
        pipeline_request_counter.labels(
            node=str(settings.node_number),
            service="gateway",
            status="error",
        ).inc()

        logger.exception(
            "Query RPC error: request_id=%s, latency=%.3fs",
            request.request_id,
            elapsed,
        )
        raise HTTPException(status_code=502, detail=str(e)) from e

    except ValidationError as e:
        elapsed = time.time() - start_time
        latency_histogram.observe(elapsed)
        gateway_error_counter.labels(error_type="validation").inc()
        pipeline_error_counter.labels(
            node=str(settings.node_number),
            service="gateway",
            error_type="validation",
        ).inc()
        gateway_request_counter.labels(status="error").inc()
        pipeline_request_counter.labels(
            node=str(settings.node_number),
            service="gateway",
            status="error",
        ).inc()

        logger.error(
            "Query validation error: request_id=%s, latency=%.3fs",
            request.request_id,
            elapsed,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        elapsed = time.time() - start_time
        latency_histogram.observe(elapsed)
        gateway_error_counter.labels(error_type="unknown").inc()
        pipeline_error_counter.labels(
            node=str(settings.node_number),
            service="gateway",
            error_type="unknown",
        ).inc()
        gateway_request_counter.labels(status="error").inc()
        pipeline_request_counter.labels(
            node=str(settings.node_number),
            service="gateway",
            status="error",
        ).inc()

        logger.exception(
            "Query unexpected error: request_id=%s, latency=%.3fs",
            request.request_id if hasattr(request, "request_id") else "unknown",
            elapsed,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/metrics")
@app.head("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
