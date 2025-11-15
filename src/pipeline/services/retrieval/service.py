"""
Retrieval service

Provides embedding generation, FAISS ANN search, and document retrieval.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
import time

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from opentelemetry import trace
from prometheus_client import Counter, Histogram, generate_latest

from ...config import get_settings
from ...telemetry import (
    SampledStageProfiler,
    batch_size_histogram as pipeline_batch_size_histogram,
    error_counter as pipeline_error_counter,
    get_resource_snapshot,
    instrument_fastapi_app,
    latency_histogram as pipeline_latency_histogram,
    memory_gauge,
    request_counter as pipeline_request_counter,
    stage_duration_gauge,
)
from .document_store import DocumentStore
from .embedding import EmbeddingGenerator
from .faiss_store import FAISSStore
from .schemas import (
    ErrorResponse,
    HealthResponse,
    RetrievalDocument,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResponseItem,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
settings = get_settings()
NODE_LABEL = str(settings.node_number)
tracer = trace.get_tracer(__name__)
embedding_generator: EmbeddingGenerator | None = None
faiss_store: FAISSStore | None = None
document_store: DocumentStore | None = None

# Prometheus metrics
retrieval_requests_total = Counter(
    "retrieval_requests_total",
    "Total number of retrieval requests",
)

retrieval_request_duration = Histogram(
    "retrieval_request_duration_seconds",
    "Retrieval request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

retrieval_batch_size = Histogram(
    "retrieval_batch_size",
    "Number of items in retrieval batch",
    buckets=[1, 2, 4, 8, 16, 32],
)

faiss_search_duration = Histogram(
    "faiss_search_duration_seconds",
    "FAISS search duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

document_fetch_duration = Histogram(
    "document_fetch_duration_seconds",
    "Document fetch duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

embedding_duration = Histogram(
    "embedding_duration_seconds",
    "Embedding generation duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager for service initialization and cleanup.

    Loads models and indices on startup, cleans up on shutdown.
    """
    global embedding_generator, faiss_store, document_store

    logger.info("=" * 60)
    logger.info("RETRIEVAL SERVICE STARTING")
    logger.info("=" * 60)
    logger.info("Node: %d/%d", settings.node_number, settings.total_nodes)
    logger.info("Role: %s", settings.role.value)
    logger.info("=" * 60)

    try:
        # Initialize document store
        logger.info("Initializing document store...")
        document_store = DocumentStore(settings)
        logger.info("Document store initialized")

        # Initialize and load FAISS index
        logger.info("Initializing FAISS store...")
        faiss_store = FAISSStore(settings)
        faiss_store.load()
        logger.info("FAISS index loaded: %d vectors", faiss_store.index_size)

        # Initialize and load embedding model
        logger.info("Initializing embedding generator...")
        embedding_generator = EmbeddingGenerator(settings)
        embedding_generator.load()
        logger.info("Embedding model loaded")

        logger.info("=" * 60)
        logger.info("RETRIEVAL SERVICE READY")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.exception("Failed to initialize retrieval service: %s", e)
        raise

    finally:
        # Cleanup
        logger.info("Shutting down retrieval service...")

        if embedding_generator is not None:
            embedding_generator.unload()

        if faiss_store is not None:
            faiss_store.unload()

        if document_store is not None:
            document_store.close_all()

        logger.info("Retrieval service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Retrieval Service",
    description="Node 1: Embedding, FAISS search, and document retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

instrument_fastapi_app(app)


def _record_memory_usage() -> None:
    """Record current process memory in Prometheus gauges."""
    snapshot = get_resource_snapshot()
    memory_gauge.labels(node=NODE_LABEL, service="retrieval", type="rss").set(snapshot.rss)
    memory_gauge.labels(node=NODE_LABEL, service="retrieval", type="vms").set(snapshot.vms)
    memory_gauge.labels(
        node=NODE_LABEL,
        service="retrieval",
        type="percent",
    ).set(snapshot.memory_percent)


def _record_pipeline_error(error_type: str) -> None:
    """Increment shared error counters for this service."""
    pipeline_error_counter.labels(
        node=NODE_LABEL,
        service="retrieval",
        error_type=error_type,
    ).inc()
    pipeline_request_counter.labels(
        node=NODE_LABEL,
        service="retrieval",
        status="error",
    ).inc()


@app.get("/health")
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status and component availability
    """
    return HealthResponse(
        status="healthy",
        node=settings.node_number,
        total_nodes=settings.total_nodes,
        embedding_loaded=embedding_generator is not None and embedding_generator.is_loaded,
        faiss_loaded=faiss_store is not None and faiss_store.is_loaded,
        documents_available=document_store is not None,
    )


@app.post(
    "/retrieve",
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        400: {"model": ErrorResponse, "description": "Bad request"},
    },
)
async def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    """
    Process a batch of retrieval requests.

    Args:
        request: Batch of queries to process

    Returns:
        RetrievalResponse with retrieved documents for each query

    Raises:
        HTTPException: If service components are not initialized or processing fails
    """
    profiler = SampledStageProfiler(
        enabled=settings.enable_profiling,
        sample_rate=settings.profiling_sample_rate,
        logger=logger,
    )
    start_time = time.time()

    _record_memory_usage()

    # Validate service state
    if embedding_generator is None or not embedding_generator.is_loaded:
        _record_pipeline_error("embedding_not_loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding model not loaded",
        )

    if faiss_store is None or not faiss_store.is_loaded:
        _record_pipeline_error("faiss_not_loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FAISS index not loaded",
        )

    if document_store is None:
        _record_pipeline_error("document_store_missing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document store not initialized",
        )

    # Validate request
    if not request.items:
        _record_pipeline_error("validation")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch must contain at least one item",
        )

    batch_size = len(request.items)
    retrieval_requests_total.inc()
    retrieval_batch_size.observe(batch_size)
    pipeline_batch_size_histogram.labels(node=NODE_LABEL, service="retrieval").observe(batch_size)

    logger.info("Processing retrieval batch: %s with %d items", request.batch_id, batch_size)

    try:
        with tracer.start_as_current_span(
            "retrieval.batch",
            attributes={
                "pipeline.batch_id": request.batch_id,
                "pipeline.service": "retrieval",
                "pipeline.node": settings.node_number,
            },
        ):
            # Step 1: Generate embeddings for all queries
            queries = [item.query for item in request.items]
            logger.debug("Generating embeddings for %d queries", len(queries))

            embed_start = time.time()
            with (
                tracer.start_as_current_span("retrieval.embedding"),
                profiler.track("retrieval.embedding"),
            ):
                embeddings = embedding_generator.encode(queries)
            embed_elapsed = time.time() - embed_start
            embedding_duration.observe(embed_elapsed)
            stage_duration_gauge.labels(node=NODE_LABEL, stage="retrieval.embedding").set(
                embed_elapsed
            )
            logger.debug("Embeddings generated in %.3fs", embed_elapsed)

            # Step 2: FAISS ANN search
            logger.debug("Performing FAISS search with k=%d", settings.retrieval_k)
            faiss_start = time.time()
            with (
                tracer.start_as_current_span("retrieval.faiss_search"),
                profiler.track("retrieval.faiss_search"),
            ):
                distances, indices = faiss_store.search(embeddings, settings.retrieval_k)
            faiss_elapsed = time.time() - faiss_start
            faiss_search_duration.observe(faiss_elapsed)
            stage_duration_gauge.labels(node=NODE_LABEL, stage="retrieval.faiss_search").set(
                faiss_elapsed
            )
            logger.debug("FAISS search completed in %.3fs", faiss_elapsed)

            # Step 3: Fetch documents from database
            logger.debug("Fetching documents from database")
            doc_fetch_start = time.time()
            doc_ids_batch = [row.tolist() for row in indices]
            with (
                tracer.start_as_current_span("retrieval.document_fetch"),
                profiler.track("retrieval.document_fetch"),
            ):
                documents_batch = document_store.fetch_documents_batch(
                    doc_ids_batch, truncate_length=settings.truncate_length
                )
            doc_fetch_elapsed = time.time() - doc_fetch_start
            document_fetch_duration.observe(doc_fetch_elapsed)
            stage_duration_gauge.labels(node=NODE_LABEL, stage="retrieval.document_fetch").set(
                doc_fetch_elapsed
            )
            logger.debug("Documents fetched in %.3fs", doc_fetch_elapsed)

            # Step 4: Build response
            response_items: list[RetrievalResponseItem] = []
            for idx, item in enumerate(request.items):
                docs = documents_batch[idx]
                scores = distances[idx].tolist()

                retrieval_docs: list[RetrievalDocument] = []
                for doc, score in zip(docs, scores, strict=False):
                    retrieval_docs.append(
                        RetrievalDocument(
                            doc_id=doc.doc_id,
                            title=doc.title,
                            content=doc.content,
                            category=doc.category or "",
                            score=float(score),
                        )
                    )

                response_items.append(
                    RetrievalResponseItem(
                        request_id=item.request_id,
                        docs=retrieval_docs,
                    )
                )

        # Record total duration
        total_duration = time.time() - start_time
        retrieval_request_duration.observe(total_duration)
        pipeline_latency_histogram.labels(node=NODE_LABEL, service="retrieval").observe(
            total_duration
        )
        pipeline_request_counter.labels(
            node=NODE_LABEL,
            service="retrieval",
            status="success",
        ).inc()

        if summary := profiler.summary():
            logger.info(
                json.dumps(
                    {
                        "event": "retrieval_profile",
                        "batch_id": request.batch_id,
                        "summary": summary,
                    }
                )
            )

        logger.info(
            "Retrieval batch %s completed in %.3fs (avg %.3fms per item)",
            request.batch_id,
            total_duration,
            (total_duration / batch_size) * 1000,
        )

        return RetrievalResponse(
            batch_id=request.batch_id,
            items=response_items,
        )

    except ValueError as e:
        _record_pipeline_error("validation")
        logger.exception("Validation error in retrieval: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except HTTPException:
        raise

    except Exception as e:
        _record_pipeline_error("unknown")
        logger.exception("Error processing retrieval batch: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process retrieval batch: {e!s}",
        ) from e


@app.get("/metrics", response_class=Response)
@app.head("/metrics", response_class=Response)
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns:
        Metrics in Prometheus text format
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
