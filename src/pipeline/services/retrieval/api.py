"""
Retrieval service API

Provides embedding generation, FAISS ANN search, and document retrieval.
"""

from collections.abc import Sequence
import json
import logging
import time
from typing import Annotated, Any, TypeVar, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from opentelemetry import trace
from prometheus_client import REGISTRY, Counter, Histogram, generate_latest

from ...components.document_store import DocumentStore
from ...components.embedding import EmbeddingGenerator
from ...components.faiss_store import FAISSStore
from ...components.reranker import Reranker
from ...components.schemas import Document
from ...config import get_settings
from ...dependencies import get_component
from ...telemetry import (
    SampledStageProfiler,
    batch_size_histogram as pipeline_batch_size_histogram,
    error_counter as pipeline_error_counter,
    get_resource_snapshot,
    latency_histogram as pipeline_latency_histogram,
    memory_gauge,
    request_counter as pipeline_request_counter,
    stage_duration_gauge,
)
from .schemas import (
    ErrorResponse,
    RetrievalDocument,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResponseItem,
)

MetricType = Counter | Histogram
T = TypeVar("T", bound=MetricType)

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


# Prometheus metrics
retrieval_requests_total = get_metric(
    "retrieval_requests_total",
    Counter,
    "Total number of retrieval requests",
)

retrieval_request_duration = get_metric(
    "retrieval_request_duration_seconds",
    Histogram,
    "Retrieval request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

retrieval_batch_size = get_metric(
    "retrieval_batch_size",
    Histogram,
    "Number of items in retrieval batch",
    buckets=[1, 2, 4, 8, 16, 32],
)

faiss_search_duration = get_metric(
    "faiss_search_duration_seconds",
    Histogram,
    "FAISS search duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

document_fetch_duration = get_metric(
    "document_fetch_duration_seconds",
    Histogram,
    "Document fetch duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

embedding_duration = get_metric(
    "embedding_duration_seconds",
    Histogram,
    "Embedding generation duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)


# Create APIRouter
router = APIRouter()


def _record_memory_usage() -> None:
    """Record current process memory in Prometheus gauges."""
    snapshot = get_resource_snapshot()
    memory_gauge.labels(
        run_id=settings.profiling_run_id, node=NODE_LABEL, service="retrieval", type="rss"
    ).set(snapshot.rss)
    memory_gauge.labels(
        run_id=settings.profiling_run_id, node=NODE_LABEL, service="retrieval", type="vms"
    ).set(snapshot.vms)
    memory_gauge.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="retrieval",
        type="percent",
    ).set(snapshot.memory_percent)


def _record_pipeline_error(error_type: str) -> None:
    """Increment shared error counters for this service."""
    pipeline_error_counter.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="retrieval",
        error_type=error_type,
    ).inc()
    pipeline_request_counter.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="retrieval",
        status="error",
    ).inc()


@router.post(
    "",
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        400: {"model": ErrorResponse, "description": "Bad request"},
    },
)
async def retrieve(
    request: Request,
    retrieval_request: RetrievalRequest,
    faiss_store: Annotated[FAISSStore, Depends(get_component("faiss_store"))],
    document_store: Annotated[DocumentStore, Depends(get_component("document_store"))],
    embedding_generator: Annotated[
        EmbeddingGenerator | None, Depends(get_component("embedding_generator"))
    ] = None,
    reranker: Annotated[Reranker | None, Depends(get_component("reranker"))] = None,
) -> RetrievalResponse:
    """
    Process a batch of retrieval requests.
    """
    profiler = SampledStageProfiler(
        enabled=settings.enable_profiling,
        sample_rate=settings.profiling_sample_rate,
        logger=logger,
    )
    start_time = time.time()

    _record_memory_usage()

    # Validate service state
    if faiss_store is None or not getattr(faiss_store, "is_loaded", False):
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
    if not retrieval_request.items:
        _record_pipeline_error("validation")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch must contain at least one item",
        )

    batch_size = len(retrieval_request.items)
    retrieval_requests_total.inc()
    retrieval_batch_size.observe(batch_size)
    pipeline_batch_size_histogram.labels(
        run_id=settings.profiling_run_id, node=NODE_LABEL, service="retrieval"
    ).observe(batch_size)

    logger.info(
        "Processing retrieval batch: %s with %d items", retrieval_request.batch_id, batch_size
    )

    try:
        with tracer.start_as_current_span(
            "retrieval.batch",
            attributes={
                "pipeline.batch_id": retrieval_request.batch_id,
                "pipeline.service": "retrieval",
                "pipeline.node": settings.node_number,
            },
        ):
            # Step 1: Get embeddings (either from request or generate locally)
            embeddings = None

            # Check if embeddings are provided in the request
            if retrieval_request.items[0].embedding is not None:
                logger.debug("Using embeddings provided in request")
                try:
                    # Extract embeddings from all items
                    embedding_list = []
                    for item in retrieval_request.items:
                        if item.embedding is None:
                            raise ValueError("Missing embedding for item in batch")
                        embedding_list.append(item.embedding)

                    import numpy as np

                    embeddings = np.array(embedding_list).astype("float32")
                except ValueError as e:
                    _record_pipeline_error("validation")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid embeddings in request: {e}",
                    ) from e
            else:
                # Generate embeddings locally
                if embedding_generator is None or not getattr(
                    embedding_generator, "is_loaded", False
                ):
                    _record_pipeline_error("embedding_not_loaded")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Embedding model not loaded and no embeddings provided",
                    )

                queries = [item.query for item in retrieval_request.items]
                logger.debug("Generating embeddings for %d queries", len(queries))

                embed_start = time.time()
                with (
                    tracer.start_as_current_span("retrieval.embedding"),
                    profiler.track("retrieval.embedding"),
                ):
                    embeddings = embedding_generator.encode(queries)
                embed_elapsed = time.time() - embed_start
                embedding_duration.observe(embed_elapsed)
                stage_duration_gauge.labels(
                    run_id=settings.profiling_run_id, node=NODE_LABEL, stage="retrieval.embedding"
                ).set(embed_elapsed)
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
            stage_duration_gauge.labels(
                run_id=settings.profiling_run_id, node=NODE_LABEL, stage="retrieval.faiss_search"
            ).set(faiss_elapsed)
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
            stage_duration_gauge.labels(
                run_id=settings.profiling_run_id, node=NODE_LABEL, stage="retrieval.document_fetch"
            ).set(doc_fetch_elapsed)
            logger.debug("Documents fetched in %.3fs", doc_fetch_elapsed)

            # Step 4: Build response
            response_items: list[RetrievalResponseItem] = []
            for idx, item in enumerate(retrieval_request.items):
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

                # Step 5: Rerank if reranker is available
                if reranker and getattr(reranker, "is_loaded", False):
                    with (
                        tracer.start_as_current_span("retrieval.rerank"),
                        profiler.track("retrieval.rerank"),
                    ):
                        # Reranker expects list[Document], RetrievalDocument is compatible
                        # Convert RetrievalDocument to Document for reranker
                        reranker_input_docs = [
                            Document(
                                doc_id=doc.doc_id,
                                title=doc.title,
                                content=doc.content,
                                category=doc.category,
                            )
                            for doc in retrieval_docs
                        ]
                        reranked_results = reranker.rerank(item.query, reranker_input_docs)

                        # Convert RerankedDocument back to RetrievalDocument
                        retrieval_docs = [
                            RetrievalDocument(
                                doc_id=doc.doc_id,
                                title=doc.title,
                                content=doc.content,
                                category=doc.category,
                                score=doc.score,
                            )
                            for doc in reranked_results
                        ]

                response_items.append(
                    RetrievalResponseItem(
                        request_id=item.request_id,
                        docs=retrieval_docs,
                    )
                )

        # Record total duration
        total_duration = time.time() - start_time
        retrieval_request_duration.observe(total_duration)
        pipeline_latency_histogram.labels(
            run_id=settings.profiling_run_id, node=NODE_LABEL, service="retrieval"
        ).observe(total_duration)
        pipeline_request_counter.labels(
            run_id=settings.profiling_run_id,
            node=NODE_LABEL,
            service="retrieval",
            status="success",
        ).inc()

        if summary := profiler.summary():
            logger.info(
                json.dumps(
                    {
                        "event": "retrieval_profile",
                        "batch_id": retrieval_request.batch_id,
                        "summary": summary,
                    }
                )
            )

        logger.info(
            "Retrieval batch %s completed in %.3fs (avg %.3fms per item)",
            retrieval_request.batch_id,
            total_duration,
            (total_duration / batch_size) * 1000,
        )

        return RetrievalResponse(
            batch_id=retrieval_request.batch_id,
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


@router.get("/metrics", response_class=Response)
@router.head("/metrics", response_class=Response)
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
