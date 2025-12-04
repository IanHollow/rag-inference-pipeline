"""
Retrieval service API

Provides embedding generation, FAISS ANN search, and document retrieval.
"""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, TypeVar, cast

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
import lz4.frame  # type: ignore[import-untyped]
import msgspec
import numpy as np
from opentelemetry import trace
from prometheus_client import REGISTRY, Counter, Histogram, generate_latest

from pipeline.component_registry import ComponentRegistry
from pipeline.components.document_store import Document as StoreDocument
from pipeline.components.schemas import Document
from pipeline.config import get_settings
from pipeline.dependencies import get_registry
from pipeline.services.gateway.batch_scheduler import Batch, BatchScheduler
from pipeline.services.gateway.schemas import PendingRequest, PendingRequestStruct
from pipeline.telemetry import (
    SampledStageProfiler,
    batch_size_histogram as pipeline_batch_size_histogram,
    error_counter as pipeline_error_counter,
    get_resource_snapshot,
    latency_histogram as pipeline_latency_histogram,
    memory_gauge,
    request_counter as pipeline_request_counter,
)
from pipeline.utils.cache import LRUCache

from .schemas import (
    ErrorResponse,
    RetrievalDocument,
    RetrievalRequest,
    RetrievalRequestItem,
    RetrievalResponse,
    RetrievalResponseItem,
)


if TYPE_CHECKING:
    from pipeline.components.document_store import DocumentStore
    from pipeline.components.embedding import EmbeddingGenerator
    from pipeline.components.faiss_store import FAISSStore
    from pipeline.components.reranker import Reranker

MetricType = Counter | Histogram
T = TypeVar("T", bound=MetricType)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_embeddings_from_requests(
    requests: Sequence[PendingRequest | PendingRequestStruct],
) -> np.ndarray:
    """
    Extract embeddings from batch requests, validating all are present.

    Raises:
        ValueError: If any request is missing an embedding.
    """
    embedding_list = []
    for req in requests:
        if req.embedding is None:
            msg = "Missing embedding in batch"
            raise ValueError(msg)
        embedding_list.append(req.embedding)
    return np.array(embedding_list).astype("float32")


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
) -> RetrievalResponse:
    """
    Process a batch of retrieval requests.
    """
    start_time = time.time()
    _record_memory_usage()

    # Validate request
    if not retrieval_request.items:
        _record_pipeline_error("validation")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch must contain at least one item",
        )

    retrieval_requests_total.inc()

    executor = await get_executor(request)

    # Check readiness before processing
    faiss_store = executor.registry.get("faiss_store")
    embedding_generator = executor.registry.get("embedding_generator")

    if not faiss_store or not getattr(faiss_store, "is_loaded", False):
        _record_pipeline_error("service_unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready (FAISS store not loaded)",
        )

    # Only check embedding generator if we actually need it (missing embeddings in request)
    needs_embedding = any(item.embedding is None for item in retrieval_request.items)
    # If we need embeddings, the generator must be present and loaded
    if needs_embedding and (
        not embedding_generator or not getattr(embedding_generator, "is_loaded", False)
    ):
        _record_pipeline_error("service_unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready (Embedding generator not loaded)",
        )

    try:
        # Enqueue all items
        futures = [executor.process_request(item) for item in retrieval_request.items]
        results = await asyncio.gather(*futures)

        # Record metrics
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

        return RetrievalResponse(
            batch_id=retrieval_request.batch_id,
            items=results,
        )

    except ValueError as e:
        _record_pipeline_error("validation")
        logger.warning("Validation error in retrieval batch: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        _record_pipeline_error("unknown")
        logger.exception("Error processing retrieval batch")
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


class RetrievalExecutor:
    """
    Executor for retrieval requests using adaptive batching.
    """

    def __init__(self, registry: ComponentRegistry) -> None:
        self.registry = registry
        self.scheduler: BatchScheduler[RetrievalResponseItem] = BatchScheduler(
            batch_size=settings.retrieval_batch_size,
            max_batch_delay_ms=settings.retrieval_max_batch_delay_ms,
            process_batch_fn=self._process_batch,
            service_name="retrieval",
            enable_adaptive=getattr(settings, "enable_adaptive_batching", True),
        )
        # Initialize cache
        self.cache = LRUCache[str, tuple[list[int], list[float]]](
            capacity=settings.retrieval_cache_capacity,
            ttl=settings.cache_max_ttl,
            name="retrieval_faiss_cache",
        )
        self._lock = threading.Lock()

    async def start(self) -> None:
        await self.scheduler.start()

    async def stop(self) -> None:
        await self.scheduler.stop()

    def clear_cache(self) -> None:
        """Clear the retrieval cache safely."""
        with self._lock:
            self.cache.clear()

    async def process_request(self, item: RetrievalRequestItem) -> RetrievalResponseItem:
        req = PendingRequest(
            request_id=item.request_id,
            query=item.query,
            embedding=item.embedding,
            timestamp=time.time(),
        )
        return await self.scheduler.enqueue(req)

    async def _process_batch(
        self, batch: Batch[RetrievalResponseItem]
    ) -> list[RetrievalResponseItem]:
        """
        Process a batch of retrieval requests.

        FAISS search runs in a thread pool executor to:
        1. Allow FAISS to use multiple threads (faiss.omp_set_num_threads)
        2. Prevent blocking the async event loop during CPU-intensive search
        This improves throughput by allowing concurrent request handling.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._process_batch_sync, batch)

    def _get_embeddings(
        self,
        batch: Batch[RetrievalResponseItem],
        profiler: SampledStageProfiler,
    ) -> np.ndarray:
        """Get embeddings from requests or generate them."""
        # Check if embeddings are provided in the first request
        if batch.requests[0].embedding is not None:
            return _extract_embeddings_from_requests(batch.requests)

        embedding_generator = cast(
            "EmbeddingGenerator | None", self.registry.get("embedding_generator")
        )
        if not embedding_generator:
            msg = "Embedding generator not available"
            raise RuntimeError(msg)

        queries = [req.query for req in batch.requests]
        emb_start = time.time()
        with profiler.track("retrieval.embedding"):
            embeddings = embedding_generator.encode(queries)
        embedding_duration.observe(time.time() - emb_start)
        logger.info("Embedding generation completed in %.3fs", time.time() - emb_start)
        return embeddings

    def _search_faiss_with_cache(
        self,
        embeddings: np.ndarray,
        batch_size: int,
        profiler: SampledStageProfiler,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Perform FAISS search with optional caching."""
        faiss_store = cast("FAISSStore", self.registry.get("faiss_store"))
        doc_ids_batch: list[list[int]] = [[] for _ in range(batch_size)]
        distances_batch: list[list[float]] = [[] for _ in range(batch_size)]

        if getattr(settings, "disable_cache_for_profiling", True):
            with profiler.track("retrieval.faiss_search"):
                distances, indices = faiss_store.search(embeddings, settings.retrieval_k)
            return [row.tolist() for row in indices], [row.tolist() for row in distances]

        # With caching: check cache first
        missing_indices = []
        missing_embeddings = []

        for i in range(batch_size):
            embedding = embeddings[i]
            key = hashlib.sha256(embedding.tobytes()).hexdigest()
            with self._lock:
                cached = self.cache.get(key)
            if cached:
                doc_ids_batch[i], distances_batch[i] = cached
            else:
                missing_indices.append(i)
                missing_embeddings.append(embedding)

        if missing_embeddings:
            missing_embeddings_np = np.array(missing_embeddings)
            with profiler.track("retrieval.faiss_search"):
                distances, indices = faiss_store.search(missing_embeddings_np, settings.retrieval_k)

            for i, idx in enumerate(missing_indices):
                doc_ids = indices[i].tolist()
                dists = distances[i].tolist()
                doc_ids_batch[idx] = doc_ids
                distances_batch[idx] = dists

                # Update cache
                key = hashlib.sha256(missing_embeddings[i].tobytes()).hexdigest()
                with self._lock:
                    self.cache.put(key, (doc_ids, dists))
        else:
            logger.debug("Cache hit for all %d retrieval items", batch_size)

        return doc_ids_batch, distances_batch

    def _fetch_documents(
        self,
        doc_ids_batch: list[list[int]],
        batch_size: int,
        profiler: SampledStageProfiler,
    ) -> list[list[StoreDocument]]:
        """Fetch documents from the document store."""
        document_store = cast("DocumentStore | None", self.registry.get("document_store"))

        if not document_store:
            # Return dummy docs
            return [
                [StoreDocument(doc_id=doc_id, title="", content="") for doc_id in doc_ids]
                for doc_ids in doc_ids_batch
            ]

        mode = getattr(settings, "documents_payload_mode", "full")

        if mode == "id_only":
            return [
                [StoreDocument(doc_id=doc_id, title="", content="") for doc_id in doc_ids]
                for doc_ids in doc_ids_batch
            ]

        # Fetch documents (compressed or full mode)
        fetch_start = time.time()
        with profiler.track("retrieval.document_fetch"):
            max_workers = min(settings.cpu_worker_threads, batch_size)
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            document_store.fetch_documents_batch,
                            [doc_ids],
                            truncate_length=settings.truncate_length,
                        )
                        for doc_ids in doc_ids_batch
                    ]
                    documents_batch = [f.result()[0] for f in futures]
            else:
                documents_batch = document_store.fetch_documents_batch(
                    doc_ids_batch, truncate_length=settings.truncate_length
                )
        document_fetch_duration.observe(time.time() - fetch_start)
        if mode != "compressed":
            logger.info("Document fetch completed in %.3fs", time.time() - fetch_start)
        return documents_batch

    def _build_response_item(
        self,
        req: PendingRequest | PendingRequestStruct,
        docs: list[StoreDocument],
        scores: list[float],
        reranker: "Reranker | None",
    ) -> RetrievalResponseItem:
        """Build a single retrieval response item with optional reranking."""
        retrieval_docs = [
            RetrievalDocument(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                category=doc.category or "",
                score=float(score),
            )
            for doc, score in zip(docs, scores, strict=False)
        ]

        if reranker:
            reranker_input = [
                Document(
                    doc_id=d.doc_id,
                    title=d.title,
                    content=d.content,
                    category=d.category,
                )
                for d in retrieval_docs
            ]
            reranked = reranker.rerank(req.query, reranker_input)
            retrieval_docs = [
                RetrievalDocument(
                    doc_id=d.doc_id,
                    title=d.title,
                    content=d.content,
                    category=d.category,
                    score=d.score,
                )
                for d in reranked
            ]

        # Handle compression
        compressed_payload = None
        mode = getattr(settings, "documents_payload_mode", "full")
        if mode == "compressed":
            docs_dicts = [d.model_dump() for d in retrieval_docs]
            serialized = msgspec.json.encode(docs_dicts)
            compressed_payload = lz4.frame.compress(serialized)
            retrieval_docs = []

        return RetrievalResponseItem(
            request_id=req.request_id,
            docs=retrieval_docs,
            compressed_docs=compressed_payload,
        )

    def _process_batch_sync(
        self, batch: Batch[RetrievalResponseItem]
    ) -> list[RetrievalResponseItem]:
        """
        Synchronous processing of a batch of retrieval requests.
        """
        faiss_store = cast("FAISSStore", self.registry.get("faiss_store"))
        reranker = cast("Reranker | None", self.registry.get("reranker"))

        if not faiss_store:
            msg = "FAISS store not available"
            raise RuntimeError(msg)

        profiler = SampledStageProfiler(
            enabled=settings.enable_profiling,
            sample_rate=settings.profiling_sample_rate,
            logger=logger,
        )

        batch_size = len(batch)
        retrieval_batch_size.observe(batch_size)
        pipeline_batch_size_histogram.labels(
            run_id=settings.profiling_run_id, node=NODE_LABEL, service="retrieval"
        ).observe(batch_size)

        with tracer.start_as_current_span(
            "retrieval.batch_exec",
            attributes={
                "pipeline.batch_id": batch.batch_id,
                "pipeline.service": "retrieval",
                "pipeline.node": settings.node_number,
            },
        ):
            # Step 1: Get embeddings
            embeddings = self._get_embeddings(batch, profiler)

            # Step 2: FAISS Search
            faiss_start = time.time()
            doc_ids_batch, distances_batch = self._search_faiss_with_cache(
                embeddings, batch_size, profiler
            )
            faiss_search_duration.observe(time.time() - faiss_start)
            logger.info("FAISS search completed in %.3fs", time.time() - faiss_start)

            # Step 3: Fetch Documents
            documents_batch = self._fetch_documents(doc_ids_batch, batch_size, profiler)

            # Step 4: Build Response Items with optional reranking
            if reranker and batch_size > 1:
                max_workers = min(settings.cpu_worker_threads, batch_size)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    response_items = list(
                        executor.map(
                            lambda args: self._build_response_item(
                                args[0], args[1], args[2], reranker
                            ),
                            zip(batch.requests, documents_batch, distances_batch, strict=False),
                        )
                    )
            else:
                response_items = [
                    self._build_response_item(req, docs, scores, reranker)
                    for req, docs, scores in zip(
                        batch.requests, documents_batch, distances_batch, strict=False
                    )
                ]

            return response_items


class _ExecutorContainer:
    """Container for singleton executor instance to avoid global statement."""

    instance: RetrievalExecutor | None = None


_executor_container = _ExecutorContainer()


async def start_retrieval_executor(registry: ComponentRegistry) -> None:
    """Start the retrieval executor."""
    if _executor_container.instance is None:
        _executor_container.instance = RetrievalExecutor(registry)
        await _executor_container.instance.start()
        logger.info("RetrievalExecutor started")


async def stop_retrieval_executor() -> None:
    """Stop the retrieval executor."""
    if _executor_container.instance:
        await _executor_container.instance.stop()
        _executor_container.instance = None
        logger.info("RetrievalExecutor stopped")


async def get_executor(request: Request) -> RetrievalExecutor:
    """Get the retrieval executor instance."""
    if _executor_container.instance is None:
        # Fallback for tests or if not started via lifecycle
        registry = get_registry(request)
        await start_retrieval_executor(registry)

    if _executor_container.instance is None:
        msg = "Failed to initialize RetrievalExecutor"
        raise RuntimeError(msg)

    return _executor_container.instance


@router.post("/clear_cache")
async def clear_cache(request: Request) -> dict[str, str]:
    """Clear all caches."""
    executor = await get_executor(request)
    with executor._lock:
        executor.cache.clear()

    # Clear component caches
    registry = get_registry(request)
    if (embedding_generator := registry.get("embedding_generator")) and hasattr(
        embedding_generator, "cache"
    ):
        embedding_generator.cache.clear()

    if (document_store := registry.get("document_store")) and hasattr(document_store, "cache"):
        document_store.cache.clear()

    return {"status": "cleared"}
