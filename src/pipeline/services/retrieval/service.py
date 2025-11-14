"""
Retrieval service

Provides embedding generation, FAISS ANN search, and document retrieval.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException, status
from prometheus_client import Counter, Histogram, generate_latest

from ...config import get_settings
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
    start_time = time.time()

    # Update metrics
    retrieval_requests_total.inc()
    retrieval_batch_size.observe(len(request.items))

    logger.info(
        "Processing retrieval batch: %s with %d items", request.batch_id, len(request.items)
    )

    # Validate service state
    if embedding_generator is None or not embedding_generator.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding model not loaded",
        )

    if faiss_store is None or not faiss_store.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FAISS index not loaded",
        )

    if document_store is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document store not initialized",
        )

    # Validate request
    if not request.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch must contain at least one item",
        )

    try:
        # Step 1: Generate embeddings for all queries
        queries = [item.query for item in request.items]
        logger.debug("Generating embeddings for %d queries", len(queries))

        embed_start = time.time()
        embeddings = embedding_generator.encode(queries)
        embedding_duration.observe(time.time() - embed_start)
        logger.debug("Embeddings generated in %.3fs", time.time() - embed_start)

        # Step 2: FAISS ANN search
        logger.debug("Performing FAISS search with k=%d", settings.retrieval_k)
        faiss_start = time.time()
        distances, indices = faiss_store.search(embeddings, settings.retrieval_k)
        faiss_search_duration.observe(time.time() - faiss_start)
        logger.debug("FAISS search completed in %.3fs", time.time() - faiss_start)

        # Step 3: Fetch documents from database
        logger.debug("Fetching documents from database")
        doc_fetch_start = time.time()

        doc_ids_batch = [row.tolist() for row in indices]
        documents_batch = document_store.fetch_documents_batch(
            doc_ids_batch, truncate_length=settings.truncate_length
        )
        document_fetch_duration.observe(time.time() - doc_fetch_start)
        logger.debug("Documents fetched in %.3fs", time.time() - doc_fetch_start)

        # Step 4: Build response
        response_items: list[RetrievalResponseItem] = []

        for idx, item in enumerate(request.items):
            # Get documents and scores for this query
            docs = documents_batch[idx]
            scores = distances[idx].tolist()

            # Create RetrievalDocument objects
            retrieval_docs: list[RetrievalDocument] = []
            for doc, score in zip(docs, scores, strict=False):
                retrieval_docs.append(
                    RetrievalDocument(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        snippet=doc.content,
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

        logger.info(
            "Retrieval batch %s completed in %.3fs (avg %.3fms per item)",
            request.batch_id,
            total_duration,
            (total_duration / len(request.items)) * 1000,
        )

        return RetrievalResponse(
            batch_id=request.batch_id,
            items=response_items,
        )

    except ValueError as e:
        logger.exception("Validation error in retrieval: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.exception("Error processing retrieval batch: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process retrieval batch: {e!s}",
        ) from e


@app.get("/metrics")
async def metrics() -> bytes:
    """
    Prometheus metrics endpoint.

    Returns:
        Metrics in Prometheus text format
    """
    return generate_latest()
