"""
Generation service - Node 2

Handles reranking, LLM generation, sentiment analysis, and toxicity filtering.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException, status
from prometheus_client import Counter, Histogram, generate_latest

from ...config import get_settings
from .llm import LLMGenerator
from .reranker import Reranker
from .schemas import (
    ErrorResponse,
    GenerationRequest,
    GenerationResponse,
    GenerationResponseItem,
    HealthResponse,
    StageMetrics,
)
from .sentiment import SentimentAnalyzer
from .toxicity import ToxicityFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
settings = get_settings()
reranker: Reranker | None = None
llm_generator: LLMGenerator | None = None
sentiment_analyzer: SentimentAnalyzer | None = None
toxicity_filter: ToxicityFilter | None = None

# Prometheus metrics
generation_requests_total = Counter(
    "generation_requests_total",
    "Total number of generation requests",
)

generation_request_duration = Histogram(
    "generation_request_duration_seconds",
    "Generation request duration in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

generation_batch_size = Histogram(
    "generation_batch_size",
    "Number of items in generation batch",
    buckets=[1, 2, 4, 8, 16, 32],
)

rerank_duration = Histogram(
    "rerank_duration_seconds",
    "Reranking duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)

llm_generation_duration = Histogram(
    "llm_generation_duration_seconds",
    "LLM generation duration in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

sentiment_duration = Histogram(
    "sentiment_duration_seconds",
    "Sentiment analysis duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)

toxicity_duration = Histogram(
    "toxicity_duration_seconds",
    "Toxicity filtering duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager for service initialization and cleanup.

    Loads all models on startup, cleans up on shutdown.
    """
    global reranker, llm_generator, sentiment_analyzer, toxicity_filter

    logger.info("=" * 60)
    logger.info("GENERATION SERVICE STARTING")
    logger.info("=" * 60)
    logger.info("Node: %d/%d", settings.node_number, settings.total_nodes)
    logger.info("Role: %s", settings.role.value)
    logger.info("=" * 60)

    try:
        # Initialize and load reranker
        logger.info("Initializing reranker...")
        reranker = Reranker(settings)
        reranker.load()
        logger.info("Reranker loaded")

        # Initialize and load LLM
        logger.info("Initializing LLM generator...")
        llm_generator = LLMGenerator(settings)
        llm_generator.load()
        logger.info("LLM generator loaded")

        # Initialize and load sentiment analyzer
        logger.info("Initializing sentiment analyzer...")
        sentiment_analyzer = SentimentAnalyzer(settings)
        sentiment_analyzer.load()
        logger.info("Sentiment analyzer loaded")

        # Initialize and load toxicity filter
        logger.info("Initializing toxicity filter...")
        toxicity_filter = ToxicityFilter(settings)
        toxicity_filter.load()
        logger.info("Toxicity filter loaded")

        logger.info("=" * 60)
        logger.info("GENERATION SERVICE READY")
        logger.info("All models loaded and resident in memory")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.exception("Failed to initialize generation service: %s", e)
        raise

    finally:
        # Cleanup
        logger.info("Shutting down generation service...")

        if reranker is not None:
            reranker.unload()

        if llm_generator is not None:
            llm_generator.unload()

        if sentiment_analyzer is not None:
            sentiment_analyzer.unload()

        if toxicity_filter is not None:
            toxicity_filter.unload()

        logger.info("Generation service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Generation Service",
    description="Node 2: Reranking, LLM generation, sentiment, and toxicity filtering",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status and whether all models are loaded.
    """
    models_loaded = (
        reranker is not None
        and reranker.is_loaded
        and llm_generator is not None
        and llm_generator.is_loaded
        and sentiment_analyzer is not None
        and sentiment_analyzer.is_loaded
        and toxicity_filter is not None
        and toxicity_filter.is_loaded
    )

    return HealthResponse(
        status="healthy" if models_loaded else "initializing",
        node=settings.node_number,
        role=settings.role.value,
        total_nodes=settings.total_nodes,
        models_loaded=models_loaded,
    )


@app.post("/generate", responses={500: {"model": ErrorResponse}})
async def generate(request: GenerationRequest) -> GenerationResponse:
    """
    Generate responses for a batch of queries.

    Args:
        request: Batch generation request containing queries and documents

    Returns:
        GenerationResponse with processed items

    Raises:
        HTTPException: If models are not loaded or processing fails
    """
    # Validate models are loaded
    if (
        reranker is None
        or not reranker.is_loaded
        or llm_generator is None
        or not llm_generator.is_loaded
        or sentiment_analyzer is None
        or not sentiment_analyzer.is_loaded
        or toxicity_filter is None
        or not toxicity_filter.is_loaded
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded yet",
        )

    # Update metrics
    generation_requests_total.inc()
    batch_size = len(request.items)
    generation_batch_size.observe(batch_size)

    logger.info("=" * 60)
    logger.info("Processing generation batch: %s", request.batch_id)
    logger.info("Batch size: %d", batch_size)
    logger.info("=" * 60)

    start_time = time.time()
    stage_metrics: list[StageMetrics] = []

    try:
        # Extract queries and documents
        queries = [item.query for item in request.items]
        documents_batch = [list(item.docs) for item in request.items]

        # Stage 1: Rerank documents
        logger.info("[Stage 1/4] Reranking documents...")
        rerank_start = time.time()

        reranked_docs_batch = reranker.rerank_batch(
            queries=queries,
            documents_batch=documents_batch,
            top_n=settings.rerank_top_n,
        )

        rerank_elapsed = time.time() - rerank_start
        rerank_duration.observe(rerank_elapsed)
        stage_metrics.append(
            StageMetrics(
                stage="rerank",
                duration=rerank_elapsed,
                batch_size=batch_size,
            )
        )
        logger.info("Reranking completed in %.2f seconds", rerank_elapsed)

        # Stage 2: Generate LLM responses
        logger.info("[Stage 2/4] Generating LLM responses...")
        llm_start = time.time()

        generated_responses = llm_generator.generate_batch(
            queries=queries,
            reranked_docs_batch=reranked_docs_batch,
        )

        llm_elapsed = time.time() - llm_start
        llm_generation_duration.observe(llm_elapsed)
        stage_metrics.append(
            StageMetrics(
                stage="llm_generation",
                duration=llm_elapsed,
                batch_size=batch_size,
            )
        )
        logger.info("LLM generation completed in %.2f seconds", llm_elapsed)

        # Stage 3: Analyze sentiment
        logger.info("[Stage 3/4] Analyzing sentiment...")
        sentiment_start = time.time()

        sentiments = sentiment_analyzer.analyze_batch(generated_responses)

        sentiment_elapsed = time.time() - sentiment_start
        sentiment_duration.observe(sentiment_elapsed)
        stage_metrics.append(
            StageMetrics(
                stage="sentiment",
                duration=sentiment_elapsed,
                batch_size=batch_size,
            )
        )
        logger.info("Sentiment analysis completed in %.2f seconds", sentiment_elapsed)

        # Stage 4: Filter toxicity
        logger.info("[Stage 4/4] Filtering toxicity...")
        toxicity_start = time.time()

        toxicity_flags = toxicity_filter.filter_batch(generated_responses)

        toxicity_elapsed = time.time() - toxicity_start
        toxicity_duration.observe(toxicity_elapsed)
        stage_metrics.append(
            StageMetrics(
                stage="toxicity",
                duration=toxicity_elapsed,
                batch_size=batch_size,
            )
        )
        logger.info("Toxicity filtering completed in %.2f seconds", toxicity_elapsed)

        # Build response items
        response_items = [
            GenerationResponseItem(
                request_id=request.items[i].request_id,
                generated_response=generated_responses[i],
                sentiment=sentiments[i],
                is_toxic=toxicity_flags[i],
            )
            for i in range(batch_size)
        ]

        total_elapsed = time.time() - start_time
        generation_request_duration.observe(total_elapsed)

        logger.info("=" * 60)
        logger.info("Batch processing complete")
        logger.info("Total time: %.2f seconds", total_elapsed)
        logger.info("Stage breakdown:")
        for metric in stage_metrics:
            logger.info("  - %s: %.2f seconds", metric.stage, metric.duration)
        logger.info("=" * 60)

        return GenerationResponse(
            batch_id=request.batch_id,
            items=response_items,
            processing_time=total_elapsed,
        )

    except Exception as e:
        logger.exception("Error processing generation batch: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation processing failed: {e!s}",
        ) from e


@app.get("/metrics")
async def metrics() -> str:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    return generate_latest().decode("utf-8")
