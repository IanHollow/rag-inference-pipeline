"""
Generation service API

Handles reranking, LLM generation, sentiment analysis, and toxicity filtering.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from prometheus_client import generate_latest

from ...components.llm import LLMGenerator
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
from .schemas import (
    ErrorResponse,
    GenerationRequest,
    GenerationResponse,
)
from .service import GenerationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
settings = get_settings()
NODE_LABEL = str(settings.node_number)


# Create APIRouter
router = APIRouter()


def _record_memory_usage() -> None:
    snapshot = get_resource_snapshot()
    memory_gauge.labels(
        run_id=settings.profiling_run_id, node=NODE_LABEL, service="generation", type="rss"
    ).set(snapshot.rss)
    memory_gauge.labels(
        run_id=settings.profiling_run_id, node=NODE_LABEL, service="generation", type="vms"
    ).set(snapshot.vms)
    memory_gauge.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="generation",
        type="percent",
    ).set(snapshot.memory_percent)


def _record_pipeline_error(error_type: str) -> None:
    pipeline_error_counter.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="generation",
        error_type=error_type,
    ).inc()
    pipeline_request_counter.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="generation",
        status="error",
    ).inc()


@router.post("", responses={500: {"model": ErrorResponse}})
async def generate(
    request: Request,
    generation_request: GenerationRequest,
    reranker: Annotated[Reranker | None, Depends(get_component("reranker"))],
    llm_generator: Annotated[LLMGenerator, Depends(get_component("llm_generator"))],
    sentiment_analyzer: Annotated[SentimentAnalyzer, Depends(get_component("sentiment_analyzer"))],
    toxicity_filter: Annotated[ToxicityFilter, Depends(get_component("toxicity_filter"))],
) -> GenerationResponse:
    """
    Generate responses for a batch of queries.
    """
    _record_memory_usage()

    # Validate models are loaded
    if (
        llm_generator is None
        or not getattr(llm_generator, "is_loaded", False)
        or sentiment_analyzer is None
        or not getattr(sentiment_analyzer, "is_loaded", False)
        or toxicity_filter is None
        or not getattr(toxicity_filter, "is_loaded", False)
    ):
        _record_pipeline_error("models_not_loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded yet",
        )

    # Check reranker if it's provided
    if reranker is not None and not getattr(reranker, "is_loaded", False):
        # If reranker is provided but not loaded, that's an error?
        # Or should we just treat it as None?
        # Let's assume if it's injected, it should be loaded if it exists.
        # But get_component might return None if not found.
        pass

    try:
        service = GenerationService(
            reranker=reranker,
            llm_generator=llm_generator,
            sentiment_analyzer=sentiment_analyzer,
            toxicity_filter=toxicity_filter,
        )
        return service.process_batch(generation_request)

    except HTTPException:
        raise

    except Exception as e:
        _record_pipeline_error("unknown")
        logger.exception("Error processing generation batch: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process generation batch: {e!s}",
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
