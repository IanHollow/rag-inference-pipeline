"""
Generation service module.

Provides reranking, LLM generation, sentiment analysis, and toxicity filtering.
"""

from pipeline.components.llm import LLMGenerator
from pipeline.components.reranker import Reranker
from pipeline.components.sentiment import SentimentAnalyzer
from pipeline.components.toxicity import ToxicityFilter

from .api import router
from .schemas import (
    Document,
    ErrorResponse,
    GenerationRequest,
    GenerationRequestItem,
    GenerationResponse,
    GenerationResponseItem,
    HealthResponse,
    RerankedDocument,
    StageMetrics,
)

__all__ = [
    "Document",
    "ErrorResponse",
    "GenerationRequest",
    "GenerationRequestItem",
    "GenerationResponse",
    "GenerationResponseItem",
    "HealthResponse",
    "LLMGenerator",
    "RerankedDocument",
    "Reranker",
    "SentimentAnalyzer",
    "StageMetrics",
    "ToxicityFilter",
    "router",
]
