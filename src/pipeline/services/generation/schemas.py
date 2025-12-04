"""
Schema definitions for the generation service.

Defines request/response models for generation, reranking, sentiment, and toxicity analysis.
"""

import base64
import logging

import msgspec
from pydantic import Field, field_validator

from pipeline.base_schemas import BaseJSONModel
from pipeline.components.schemas import Document, DocumentStruct, RerankedDocument


logger = logging.getLogger(__name__)

__all__ = [
    "Document",
    "ErrorResponse",
    "GenerationRequest",
    "GenerationRequestItem",
    "GenerationResponse",
    "GenerationResponseItem",
    "HealthResponse",
    "RerankedDocument",
    "StageMetrics",
]


class GenerationRequestItem(BaseJSONModel):
    """Single item in a generation batch request."""

    request_id: str = Field(..., description="Unique request identifier")
    query: str = Field(..., description="User query")
    docs: list[Document] = Field(..., description="Retrieved documents for this query")
    compressed_docs: bytes | None = Field(None, description="Compressed documents payload")

    @field_validator("compressed_docs", mode="before")
    @classmethod
    def decode_base64(cls, v: str | bytes | None) -> bytes | None:
        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except Exception:
                logger.exception("Base64 decode failed")
                raise
        return v


class GenerationRequest(BaseJSONModel):
    """Batch generation request."""

    batch_id: str = Field(..., description="Unique batch identifier")
    items: list[GenerationRequestItem] = Field(..., description="Batch items to process")


class GenerationResponseItem(BaseJSONModel):
    """Single item in a generation batch response."""

    request_id: str = Field(..., description="Unique request identifier")
    generated_response: str = Field(..., description="Generated LLM response")
    sentiment: str | None = Field(
        None,
        description="Sentiment analysis result: 'very negative', 'negative', 'neutral', 'positive', or 'very positive'",
    )
    is_toxic: str | None = Field(None, description="Toxicity flag: 'true' or 'false'")


class GenerationResponse(BaseJSONModel):
    """Batch generation response."""

    batch_id: str = Field(..., description="Unique batch identifier")
    items: list[GenerationResponseItem] = Field(..., description="Processed items")
    processing_time: float = Field(..., description="Total processing time in seconds")


class HealthResponse(BaseJSONModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    node: int = Field(..., description="Node number")
    role: str = Field(..., description="Node role")
    total_nodes: int = Field(..., description="Total nodes in cluster")
    models_loaded: bool = Field(..., description="Whether all models are loaded")


class ErrorResponse(BaseJSONModel):
    """Error response."""

    detail: str = Field(..., description="Error message")


class StageMetrics(BaseJSONModel):
    """Metrics for a single processing stage."""

    stage: str = Field(..., description="Stage name")
    duration: float = Field(..., description="Duration in seconds")
    batch_size: int = Field(..., description="Number of items processed")


# === Msgspec Structs ===


class GenerationRequestItemStruct(msgspec.Struct):
    """Msgspec equivalent of GenerationRequestItem."""

    request_id: str
    query: str
    docs: list[DocumentStruct]
    compressed_docs: bytes | None = None


class GenerationRequestStruct(msgspec.Struct):
    """Msgspec equivalent of GenerationRequest."""

    batch_id: str
    items: list[GenerationRequestItemStruct]


class GenerationResponseItemStruct(msgspec.Struct):
    """Msgspec equivalent of GenerationResponseItem."""

    request_id: str
    generated_response: str
    sentiment: str | None = None
    is_toxic: str | None = None


class GenerationResponseStruct(msgspec.Struct):
    """Msgspec equivalent of GenerationResponse."""

    batch_id: str
    items: list[GenerationResponseItemStruct]
    processing_time: float
