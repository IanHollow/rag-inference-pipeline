"""
Schema definitions for the generation service.

Defines request/response models for generation, reranking, sentiment, and toxicity analysis.
"""

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document structure for reranking."""

    doc_id: int = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field(default="", description="Document category")


class RerankedDocument(BaseModel):
    """Document with reranking score."""

    doc_id: int = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field(default="", description="Document category")
    score: float = Field(..., description="Reranking score")


class GenerationRequestItem(BaseModel):
    """Single item in a generation batch request."""

    request_id: str = Field(..., description="Unique request identifier")
    query: str = Field(..., description="User query")
    docs: list[Document] = Field(..., description="Retrieved documents for this query")


class GenerationRequest(BaseModel):
    """Batch generation request."""

    batch_id: str = Field(..., description="Unique batch identifier")
    items: list[GenerationRequestItem] = Field(..., description="Batch items to process")


class GenerationResponseItem(BaseModel):
    """Single item in a generation batch response."""

    request_id: str = Field(..., description="Unique request identifier")
    generated_response: str = Field(..., description="Generated LLM response")
    sentiment: str = Field(
        ...,
        description="Sentiment analysis result: 'very negative', 'negative', 'neutral', 'positive', or 'very positive'",
    )
    is_toxic: str = Field(..., description="Toxicity flag: 'true' or 'false'")


class GenerationResponse(BaseModel):
    """Batch generation response."""

    batch_id: str = Field(..., description="Unique batch identifier")
    items: list[GenerationResponseItem] = Field(..., description="Processed items")
    processing_time: float = Field(..., description="Total processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    node: int = Field(..., description="Node number")
    role: str = Field(..., description="Node role")
    total_nodes: int = Field(..., description="Total nodes in cluster")
    models_loaded: bool = Field(..., description="Whether all models are loaded")


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str = Field(..., description="Error message")


class StageMetrics(BaseModel):
    """Metrics for a single processing stage."""

    stage: str = Field(..., description="Stage name")
    duration: float = Field(..., description="Duration in seconds")
    batch_size: int = Field(..., description="Number of items processed")
