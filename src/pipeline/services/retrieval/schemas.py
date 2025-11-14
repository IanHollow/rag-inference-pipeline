"""
Schemas for retrieval service API.
"""

from pydantic import BaseModel, Field


class RetrievalRequestItem(BaseModel):
    """Single query item in a retrieval batch request."""

    request_id: str = Field(..., description="Unique identifier for this request")
    query: str = Field(..., description="Query text to process")


class RetrievalRequest(BaseModel):
    """Batch request for retrieval service."""

    batch_id: str = Field(..., description="Unique identifier for this batch")
    items: list[RetrievalRequestItem] = Field(..., description="List of queries to process")


class RetrievalDocument(BaseModel):
    """Document returned by retrieval service."""

    doc_id: int = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    snippet: str = Field(..., description="Document content snippet")
    score: float = Field(..., description="FAISS similarity score")


class RetrievalResponseItem(BaseModel):
    """Single result item in a retrieval batch response."""

    request_id: str = Field(..., description="Unique identifier for this request")
    docs: list[RetrievalDocument] = Field(..., description="Retrieved documents")


class RetrievalResponse(BaseModel):
    """Batch response from retrieval service."""

    batch_id: str = Field(..., description="Unique identifier for this batch")
    items: list[RetrievalResponseItem] = Field(..., description="List of results")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    node: int = Field(..., description="Node number")
    total_nodes: int = Field(..., description="Total number of nodes")
    embedding_loaded: bool = Field(..., description="Whether embedding model is loaded")
    faiss_loaded: bool = Field(..., description="Whether FAISS index is loaded")
    documents_available: bool = Field(..., description="Whether document database is available")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
