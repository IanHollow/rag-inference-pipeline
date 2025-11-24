"""
Schemas for retrieval service API.
"""

import base64

import msgspec
from pydantic import Field, field_validator

from ...base_schemas import BaseJSONModel


class RetrievalRequestItem(BaseJSONModel):
    """Single query item in a retrieval batch request."""

    request_id: str = Field(..., description="Unique identifier for this request")
    query: str = Field(..., description="Query text to process")
    embedding: list[float] | None = Field(None, description="Pre-computed embedding vector")


class RetrievalRequest(BaseJSONModel):
    """Batch request for retrieval service."""

    batch_id: str = Field(..., description="Unique identifier for this batch")
    items: list[RetrievalRequestItem] = Field(..., description="List of queries to process")


class RetrievalDocument(BaseJSONModel):
    """Document returned by retrieval service."""

    doc_id: int = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field(default="", description="Document category")
    score: float = Field(..., description="FAISS similarity score")


class RetrievalResponseItem(BaseJSONModel):
    """Single result item in a retrieval batch response."""

    request_id: str = Field(..., description="Unique identifier for this request")
    docs: list[RetrievalDocument] = Field(..., description="Retrieved documents")
    compressed_docs: bytes | None = Field(None, description="Compressed documents payload")

    @field_validator("compressed_docs", mode="before")
    @classmethod
    def decode_base64(cls, v: str | bytes | None) -> bytes | None:
        if isinstance(v, str):
            return base64.b64decode(v)
        return v


class RetrievalResponse(BaseJSONModel):
    """Batch response from retrieval service."""

    batch_id: str = Field(..., description="Unique identifier for this batch")
    items: list[RetrievalResponseItem] = Field(..., description="List of results")


class HealthResponse(BaseJSONModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    node: int = Field(..., description="Node number")
    total_nodes: int = Field(..., description="Total number of nodes")
    embedding_loaded: bool = Field(..., description="Whether embedding model is loaded")
    faiss_loaded: bool = Field(..., description="Whether FAISS index is loaded")
    documents_available: bool = Field(..., description="Whether document database is available")


class ErrorResponse(BaseJSONModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")


# === Msgspec Structs ===


class RetrievalRequestItemStruct(msgspec.Struct):
    """Msgspec equivalent of RetrievalRequestItem."""

    request_id: str
    query: str
    embedding: list[float] | None = None


class RetrievalRequestStruct(msgspec.Struct):
    """Msgspec equivalent of RetrievalRequest."""

    batch_id: str
    items: list[RetrievalRequestItemStruct]


class RetrievalDocumentStruct(msgspec.Struct):
    """Msgspec equivalent of RetrievalDocument."""

    doc_id: int
    title: str
    content: str
    score: float
    category: str = ""


class RetrievalResponseItemStruct(msgspec.Struct):
    """Msgspec equivalent of RetrievalResponseItem."""

    request_id: str
    docs: list[RetrievalDocumentStruct]
    compressed_docs: bytes | None = None


class RetrievalResponseStruct(msgspec.Struct):
    """Msgspec equivalent of RetrievalResponse."""

    batch_id: str
    items: list[RetrievalResponseItemStruct]
