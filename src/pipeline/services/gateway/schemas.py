"""
Pydantic schemas for gateway service requests and responses.
"""

import base64

import msgspec
from pydantic import Field, field_validator

from pipeline.base_schemas import BaseJSONModel

# === Client schemas ===


class QueryRequest(BaseJSONModel):
    """Request from client to the /query endpoint."""

    request_id: str = Field(..., description="Unique identifier for this request")
    query: str = Field(..., min_length=1, description="User's query string")


class QueryResponse(BaseJSONModel):
    """Response to client from the /query endpoint."""

    request_id: str = Field(..., description="Echo of the request ID")
    generated_response: str = Field(..., description="LLM generated response")
    sentiment: str = Field(
        ...,
        description="Sentiment: 'very negative', 'negative', 'neutral', 'positive', 'very positive'",
    )
    is_toxic: str = Field(..., description="'true' or 'false'")


# === Internal RPC schemas ===


class RetrievalRequest(BaseJSONModel):
    """Request sent to Node 1 /retrieve endpoint."""

    request_id: str = Field(..., description="Request identifier")
    query: str = Field(..., description="Query text to embed and retrieve for")
    embedding: list[float] | None = Field(None, description="Pre-computed embedding vector")


class RetrievalResponse(BaseJSONModel):
    """Response from Node 1 /retrieve endpoint."""

    request_id: str = Field(..., description="Echo of request ID")
    docs: list[dict[str, str | int | float]] = Field(
        ..., description="List of retrieved documents with doc_id, title, snippet, score"
    )
    compressed_docs: bytes | None = Field(None, description="Compressed documents payload")

    @field_validator("compressed_docs", mode="before")
    @classmethod
    def decode_base64(cls, v: str | bytes | None) -> bytes | None:
        if isinstance(v, str):
            return base64.b64decode(v)
        return v


class GenerationRequest(BaseJSONModel):
    """Request sent to Node 2 /generate endpoint."""

    request_id: str = Field(..., description="Request identifier")
    query: str = Field(..., description="Original query")
    docs: list[dict[str, str | int | float]] = Field(
        ..., description="Retrieved documents from Node 1"
    )
    compressed_docs: bytes | None = Field(None, description="Compressed documents payload")

    @field_validator("compressed_docs", mode="before")
    @classmethod
    def decode_base64(cls, v: str | bytes | None) -> bytes | None:
        if isinstance(v, str):
            return base64.b64decode(v)
        return v


class GenerationResponse(BaseJSONModel):
    """Response from Node 2 /generate endpoint."""

    request_id: str = Field(..., description="Echo of request ID")
    generated_response: str = Field(..., description="LLM generated text")
    sentiment: str | None = Field(None, description="Sentiment classification")
    is_toxic: str | None = Field(None, description="'true' or 'false'")


# === Internal batch processing ===


class PendingRequest(BaseJSONModel):
    """Internal representation of a request awaiting batch processing."""

    request_id: str
    query: str
    embedding: list[float] | None = None
    docs: list[dict[str, str | int | float]] | None = None
    compressed_docs: bytes | None = None
    timestamp: float = Field(..., description="Time request was received")

    @field_validator("compressed_docs", mode="before")
    @classmethod
    def decode_base64(cls, v: str | bytes | None) -> bytes | None:
        if isinstance(v, str):
            return base64.b64decode(v)
        return v


# === Msgspec Structs ===


class PendingRequestStruct(msgspec.Struct):
    """Msgspec equivalent of PendingRequest."""

    request_id: str
    query: str
    timestamp: float
    embedding: list[float] | None = None
    docs: list[dict[str, str | int | float]] | None = None
    compressed_docs: bytes | None = None


class RetrievalRequestStruct(msgspec.Struct):
    """Msgspec equivalent of RetrievalRequest."""

    request_id: str
    query: str
    embedding: list[float] | None = None


class RetrievalResponseStruct(msgspec.Struct):
    """Msgspec equivalent of RetrievalResponse."""

    request_id: str
    docs: list[dict[str, str | int | float]]
    compressed_docs: bytes | None = None


class GenerationRequestStruct(msgspec.Struct):
    """Msgspec equivalent of GenerationRequest."""

    request_id: str
    query: str
    docs: list[dict[str, str | int | float]]
    compressed_docs: bytes | None = None


class GenerationResponseStruct(msgspec.Struct):
    """Msgspec equivalent of GenerationResponse."""

    request_id: str
    generated_response: str
    sentiment: str | None = None
    is_toxic: str | None = None
