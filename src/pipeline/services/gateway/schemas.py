"""
Pydantic schemas for gateway service requests and responses.
"""

from pydantic import BaseModel, Field

# === Client schemas ===


class QueryRequest(BaseModel):
    """Request from client to the /query endpoint."""

    request_id: str = Field(..., description="Unique identifier for this request")
    query: str = Field(..., min_length=1, description="User's query string")


class QueryResponse(BaseModel):
    """Response to client from the /query endpoint."""

    request_id: str = Field(..., description="Echo of the request ID")
    generated_response: str = Field(..., description="LLM generated response")
    sentiment: str = Field(
        ...,
        description="Sentiment: 'very negative', 'negative', 'neutral', 'positive', 'very positive'",
    )
    is_toxic: str = Field(..., description="'true' or 'false'")


# === Internal RPC schemas ===


class RetrievalRequest(BaseModel):
    """Request sent to Node 1 /retrieve endpoint."""

    request_id: str = Field(..., description="Request identifier")
    query: str = Field(..., description="Query text to embed and retrieve for")
    embedding: list[float] | None = Field(None, description="Pre-computed embedding vector")


class RetrievalResponse(BaseModel):
    """Response from Node 1 /retrieve endpoint."""

    request_id: str = Field(..., description="Echo of request ID")
    docs: list[dict[str, str | int | float]] = Field(
        ..., description="List of retrieved documents with doc_id, title, snippet, score"
    )


class GenerationRequest(BaseModel):
    """Request sent to Node 2 /generate endpoint."""

    request_id: str = Field(..., description="Request identifier")
    query: str = Field(..., description="Original query")
    docs: list[dict[str, str | int | float]] = Field(
        ..., description="Retrieved documents from Node 1"
    )


class GenerationResponse(BaseModel):
    """Response from Node 2 /generate endpoint."""

    request_id: str = Field(..., description="Echo of request ID")
    generated_response: str = Field(..., description="LLM generated text")
    sentiment: str | None = Field(None, description="Sentiment classification")
    is_toxic: str | None = Field(None, description="'true' or 'false'")


# === Internal batch processing ===


class PendingRequest(BaseModel):
    """Internal representation of a request awaiting batch processing."""

    request_id: str
    query: str
    embedding: list[float] | None = None
    timestamp: float = Field(..., description="Time request was received")
