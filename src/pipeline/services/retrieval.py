"""
Retrieval service - Handles embedding, FAISS search, and reranking.

This service provides document retrieval capabilities for the pipeline.
"""

from fastapi import FastAPI

from ..config import get_settings

settings = get_settings()

app = FastAPI(
    title="ML Pipeline Retrieval Service",
    description="Embedding, FAISS search, and document reranking",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str | int]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "node": settings.node_number,
        "role": settings.role.value,
        "total_nodes": settings.total_nodes,
    }


# TODO: Implement Retrieval Service
# - POST /retrieval/embed - Generate embeddings
# - POST /retrieval/search - FAISS ANN search
# - POST /retrieval/rerank - Rerank retrieved documents
