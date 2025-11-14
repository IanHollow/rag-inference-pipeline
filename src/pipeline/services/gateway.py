"""
Gateway service - Orchestrates the ML pipeline.

This service receives client requests and coordinates with retrieval and generation services.
"""

from fastapi import FastAPI

from ..config import get_settings

settings = get_settings()

app = FastAPI(
    title="ML Pipeline Gateway",
    description="Orchestrates distributed ML inference pipeline",
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


@app.post("/query")
async def query() -> dict[str, str]:
    """
    Main query endpoint - receives client requests.

    TODO: Implement Gateway and Batching
    """
    return {"status": "not_implemented", "message": "Gateway service coming in Issue 04"}
