"""
Generation service - Handles LLM generation, sentiment, and safety filtering.

This service provides text generation and analysis capabilities for the pipeline.
"""

from fastapi import FastAPI

from ..config import get_settings

settings = get_settings()

app = FastAPI(
    title="ML Pipeline Generation Service",
    description="LLM generation, sentiment analysis, and safety filtering",
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


# TODO: Implement Generation Service
# - POST /generation/generate - Generate LLM responses
# - POST /generation/sentiment - Analyze sentiment
# - POST /generation/safety - Check toxicity/safety
