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
