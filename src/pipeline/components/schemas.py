import msgspec
from pydantic import Field

from ..base_schemas import BaseJSONModel


class Document(BaseJSONModel):
    """Document structure for reranking."""

    doc_id: int = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field(default="", description="Document category")


class RerankedDocument(BaseJSONModel):
    """Document with reranking score."""

    doc_id: int = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field(default="", description="Document category")
    score: float = Field(..., description="Reranking score")


# === Msgspec Structs ===


class DocumentStruct(msgspec.Struct):
    """Msgspec equivalent of Document."""

    doc_id: int
    title: str
    content: str
    category: str = ""


class RerankedDocumentStruct(msgspec.Struct):
    """Msgspec equivalent of RerankedDocument."""

    doc_id: int
    title: str
    content: str
    score: float
    category: str = ""
