"""
Tests for Pydantic schema validation and serialization.

Tests cover:
- Gateway schemas (QueryRequest, QueryResponse, etc.)
- Component schemas (Document, RerankedDocument)
- Profile schemas (ProfileFile, ComponentConfig, RouteConfig)
- Base64 decoding validators
"""

import base64

from pydantic import ValidationError
import pytest

from pipeline.components.schemas import Document, RerankedDocument
from pipeline.config.profile_schema import ComponentConfig, ProfileFile, RouteConfig
from pipeline.services.gateway.schemas import (
    GenerationRequest,
    GenerationResponse,
    PendingRequest,
    QueryRequest,
    QueryResponse,
    RetrievalRequest,
    RetrievalResponse,
)


class TestQuerySchemas:
    """Tests for QueryRequest and QueryResponse schemas."""

    def test_query_request_valid(self) -> None:
        """Test valid QueryRequest."""
        request = QueryRequest(request_id="req_123", query="What is AI?")

        assert request.request_id == "req_123"
        assert request.query == "What is AI?"

    def test_query_request_empty_query_fails(self) -> None:
        """Test that empty query fails validation."""
        with pytest.raises(ValidationError):
            QueryRequest(request_id="req_123", query="")

    def test_query_request_missing_fields(self) -> None:
        """Test that missing required fields fail."""
        with pytest.raises(ValidationError):
            QueryRequest(request_id="req_123")  # type: ignore

        with pytest.raises(ValidationError):
            QueryRequest(query="test")  # type: ignore

    def test_query_response_valid(self) -> None:
        """Test valid QueryResponse."""
        response = QueryResponse(
            request_id="req_123",
            generated_response="AI is artificial intelligence.",
            sentiment="positive",
            is_toxic="false",
        )

        assert response.request_id == "req_123"
        assert response.generated_response == "AI is artificial intelligence."
        assert response.sentiment == "positive"
        assert response.is_toxic == "false"


class TestRetrievalSchemas:
    """Tests for RetrievalRequest and RetrievalResponse schemas."""

    def test_retrieval_request_minimal(self) -> None:
        """Test RetrievalRequest with minimal fields."""
        request = RetrievalRequest(request_id="req_1", query="search query")

        assert request.request_id == "req_1"
        assert request.query == "search query"
        assert request.embedding is None

    def test_retrieval_request_with_embedding(self) -> None:
        """Test RetrievalRequest with pre-computed embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        request = RetrievalRequest(
            request_id="req_1",
            query="search query",
            embedding=embedding,
        )

        assert request.embedding == embedding

    def test_retrieval_response_valid(self) -> None:
        """Test valid RetrievalResponse."""
        docs = [{"doc_id": 1, "title": "Doc 1", "snippet": "text"}]
        response = RetrievalResponse(request_id="req_1", docs=docs)

        assert response.request_id == "req_1"
        assert len(response.docs) == 1

    def test_retrieval_response_with_compressed_docs_base64(self) -> None:
        """Test RetrievalResponse with base64-encoded compressed_docs."""
        original_data = b"compressed document data"
        encoded = base64.b64encode(original_data).decode()

        response = RetrievalResponse(
            request_id="req_1",
            docs=[],
            compressed_docs=encoded,
        )

        assert response.compressed_docs == original_data

    def test_retrieval_response_with_compressed_docs_bytes(self) -> None:
        """Test RetrievalResponse with bytes compressed_docs."""
        original_data = b"compressed document data"

        response = RetrievalResponse(
            request_id="req_1",
            docs=[],
            compressed_docs=original_data,
        )

        assert response.compressed_docs == original_data


class TestGenerationSchemas:
    """Tests for GenerationRequest and GenerationResponse schemas."""

    def test_generation_request_valid(self) -> None:
        """Test valid GenerationRequest."""
        docs = [{"doc_id": 1, "title": "Title", "content": "Content"}]
        request = GenerationRequest(
            request_id="req_1",
            query="What is this about?",
            docs=docs,
        )

        assert request.request_id == "req_1"
        assert request.query == "What is this about?"
        assert len(request.docs) == 1

    def test_generation_request_with_compressed_docs(self) -> None:
        """Test GenerationRequest with compressed docs."""
        original_data = b"compressed data"
        encoded = base64.b64encode(original_data).decode()

        request = GenerationRequest(
            request_id="req_1",
            query="query",
            docs=[],
            compressed_docs=encoded,
        )

        assert request.compressed_docs == original_data

    def test_generation_response_valid(self) -> None:
        """Test valid GenerationResponse."""
        response = GenerationResponse(
            request_id="req_1",
            generated_response="This is the generated text.",
            sentiment="neutral",
            is_toxic="false",
        )

        assert response.request_id == "req_1"
        assert response.generated_response == "This is the generated text."
        assert response.sentiment == "neutral"

    def test_generation_response_optional_fields(self) -> None:
        """Test GenerationResponse with optional fields as None."""
        response = GenerationResponse(
            request_id="req_1",
            generated_response="Response text.",
        )

        assert response.sentiment is None
        assert response.is_toxic is None


class TestPendingRequest:
    """Tests for PendingRequest schema."""

    def test_pending_request_minimal(self) -> None:
        """Test PendingRequest with minimal fields."""
        request = PendingRequest(
            request_id="req_1",
            query="test query",
            timestamp=12345.67,
        )

        assert request.request_id == "req_1"
        assert request.query == "test query"
        assert request.timestamp == 12345.67
        assert request.embedding is None
        assert request.docs is None

    def test_pending_request_full(self) -> None:
        """Test PendingRequest with all fields."""
        request = PendingRequest(
            request_id="req_1",
            query="test query",
            timestamp=12345.67,
            embedding=[0.1, 0.2],
            docs=[{"doc_id": 1}],
        )

        assert request.embedding == [0.1, 0.2]
        assert request.docs == [{"doc_id": 1}]


class TestComponentSchemas:
    """Tests for Document and RerankedDocument schemas."""

    def test_document_creation(self) -> None:
        """Test Document creation."""
        doc = Document(
            doc_id=123,
            title="Test Document",
            content="This is the content.",
        )

        assert doc.doc_id == 123
        assert doc.title == "Test Document"
        assert doc.content == "This is the content."
        assert doc.category == ""  # Default

    def test_document_with_category(self) -> None:
        """Test Document with category."""
        doc = Document(
            doc_id=123,
            title="Test",
            content="Content",
            category="Science",
        )

        assert doc.category == "Science"

    def test_reranked_document_creation(self) -> None:
        """Test RerankedDocument creation."""
        doc = RerankedDocument(
            doc_id=123,
            title="Test",
            content="Content",
            score=0.95,
        )

        assert doc.doc_id == 123
        assert doc.score == 0.95


class TestProfileSchemas:
    """Tests for profile configuration schemas."""

    def test_component_config_minimal(self) -> None:
        """Test ComponentConfig with minimal fields."""
        config = ComponentConfig(name="my_comp", type="embedding")

        assert config.name == "my_comp"
        assert config.type == "embedding"
        assert config.config == {}
        assert config.aliases == []

    def test_component_config_with_extras(self) -> None:
        """Test ComponentConfig with config and aliases."""
        config = ComponentConfig(
            name="my_comp",
            type="embedding",
            config={"model": "test-model"},
            aliases=["emb", "encoder"],
        )

        assert config.config == {"model": "test-model"}
        assert config.aliases == ["emb", "encoder"]

    def test_route_config_minimal(self) -> None:
        """Test RouteConfig with minimal fields."""
        route = RouteConfig(target="gateway")

        assert route.target == "gateway"
        assert route.prefix == "/"
        assert route.component_aliases == {}

    def test_route_config_full(self) -> None:
        """Test RouteConfig with all fields."""
        route = RouteConfig(
            target="retrieval",
            prefix="/retrieve",
            component_aliases={"emb": "my_embedding"},
        )

        assert route.target == "retrieval"
        assert route.prefix == "/retrieve"
        assert route.component_aliases == {"emb": "my_embedding"}

    def test_route_config_invalid_target(self) -> None:
        """Test RouteConfig with invalid target."""
        with pytest.raises(ValidationError):
            RouteConfig(target="invalid_target")

    def test_profile_file_minimal(self) -> None:
        """Test ProfileFile with minimal fields."""
        profile = ProfileFile(name="test_profile")

        assert profile.name == "test_profile"
        assert profile.description == ""
        assert profile.batch_size is None  # Defaults to None, uses settings
        assert profile.batch_timeout is None  # Defaults to None, uses settings
        assert profile.components == []
        assert profile.routes == []

    def test_profile_file_full(self) -> None:
        """Test ProfileFile with all fields."""
        profile = ProfileFile(
            name="full_profile",
            description="A complete profile",
            batch_size=64,
            batch_timeout=0.2,
            components=[
                ComponentConfig(name="comp1", type="embedding"),
                ComponentConfig(name="comp2", type="faiss"),
            ],
            routes=[
                RouteConfig(target="retrieval", prefix="/retrieve"),
            ],
        )

        assert profile.name == "full_profile"
        assert profile.description == "A complete profile"
        assert profile.batch_size == 64
        assert len(profile.components) == 2
        assert len(profile.routes) == 1

    def test_profile_file_duplicate_prefixes_fail(self) -> None:
        """Test that duplicate route prefixes fail validation."""
        with pytest.raises(ValidationError, match="Duplicate prefixes"):
            ProfileFile(
                name="test",
                routes=[
                    RouteConfig(target="retrieval", prefix="/api"),
                    RouteConfig(target="generation", prefix="/api"),
                ],
            )

    def test_profile_file_invalid_alias_target(self) -> None:
        """Test that alias pointing to unknown component fails."""
        with pytest.raises(ValidationError, match="unknown component"):
            ProfileFile(
                name="test",
                components=[ComponentConfig(name="comp1", type="embedding")],
                routes=[
                    RouteConfig(
                        target="retrieval",
                        component_aliases={"alias": "nonexistent_component"},
                    ),
                ],
            )

    def test_profile_file_valid_alias_target(self) -> None:
        """Test that alias pointing to valid component succeeds."""
        profile = ProfileFile(
            name="test",
            components=[
                ComponentConfig(name="my_embedding", type="embedding"),
            ],
            routes=[
                RouteConfig(
                    target="retrieval",
                    component_aliases={"emb": "my_embedding"},
                ),
            ],
        )

        assert profile.routes[0].component_aliases["emb"] == "my_embedding"
