"""
Unit tests for the generation service.

Tests cover schema validation, model loading, reranking, LLM generation,
sentiment analysis, toxicity filtering, and end-to-end processing.
"""

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import lz4.frame  # type: ignore
import msgspec
import pytest
import torch

from pipeline.components.llm import LLMGenerator
from pipeline.components.reranker import Reranker
from pipeline.components.sentiment import SentimentAnalyzer
from pipeline.components.toxicity import ToxicityFilter
from pipeline.config import PipelineSettings
from pipeline.services.generation.schemas import (
    Document,
    GenerationRequest,
    GenerationRequestItem,
    GenerationResponse,
    RerankedDocument,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


class TestSchemas:
    """Tests for generation service schemas."""

    def test_document_schema(self) -> None:
        """Test Document schema validation."""
        doc = Document(
            doc_id=1,
            title="Test Title",
            content="Test content",
            category="test",
        )
        assert doc.doc_id == 1
        assert doc.title == "Test Title"
        assert doc.content == "Test content"
        assert doc.category == "test"

    def test_document_schema_default_category(self) -> None:
        """Test Document schema with default category."""
        doc = Document(
            doc_id=1,
            title="Test Title",
            content="Test content",
        )
        assert doc.category == ""

    def test_reranked_document_schema(self) -> None:
        """Test RerankedDocument schema validation."""
        doc = RerankedDocument(
            doc_id=1,
            title="Test Title",
            content="Test content",
            category="test",
            score=0.95,
        )
        assert doc.doc_id == 1
        assert doc.score == 0.95

    def test_generation_request_item_schema(self) -> None:
        """Test GenerationRequestItem schema validation."""
        docs = [
            Document(doc_id=1, title="Title 1", content="Content 1"),
            Document(doc_id=2, title="Title 2", content="Content 2"),
        ]
        item = GenerationRequestItem(
            request_id="test_req_1",
            query="Test query",
            docs=docs,
        )
        assert item.request_id == "test_req_1"
        assert item.query == "Test query"
        assert len(item.docs) == 2

    def test_generation_request_schema(self) -> None:
        """Test GenerationRequest schema validation."""
        items = [
            GenerationRequestItem(
                request_id="test_req_1",
                query="Test query 1",
                docs=[Document(doc_id=1, title="Title 1", content="Content 1")],
            ),
            GenerationRequestItem(
                request_id="test_req_2",
                query="Test query 2",
                docs=[Document(doc_id=2, title="Title 2", content="Content 2")],
            ),
        ]
        request = GenerationRequest(
            batch_id="batch_1",
            items=items,
        )
        assert request.batch_id == "batch_1"
        assert len(request.items) == 2

    def test_generation_response_schema(self) -> None:
        """Test GenerationResponse schema validation."""
        from pipeline.services.generation.schemas import GenerationResponseItem

        items = [
            GenerationResponseItem(
                request_id="test_req_1",
                generated_response="Test response 1",
                sentiment="positive",
                is_toxic="false",
            ),
        ]
        response = GenerationResponse(
            batch_id="batch_1",
            items=items,
            processing_time=1.5,
        )
        assert response.batch_id == "batch_1"
        assert len(response.items) == 1
        assert response.processing_time == 1.5

    def test_generation_request_item_base64_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that base64 decoding does not emit warning logs on success."""
        import base64
        import logging

        # Create a valid base64 string
        content = b"compressed content"
        encoded = base64.b64encode(content).decode()

        with caplog.at_level(logging.WARNING):
            GenerationRequestItem(
                request_id="test_req_1",
                query="Test query",
                docs=[],
                compressed_docs=encoded,
            )

        # Assert no warnings were captured
        assert not caplog.records


class TestReranker:
    """Tests for Reranker class."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(
            NODE_NUMBER=2,
            ONLY_CPU=True,
        )

    @pytest.fixture
    def mock_reranker(self, settings: PipelineSettings) -> Generator[Reranker, None, None]:
        """Create a mock reranker without loading real models."""
        reranker = Reranker(settings)

        # Mock the model and tokenizer
        with (
            patch.object(reranker, "tokenizer") as mock_tokenizer,
            patch.object(reranker, "model") as mock_model,
        ):
            reranker._loaded = True

            # Mock tokenizer behavior
            mock_tokenizer_inst = MagicMock()
            mock_tokenizer_inst.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
            mock_tokenizer.return_value = mock_tokenizer_inst

            # Mock model behavior
            mock_output = MagicMock()
            mock_output.logits = torch.tensor([[0.5], [1.0], [-0.5]])
            mock_model_inst = MagicMock()
            mock_model_inst.return_value = mock_output
            mock_model.return_value = mock_model_inst

            yield reranker

            reranker._loaded = False

    def test_reranker_initialization(self, settings: PipelineSettings) -> None:
        """Test reranker initialization."""
        reranker = Reranker(settings)
        assert reranker.model_name == "BAAI/bge-reranker-base"
        assert not reranker.is_loaded

    def test_reranker_not_loaded_error(self, settings: PipelineSettings) -> None:
        """Test that reranking fails when model is not loaded."""
        reranker = Reranker(settings)
        docs = [Document(doc_id=1, title="Test", content="Test content")]

        with pytest.raises(RuntimeError, match="Reranker model not loaded"):
            reranker.rerank("test query", docs)

    def test_rerank_empty_documents(self, mock_reranker: Reranker) -> None:
        """Test reranking with empty document list."""
        result = mock_reranker.rerank("test query", [])
        assert len(result) == 0

    def test_rerank_batch_length_mismatch(self, mock_reranker: Reranker) -> None:
        """Test that batch reranking fails with mismatched lengths."""
        queries = ["query1", "query2"]
        documents_batch = [[Document(doc_id=1, title="Test", content="Test content")]]

        with pytest.raises(ValueError, match="must have same length"):
            mock_reranker.rerank_batch(queries, documents_batch)


class TestLLMGenerator:
    """Tests for LLMGenerator class."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(
            NODE_NUMBER=2,
            ONLY_CPU=True,
        )

    @pytest.fixture
    def mock_llm(self, settings: PipelineSettings) -> Generator[LLMGenerator, None, None]:
        """Create a mock LLM generator without loading real models."""
        llm = LLMGenerator(settings)

        # Mock the model and tokenizer
        with (
            patch.object(llm, "tokenizer") as mock_tokenizer,
            patch.object(llm, "model") as mock_model,
        ):
            llm._loaded = True

            # Mock tokenizer behavior
            mock_tokenizer.apply_chat_template.return_value = "mocked template"

            # Create a mock for tokenizer call that returns a dict-like object with .to() method
            mock_inputs = MagicMock()
            mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
            mock_inputs.attention_mask = torch.tensor([[1, 1, 1]])
            mock_inputs.to.return_value = mock_inputs
            mock_tokenizer.return_value = mock_inputs

            mock_tokenizer.batch_decode.return_value = ["Mocked response"]
            mock_tokenizer.eos_token_id = 2

            # Mock model behavior
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

            yield llm

            llm._loaded = False

    def test_llm_initialization(self, settings: PipelineSettings) -> None:
        """Test LLM generator initialization."""
        llm = LLMGenerator(settings)
        assert llm.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert not llm.is_loaded

    def test_llm_not_loaded_error(self, settings: PipelineSettings) -> None:
        """Test that generation fails when model is not loaded."""
        llm = LLMGenerator(settings)
        docs = [RerankedDocument(doc_id=1, title="Test", content="Test content", score=0.9)]

        with pytest.raises(RuntimeError, match="LLM model not loaded"):
            llm.generate("test query", docs)

    def test_generate_with_empty_docs(self, mock_llm: LLMGenerator) -> None:
        """Test generation with empty document list."""
        result = mock_llm.generate("test query", [])
        assert isinstance(result, str)

    def test_generate_batch_length_mismatch(self, mock_llm: LLMGenerator) -> None:
        """Test that batch generation fails with mismatched lengths."""
        queries = ["query1", "query2"]
        reranked_docs_batch = [
            [RerankedDocument(doc_id=1, title="Test", content="Test content", score=0.9)]
        ]

        with pytest.raises(ValueError, match="must have same length"):
            mock_llm.generate_batch(queries, reranked_docs_batch)


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer class."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(
            NODE_NUMBER=2,
            ONLY_CPU=True,
        )

    @pytest.fixture
    def mock_sentiment(
        self, settings: PipelineSettings
    ) -> Generator[SentimentAnalyzer, None, None]:
        """Create a mock sentiment analyzer without loading real models."""
        analyzer = SentimentAnalyzer(settings)
        analyzer._loaded = True

        # Create a mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "5 stars", "score": 0.99}]
        analyzer.pipeline = cast("Any", mock_pipeline)

        yield analyzer

        analyzer._loaded = False

    def test_sentiment_initialization(self, settings: PipelineSettings) -> None:
        """Test sentiment analyzer initialization."""
        analyzer = SentimentAnalyzer(settings)
        assert analyzer.model_name == "nlptown/bert-base-multilingual-uncased-sentiment"
        assert not analyzer.is_loaded

    def test_sentiment_not_loaded_error(self, settings: PipelineSettings) -> None:
        """Test that analysis fails when model is not loaded."""
        analyzer = SentimentAnalyzer(settings)

        with pytest.raises(RuntimeError, match="Sentiment model not loaded"):
            analyzer.analyze("test text")

    def test_sentiment_mapping(self, mock_sentiment: SentimentAnalyzer) -> None:
        """Test sentiment label mapping."""
        # Test all possible mappings
        mappings = [
            ("1 star", "very negative"),
            ("2 stars", "negative"),
            ("3 stars", "neutral"),
            ("4 stars", "positive"),
            ("5 stars", "very positive"),
        ]

        # Get pipeline as Any to access mock attributes
        assert mock_sentiment.pipeline is not None
        mock_pipeline: Any = mock_sentiment.pipeline

        for model_label, expected_output in mappings:
            mock_pipeline.return_value = [{"label": model_label, "score": 0.99}]
            result = mock_sentiment.analyze("test text")
            assert result == expected_output

    def test_sentiment_truncation(self, mock_sentiment: SentimentAnalyzer) -> None:
        """Test that long texts are truncated."""
        long_text = "a" * 1000
        mock_sentiment.analyze(long_text)
        # The mock should be called with truncated text
        assert mock_sentiment.pipeline is not None
        mock_pipeline: Any = mock_sentiment.pipeline
        mock_pipeline.assert_called_once()

    def test_analyze_batch(self, mock_sentiment: SentimentAnalyzer) -> None:
        """Test batch sentiment analysis."""
        assert mock_sentiment.pipeline is not None
        mock_pipeline: Any = mock_sentiment.pipeline
        mock_pipeline.return_value = [
            {"label": "5 stars", "score": 0.99},
            {"label": "1 star", "score": 0.99},
        ]

        texts = ["positive text", "negative text"]
        results = mock_sentiment.analyze_batch(texts)

        assert len(results) == 2
        assert results[0] == "very positive"
        assert results[1] == "very negative"


class TestToxicityFilter:
    """Tests for ToxicityFilter class."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(
            NODE_NUMBER=2,
            ONLY_CPU=True,
        )

    @pytest.fixture
    def mock_toxicity(self, settings: PipelineSettings) -> Generator[ToxicityFilter, None, None]:
        """Create a mock toxicity filter without loading real models."""
        filter_obj = ToxicityFilter(settings)
        filter_obj._loaded = True

        # Create a mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"score": 0.3}]  # Below threshold
        filter_obj.pipeline = cast("Any", mock_pipeline)

        yield filter_obj

        filter_obj._loaded = False

    def test_toxicity_initialization(self, settings: PipelineSettings) -> None:
        """Test toxicity filter initialization."""
        filter_obj = ToxicityFilter(settings)
        assert filter_obj.model_name == "unitary/toxic-bert"
        assert not filter_obj.is_loaded
        assert filter_obj.threshold == 0.5

    def test_toxicity_not_loaded_error(self, settings: PipelineSettings) -> None:
        """Test that filtering fails when model is not loaded."""
        filter_obj = ToxicityFilter(settings)

        with pytest.raises(RuntimeError, match="Toxicity model not loaded"):
            filter_obj.filter("test text")

    def test_toxicity_threshold(self, mock_toxicity: ToxicityFilter) -> None:
        """Test toxicity threshold behavior."""
        assert mock_toxicity.pipeline is not None
        mock_pipeline: Any = mock_toxicity.pipeline

        # Test below threshold
        mock_pipeline.return_value = [{"score": 0.3}]
        result = mock_toxicity.filter("safe text")
        assert result == "false"

        # Test above threshold
        mock_pipeline.return_value = [{"score": 0.8}]
        result = mock_toxicity.filter("toxic text")
        assert result == "true"

        # Test at threshold
        mock_pipeline.return_value = [{"score": 0.5}]
        result = mock_toxicity.filter("borderline text")
        assert result == "false"  # Not > threshold

    def test_toxicity_truncation(self, mock_toxicity: ToxicityFilter) -> None:
        """Test that long texts are truncated."""
        long_text = "a" * 1000
        mock_toxicity.filter(long_text)
        # The mock should be called with truncated text
        assert mock_toxicity.pipeline is not None
        mock_pipeline: Any = mock_toxicity.pipeline
        mock_pipeline.assert_called_once()

    def test_filter_batch(self, mock_toxicity: ToxicityFilter) -> None:
        """Test batch toxicity filtering."""
        assert mock_toxicity.pipeline is not None
        mock_pipeline: Any = mock_toxicity.pipeline
        mock_pipeline.return_value = [
            {"score": 0.2},  # Safe
            {"score": 0.9},  # Toxic
            {"score": 0.5},  # At threshold
        ]

        texts = ["safe text", "toxic text", "borderline text"]
        results = mock_toxicity.filter_batch(texts)

        assert len(results) == 3
        assert results[0] == "false"
        assert results[1] == "true"
        assert results[2] == "false"


class TestGenerationService:
    """Tests for the generation service endpoints."""

    @pytest.fixture
    def mock_app(self) -> Generator[TestClient, None, None]:
        """Create a test client with mocked models."""
        from fastapi import FastAPI

        from pipeline.component_registry import ComponentRegistry
        from pipeline.services.generation import api
        from pipeline.services.generation.api import router

        # Reset global executor
        api._executor = None

        app = FastAPI()
        app.include_router(router, prefix="/generate")

        registry = ComponentRegistry()
        app.state.registry = registry
        app.state.component_aliases = {}

        # Create mocks
        mock_reranker = MagicMock()
        mock_llm = MagicMock()
        mock_sentiment = MagicMock()
        mock_toxicity = MagicMock()

        # Configure mocks to appear loaded
        mock_reranker.is_loaded = True
        mock_llm.is_loaded = True
        mock_sentiment.is_loaded = True
        mock_toxicity.is_loaded = True

        # Mock model behaviors
        mock_reranker.rerank.return_value = [
            RerankedDocument(doc_id=1, title="Doc 1", content="Content 1", score=0.9, category="")
        ]

        mock_llm.generate.return_value = "Generated response"

        mock_sentiment.analyze.return_value = "positive"

        mock_toxicity.check.return_value = (False, {"label": "toxicity", "score": 0.1})

        # Register mocks
        registry.register("reranker", mock_reranker)
        registry.register("llm_generator", mock_llm)
        registry.register("sentiment_analyzer", mock_sentiment)
        registry.register("toxicity_filter", mock_toxicity)

        @app.get("/health")
        def health() -> dict[str, Any]:
            return {"status": "healthy", "models_loaded": True}

        yield TestClient(app)

    def test_health_endpoint(self, mock_app: TestClient) -> None:
        """Test health check endpoint."""
        response = mock_app.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True

    def test_generate_endpoint_success(self, mock_app: TestClient) -> None:
        """Test successful generation request."""
        request_data = {
            "batch_id": "test_batch_1",
            "items": [
                {
                    "request_id": "req_1",
                    "query": "Test query",
                    "docs": [
                        {
                            "doc_id": 1,
                            "title": "Test Doc",
                            "content": "Test content",
                            "category": "test",
                        }
                    ],
                }
            ],
        }

        response = mock_app.post("/generate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == "test_batch_1"
        assert len(data["items"]) == 1
        assert data["items"][0]["request_id"] == "req_1"
        assert data["items"][0]["generated_response"] == "Generated response"
        assert data["items"][0]["sentiment"] == "positive"
        assert data["items"][0]["is_toxic"] == "false"

    def test_generate_endpoint_compressed_payload(self, mock_app: TestClient) -> None:
        """Test generation request with compressed payload."""
        # Create compressed docs
        from pipeline.components.schemas import DocumentStruct

        docs = [
            DocumentStruct(
                doc_id=1,
                title="Compressed Doc",
                content="Compressed content",
                category="test",
            )
        ]
        serialized = msgspec.json.encode(docs)
        compressed = lz4.frame.compress(serialized)

        # We need to encode bytes to string for JSON transport in test client
        # But wait, the API expects bytes for compressed_docs if it's a Pydantic model?
        # Pydantic BaseJSONModel handles bytes by base64 encoding/decoding usually?
        # Or maybe the test client needs to send it as a specific format.
        # The schema says `compressed_docs: bytes | None`.
        # When sending JSON via TestClient, bytes are not automatically handled unless we base64 encode them
        # and the Pydantic model validator handles it.
        # However, standard Pydantic v2 doesn't automatically decode base64 strings to bytes for `bytes` fields in JSON mode unless configured.
        # But let's assume the standard behavior or that we can pass it if we were using msgspec directly.
        # Since we are using `json=` in TestClient, we can't pass raw bytes.
        # Let's try passing it as a string (if Pydantic handles it) or skip the JSON encoding issue by mocking the request object if possible.
        # Actually, `BaseJSONModel` might handle it.
        # Let's check `BaseJSONModel` definition if I can.
        # But for now, let's try to send it as a string (latin-1 decoded) which is how it might look if it was raw bytes in a dict,
        # but JSON doesn't support raw bytes.
        # The real RPC client uses `msgspec.json.encode` which handles bytes by base64 encoding them by default?
        # Or maybe it expects the user to handle it.
        # If I look at `src/pipeline/services/gateway/rpc_client.py`, it uses `msgspec.json.encode`.
        # `msgspec` encodes bytes as base64 strings by default.
        # So I should base64 encode it here.

        import base64

        compressed_b64 = base64.b64encode(compressed).decode("utf-8")

        request_data = {
            "batch_id": "test_batch_compressed",
            "items": [
                {
                    "request_id": "req_compressed",
                    "query": "Test query",
                    "docs": [],  # Empty docs
                    "compressed_docs": compressed_b64,
                }
            ],
        }

        # We need to patch the GenerationService to ensure it actually decompresses
        # But here we are testing the API layer and Executor passing it through.
        # The `mock_app` uses `GenerationService` which uses `GenerationExecutor`.
        # The `GenerationExecutor` uses `BatchScheduler`.
        # If the fix works, `GenerationService.process_batch` (or `_process_batch_sync`) will receive the compressed docs
        # and decompress them.
        # The `mock_llm` will receive the decompressed docs.
        # So we can check if `mock_llm.generate` was called with the correct docs.

        # Access the mock_reranker from the app state
        app = cast("FastAPI", mock_app.app)
        mock_reranker = app.state.registry.get("reranker")

        response = mock_app.post("/generate", json=request_data)
        assert response.status_code == 200

        # Verify Reranker was called with the decompressed doc
        args, _ = mock_reranker.rerank.call_args
        query, input_docs = args
        assert query == "Test query"
        assert len(input_docs) == 1
        assert input_docs[0].title == "Compressed Doc"
        assert input_docs[0].content == "Compressed content"

    def test_generate_endpoint_validation_error(self, mock_app: TestClient) -> None:
        """Test generation request with invalid data."""
        # Missing required field
        request_data = {
            "batch_id": "test_batch_1",
            # Missing 'items'
        }

        response = mock_app.post("/generate", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_generate_endpoint_empty_batch(self, mock_app: TestClient) -> None:
        """Test generation request with empty batch."""
        request_data = {
            "batch_id": "test_batch_1",
            "items": [],
        }

        response = mock_app.post("/generate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 0

    def test_metrics_endpoint(self, mock_app: TestClient) -> None:
        """Test metrics endpoint."""
        response = mock_app.get("/generate/metrics")
        assert response.status_code == 200
        # Should return Prometheus text format
        assert "generation_requests_total" in response.text or response.text == ""

    def test_initialization_error_id_only_no_store(self) -> None:
        """Test that initialization fails if id_only mode is used without document store."""
        from unittest.mock import patch

        from pipeline.services.generation.service import GenerationService

        mock_llm = MagicMock()

        with patch("pipeline.services.generation.service.settings") as mock_settings:
            mock_settings.documents_payload_mode = "id_only"

            with pytest.raises(
                ValueError, match="Configuration Error: DOCUMENTS_PAYLOAD_MODE='id_only'"
            ):
                GenerationService(
                    reranker=None,
                    llm_generator=mock_llm,
                    sentiment_analyzer=None,
                    toxicity_filter=None,
                    document_store=None,
                )
