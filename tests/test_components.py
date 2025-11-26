"""
Tests for pipeline components.

Tests cover:
- FAISSStore: initialization, load, search, validation
- EmbeddingGenerator: initialization, encode, cache behavior
- Document/DocumentStore: initialization, fetch
- Reranker: initialization, rerank
- LLMGenerator: initialization
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.components.document_store import Document
from pipeline.components.faiss_store import FAISSStore
from pipeline.config import PipelineSettings


class TestFAISSStore:
    """Tests for FAISSStore class."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(
            NODE_NUMBER=1,
            FAISS_INDEX_PATH="test_data/test_faiss.bin",
        )

    def test_init(self, settings: PipelineSettings) -> None:
        """Test FAISSStore initialization."""
        store = FAISSStore(settings)

        assert store.settings == settings
        assert store.index_path == Path("test_data/test_faiss.bin")
        assert store._index is None
        assert not store._is_loaded

    def test_load_file_not_found(self, settings: PipelineSettings) -> None:
        """Test that loading non-existent file raises error."""
        store = FAISSStore(settings)

        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            store.load()

    @patch("pipeline.components.faiss_store.faiss.read_index")
    @patch.object(Path, "exists", return_value=True)
    def test_load_success(
        self, mock_exists: MagicMock, mock_read_index: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test successful FAISS index loading."""
        mock_index = MagicMock()
        mock_index.ntotal = 1000
        mock_index.search.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        mock_read_index.return_value = mock_index

        store = FAISSStore(settings)
        store.load()

        assert store._is_loaded
        assert store._index == mock_index
        mock_read_index.assert_called_once()

    @patch("pipeline.components.faiss_store.faiss.read_index")
    @patch.object(Path, "exists", return_value=True)
    def test_load_already_loaded(
        self, mock_exists: MagicMock, mock_read_index: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test that loading when already loaded is a no-op."""
        mock_index = MagicMock()
        mock_index.ntotal = 1000
        mock_index.search.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        mock_read_index.return_value = mock_index

        store = FAISSStore(settings)
        store.load()
        store.load()  # Second call should be no-op

        # Should only be called once
        assert mock_read_index.call_count == 1

    def test_search_not_loaded_raises_error(self, settings: PipelineSettings) -> None:
        """Test that searching without loading raises error."""
        store = FAISSStore(settings)
        rng = np.random.default_rng(42)
        embeddings = rng.random((1, settings.faiss_dim)).astype(np.float32)

        with pytest.raises(RuntimeError, match="not loaded"):
            store.search(embeddings, k=10)

    @patch("pipeline.components.faiss_store.faiss.read_index")
    @patch.object(Path, "exists", return_value=True)
    def test_search_invalid_embedding_shape(
        self, mock_exists: MagicMock, mock_read_index: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test that search with wrong embedding shape raises error."""
        mock_index = MagicMock()
        mock_index.ntotal = 1000
        mock_index.search.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        mock_read_index.return_value = mock_index

        store = FAISSStore(settings)
        store.load()

        # 1D array should fail
        rng = np.random.default_rng(42)
        embeddings_1d = rng.random(settings.faiss_dim).astype(np.float32)
        with pytest.raises(ValueError, match="2D array"):
            store.search(embeddings_1d, k=10)

    @patch("pipeline.components.faiss_store.faiss.read_index")
    @patch.object(Path, "exists", return_value=True)
    def test_search_wrong_dimension(
        self, mock_exists: MagicMock, mock_read_index: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test that search with wrong dimension raises error."""
        mock_index = MagicMock()
        mock_index.ntotal = 1000
        mock_index.search.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        mock_read_index.return_value = mock_index

        store = FAISSStore(settings)
        store.load()

        # Wrong dimension
        rng = np.random.default_rng(42)
        wrong_dim = settings.faiss_dim + 10
        embeddings = rng.random((1, wrong_dim)).astype(np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.search(embeddings, k=10)

    @patch("pipeline.components.faiss_store.faiss.read_index")
    @patch.object(Path, "exists", return_value=True)
    def test_search_success(
        self, mock_exists: MagicMock, mock_read_index: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test successful search."""
        mock_index = MagicMock()
        mock_index.ntotal = 1000
        expected_distances = np.array([[0.1, 0.2, 0.3]])
        expected_indices = np.array([[1, 2, 3]])
        mock_index.search.return_value = (expected_distances, expected_indices)
        mock_read_index.return_value = mock_index

        store = FAISSStore(settings)
        store.load()

        rng = np.random.default_rng(42)
        embeddings = rng.random((1, settings.faiss_dim)).astype(np.float32)
        distances, indices = store.search(embeddings, k=3)

        np.testing.assert_array_equal(distances, expected_distances)
        np.testing.assert_array_equal(indices, expected_indices)

    @patch("pipeline.components.faiss_store.faiss.read_index")
    @patch.object(Path, "exists", return_value=True)
    def test_unload(
        self, mock_exists: MagicMock, mock_read_index: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test unloading FAISS index."""
        mock_index = MagicMock()
        mock_index.ntotal = 1000
        mock_index.search.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))
        mock_read_index.return_value = mock_index

        store = FAISSStore(settings)
        store.load()
        assert store._is_loaded

        store.unload()
        assert not store._is_loaded
        assert store._index is None

    def test_is_loaded_property(self, settings: PipelineSettings) -> None:
        """Test is_loaded property."""
        store = FAISSStore(settings)
        assert not store.is_loaded

        store._is_loaded = True
        assert store.is_loaded


class TestDocument:
    """Tests for Document class."""

    def test_document_creation(self) -> None:
        """Test creating a Document."""
        doc = Document(
            doc_id=123,
            title="Test Title",
            content="Test content here.",
            category="Science",
        )

        assert doc.doc_id == 123
        assert doc.title == "Test Title"
        assert doc.content == "Test content here."
        assert doc.category == "Science"

    def test_document_default_category(self) -> None:
        """Test Document with default category."""
        doc = Document(
            doc_id=1,
            title="Title",
            content="Content",
        )

        assert doc.category is None

    def test_document_to_dict(self) -> None:
        """Test Document to_dict conversion."""
        doc = Document(
            doc_id=123,
            title="Title",
            content="Content",
            category="Category",
        )

        result = doc.to_dict()

        assert result["doc_id"] == 123
        assert result["title"] == "Title"
        assert result["content"] == "Content"
        assert result["category"] == "Category"

    def test_document_to_dict_without_category(self) -> None:
        """Test Document to_dict without category."""
        doc = Document(
            doc_id=1,
            title="Title",
            content="Content",
        )

        result = doc.to_dict()

        assert "category" not in result

    def test_document_truncate(self) -> None:
        """Test Document truncation."""
        doc = Document(
            doc_id=1,
            title="A" * 100,
            content="B" * 100,
            category="Science",
        )

        truncated = doc.truncate(max_length=50)

        assert len(truncated.title) == 50
        assert len(truncated.content) == 50
        assert truncated.doc_id == 1
        assert truncated.category == "Science"

    def test_document_truncate_shorter_content(self) -> None:
        """Test Document truncation with already short content."""
        doc = Document(
            doc_id=1,
            title="Short",
            content="Also short",
        )

        truncated = doc.truncate(max_length=50)

        assert truncated.title == "Short"
        assert truncated.content == "Also short"


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(node_number=1, only_cpu=True)

    def test_init(self, settings: PipelineSettings) -> None:
        """Test EmbeddingGenerator initialization."""
        from pipeline.components.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(settings)

        assert generator.settings == settings
        assert generator.model_name == settings.embedding_model_name
        assert not generator._is_loaded
        assert generator._model is None

    def test_is_loaded_property(self, settings: PipelineSettings) -> None:
        """Test is_loaded property."""
        from pipeline.components.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(settings)
        assert not generator.is_loaded

        generator._is_loaded = True
        assert generator.is_loaded

    def test_encode_not_loaded_raises_error(self, settings: PipelineSettings) -> None:
        """Test that encoding without loading raises error."""
        from pipeline.components.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(settings)

        with pytest.raises(RuntimeError, match="not loaded"):
            generator.encode(["test"])

    def test_encode_empty_list_raises_error(self, settings: PipelineSettings) -> None:
        """Test that encoding empty list raises error."""
        from pipeline.components.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(settings)
        generator._is_loaded = True
        generator._model = MagicMock()

        with pytest.raises(ValueError, match="empty"):
            generator.encode([])

    def test_repr(self, settings: PipelineSettings) -> None:
        """Test string representation."""
        from pipeline.components.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(settings)

        repr_str = repr(generator)
        assert "EmbeddingGenerator" in repr_str
        assert "not loaded" in repr_str

        generator._is_loaded = True
        repr_str = repr(generator)
        assert "loaded" in repr_str

    def test_clear_cache(self, settings: PipelineSettings) -> None:
        """Test clearing the cache."""
        from pipeline.components.embedding import EmbeddingGenerator

        generator = EmbeddingGenerator(settings)

        # Add something to cache
        generator.cache.put("test_key", np.array([0.1, 0.2]))

        generator.clear_cache()

        assert generator.cache.get("test_key") is None


class TestReranker:
    """Tests for Reranker class."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(node_number=2, only_cpu=True)

    def test_init(self, settings: PipelineSettings) -> None:
        """Test Reranker initialization."""
        from pipeline.components.reranker import Reranker

        reranker = Reranker(settings)

        assert reranker.settings == settings
        assert reranker.model_name == settings.reranker_model_name
        assert not reranker._loaded
        assert reranker.model is None
        assert reranker.tokenizer is None

    def test_is_loaded_property(self, settings: PipelineSettings) -> None:
        """Test is_loaded property."""
        from pipeline.components.reranker import Reranker

        reranker = Reranker(settings)
        assert not reranker.is_loaded

        reranker._loaded = True
        assert reranker.is_loaded

    def test_rerank_not_loaded_raises_error(self, settings: PipelineSettings) -> None:
        """Test that reranking without loading raises error."""
        from pipeline.components.reranker import Reranker
        from pipeline.components.schemas import Document

        reranker = Reranker(settings)
        docs = [Document(doc_id=1, title="Test", content="Content")]

        with pytest.raises(RuntimeError, match="not loaded"):
            reranker.rerank("query", docs)

    def test_rerank_empty_documents(self, settings: PipelineSettings) -> None:
        """Test reranking with empty documents."""
        from pipeline.components.reranker import Reranker

        reranker = Reranker(settings)
        reranker._loaded = True
        reranker.model = MagicMock()
        reranker.tokenizer = MagicMock()

        result = reranker.rerank("query", [])

        assert result == []


class TestLLMGenerator:
    """Tests for LLMGenerator class."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(node_number=2, only_cpu=True)

    def test_init(self, settings: PipelineSettings) -> None:
        """Test LLMGenerator initialization."""
        from pipeline.components.llm import LLMGenerator

        generator = LLMGenerator(settings)

        assert generator.settings == settings
        assert generator.model_name == settings.llm_model_name
        assert not generator._loaded
        assert generator.model is None
        assert generator.tokenizer is None

    def test_is_loaded_property(self, settings: PipelineSettings) -> None:
        """Test is_loaded property."""
        from pipeline.components.llm import LLMGenerator

        generator = LLMGenerator(settings)
        assert not generator.is_loaded

        generator._loaded = True
        assert generator.is_loaded
