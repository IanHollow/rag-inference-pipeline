import time
from typing import cast
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline.components.document_store import Document, DocumentStore
from pipeline.components.embedding import EmbeddingGenerator
from pipeline.config import PipelineSettings
from pipeline.utils.cache import CompressedLRUCache, LRUCache


class TestLRUCache(unittest.TestCase):
    def test_lru_eviction(self) -> None:
        cache = LRUCache[str, int](capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_ttl(self) -> None:
        cache = LRUCache[str, int](capacity=10, ttl=0.1)
        cache.put("a", 1)
        assert cache.get("a") == 1
        time.sleep(0.2)
        assert cache.get("a") is None


class TestLRUCacheExtended:
    """Extended tests for LRUCache."""

    def test_lru_order_preserved(self) -> None:
        """Test that LRU order is maintained correctly."""
        cache = LRUCache[str, int](capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Access "a" to make it recently used
        cache.get("a")

        # Add new item - should evict "b" (least recently used)
        cache.put("d", 4)

        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_update_existing_key(self) -> None:
        """Test updating an existing key."""
        cache = LRUCache[str, int](capacity=2)
        cache.put("a", 1)
        cache.put("a", 2)

        assert cache.get("a") == 2
        assert len(cache.cache) == 1

    def test_clear_resets_counters(self) -> None:
        """Test that clear resets all counters."""
        cache = LRUCache[str, int](capacity=10, name="test_cache")
        cache.put("a", 1)
        cache.get("a")
        cache.get("nonexistent")

        assert cache.hits == 1
        assert cache.misses == 1

        cache.clear()

        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0
        assert len(cache.cache) == 0

    def test_ttl_expired_item_counted_as_miss(self) -> None:
        """Test that expired TTL item is counted as miss."""
        cache = LRUCache[str, int](capacity=10, ttl=0.05)
        cache.put("a", 1)

        # First access - hit
        cache.get("a")
        assert cache.hits == 1
        assert cache.misses == 0

        # Wait for expiry
        time.sleep(0.1)

        # Second access - should be miss due to TTL
        result = cache.get("a")
        assert result is None
        assert cache.misses == 1

    def test_eviction_counter_increments(self) -> None:
        """Test that eviction counter increments correctly."""
        cache = LRUCache[str, int](capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.evictions == 0

        cache.put("c", 3)  # Should evict "a"
        assert cache.evictions == 1

        cache.put("d", 4)  # Should evict "b"
        assert cache.evictions == 2

    def test_capacity_one(self) -> None:
        """Test cache with capacity of 1."""
        cache = LRUCache[str, int](capacity=1)
        cache.put("a", 1)
        assert cache.get("a") == 1

        cache.put("b", 2)
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_get_nonexistent_key(self) -> None:
        """Test getting a key that was never added."""
        cache = LRUCache[str, int](capacity=10)
        assert cache.get("nonexistent") is None
        assert cache.misses == 1

    def test_custom_name(self) -> None:
        """Test that custom name is set correctly."""
        cache = LRUCache[str, int](capacity=10, name="custom_cache")
        assert cache.name == "custom_cache"


class TestCompressedLRUCache:
    """Tests for CompressedLRUCache."""

    def test_basic_put_get(self) -> None:
        """Test basic put and get operations."""
        cache = CompressedLRUCache[str, dict](capacity=10, name="test_compressed")
        data = {"key": "value", "number": 42}

        cache.put("test", data)
        result = cache.get("test")

        assert result == data

    def test_complex_data_structures(self) -> None:
        """Test with complex nested data structures."""
        cache = CompressedLRUCache[str, dict](capacity=10)
        data = {
            "nested": {
                "array": [1, 2, 3],
                "string": "hello",
                "null": None,
            },
            "list_of_dicts": [{"a": 1}, {"b": 2}],
        }

        cache.put("complex", data)
        result = cache.get("complex")

        assert result == data

    def test_list_data(self) -> None:
        """Test with list data."""
        cache = CompressedLRUCache[str, list](capacity=10)
        data = [1, 2, 3, "four", {"five": 5}]

        cache.put("list", data)
        result = cache.get("list")

        assert result == data

    def test_get_nonexistent_returns_none(self) -> None:
        """Test that getting nonexistent key returns None."""
        cache = CompressedLRUCache[str, dict](capacity=10)
        assert cache.get("nonexistent") is None

    def test_lru_eviction_with_compression(self) -> None:
        """Test LRU eviction works with compressed values."""
        cache = CompressedLRUCache[str, dict](capacity=2)
        cache.put("a", {"data": "first"})
        cache.put("b", {"data": "second"})
        cache.put("c", {"data": "third"})

        assert cache.get("a") is None
        assert cache.get("b") == {"data": "second"}
        assert cache.get("c") == {"data": "third"}

    def test_update_compressed_value(self) -> None:
        """Test updating an existing compressed value."""
        cache = CompressedLRUCache[str, dict](capacity=10)
        cache.put("key", {"version": 1})
        cache.put("key", {"version": 2})

        result = cache.get("key")
        assert result == {"version": 2}

    def test_large_data_compression(self) -> None:
        """Test that large data is compressed effectively."""
        cache = CompressedLRUCache[str, dict](capacity=10)
        # Create large repetitive data that compresses well
        large_data = {"content": "x" * 10000, "items": list(range(1000))}

        cache.put("large", large_data)
        result = cache.get("large")

        assert result == large_data

    def test_ttl_with_compression(self) -> None:
        """Test TTL works with compressed cache."""
        cache = CompressedLRUCache[str, dict](capacity=10, ttl=0.05)
        cache.put("temp", {"temporary": True})

        assert cache.get("temp") == {"temporary": True}

        time.sleep(0.1)

        assert cache.get("temp") is None


class TestEmbeddingCache(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = PipelineSettings()
        self.settings.disable_cache_for_profiling = False
        self.settings.only_cpu = True
        self.generator = EmbeddingGenerator(self.settings)
        # Mock model
        self.generator._model = MagicMock()
        self.generator._model.encode.return_value = np.array([[0.1, 0.2]])
        self.generator._is_loaded = True

    def test_embedding_cache(self) -> None:
        text = "test query"

        # First call - miss
        self.generator.encode([text])
        cast("MagicMock", self.generator._model).encode.assert_called_once()

        # Second call - hit
        cast("MagicMock", self.generator._model).encode.reset_mock()
        self.generator.encode([text])
        cast("MagicMock", self.generator._model).encode.assert_not_called()


class TestDocumentStoreCache(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = PipelineSettings()
        self.settings.disable_cache_for_profiling = False
        # Mock db path
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024)),
            patch("diskcache.Cache"),  # Mock diskcache if it was still used, but we replaced it
            patch("sqlite3.connect"),  # We need to mock sqlite3 connection to avoid DB errors
        ):
            self.store = DocumentStore(self.settings)

    def test_document_cache(self) -> None:
        doc_id = 123
        doc = Document(doc_id=doc_id, title="Test", content="Content")

        # Manually populate cache
        self.store.cache.put(doc_id, doc.to_dict())

        # Fetch - should hit cache and not use DB
        with patch.object(self.store, "_get_connection") as mock_conn:
            docs = self.store.fetch_documents([doc_id])
            mock_conn.assert_not_called()
            assert len(docs) == 1
            assert docs[0].title == "Test"


if __name__ == "__main__":
    unittest.main()
