import time
from typing import cast
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline.components.document_store import Document, DocumentStore
from pipeline.components.embedding import EmbeddingGenerator
from pipeline.config import PipelineSettings
from pipeline.utils.cache import LRUCache


class TestLRUCache(unittest.TestCase):
    def test_lru_eviction(self) -> None:
        cache = LRUCache[str, int](capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)

    def test_ttl(self) -> None:
        cache = LRUCache[str, int](capacity=10, ttl=0.1)
        cache.put("a", 1)
        self.assertEqual(cache.get("a"), 1)
        time.sleep(0.2)
        self.assertIsNone(cache.get("a"))


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
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].title, "Test")


if __name__ == "__main__":
    unittest.main()
