"""
FAISS index management for retrieval service.

Handles lazy loading and persistent storage of FAISS index for ANN search.
"""

import logging
from pathlib import Path

import faiss
import numpy as np

from ..config import PipelineSettings

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    Manages FAISS index for approximate nearest neighbor search.

    The index is lazily loaded on first use and kept in memory for subsequent searches.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the FAISS store.

        Args:
            settings: Pipeline configuration containing index path
        """
        self.settings = settings
        self.index_path = Path(settings.faiss_index_path)
        self._index: faiss.Index | None = None
        self._is_loaded = False

    def load(self) -> None:
        """
        Load the FAISS index from disk into memory.

        This pins the index in memory for fast repeated searches.

        Raises:
            FileNotFoundError: If index file doesn't exist
            RuntimeError: If index loading fails
        """
        if self._is_loaded:
            logger.info("FAISS index already loaded")
            return

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")

        logger.info("Loading FAISS index from %s", self.index_path)
        try:
            self._index = faiss.read_index(str(self.index_path))
            self._is_loaded = True

            # Log index info
            index_size = self._index.ntotal if self._index is not None else 0
            logger.info(
                "FAISS index loaded successfully: %d vectors, dimension=%d",
                index_size,
                self.settings.faiss_dim,
            )
        except Exception as e:
            logger.exception("Failed to load FAISS index: %s", e)
            raise RuntimeError(f"FAISS index loading failed: {e}") from e

    def search(self, embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform ANN search on the FAISS index.

        Args:
            embeddings: Query embeddings of shape (batch_size, dim)
            k: Number of nearest neighbors to retrieve per query

        Returns:
            Tuple of (distances, indices) where:
                - distances: array of shape (batch_size, k) with L2 distances
                - indices: array of shape (batch_size, k) with document IDs

        Raises:
            RuntimeError: If index is not loaded
            ValueError: If embeddings have wrong shape
        """
        if not self._is_loaded or self._index is None:
            raise RuntimeError("FAISS index not loaded. Call load() first.")

        # Validate embedding shape
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")

        if embeddings.shape[1] != self.settings.faiss_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.settings.faiss_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Ensure correct dtype for FAISS
        embeddings = embeddings.astype("float32")

        logger.debug("Searching FAISS index with %d queries, k=%d", embeddings.shape[0], k)

        try:
            distances, indices = self._index.search(embeddings, k)
            logger.debug("FAISS search completed: found %s results", indices.shape)
            return distances, indices
        except Exception as e:
            logger.exception("FAISS search failed: %s", e)
            raise

    def unload(self) -> None:
        """
        Unload the index from memory.

        This is typically called during service shutdown.
        """
        if self._is_loaded:
            logger.info("Unloading FAISS index")
            self._index = None
            self._is_loaded = False
            # Force garbage collection
            import gc

            gc.collect()

    @property
    def is_loaded(self) -> bool:
        """Check if the index is currently loaded."""
        return self._is_loaded

    @property
    def index_size(self) -> int:
        """Get the number of vectors in the index."""
        if not self._is_loaded or self._index is None:
            return 0
        return self._index.ntotal

    def __repr__(self) -> str:
        """String representation of the store."""
        status = "loaded" if self._is_loaded else "not loaded"
        size = self.index_size if self._is_loaded else "unknown"
        return f"FAISSStore(path={self.index_path}, status={status}, size={size})"
