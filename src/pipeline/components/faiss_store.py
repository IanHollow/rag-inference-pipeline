"""
FAISS index management for retrieval service.

Handles lazy loading and persistent storage of FAISS index for ANN search.
Supports both CPU (faiss-cpu) and GPU (faiss-gpu) backends.
"""

import gc
import logging
from pathlib import Path

import faiss
import numpy as np
import torch

from pipeline.config import PipelineSettings


logger = logging.getLogger(__name__)


def _is_faiss_gpu_available() -> bool:
    """
    Check if FAISS GPU support is available.

    Returns True only if:
    1. faiss-gpu package is installed (provides GPU functions)
    2. CUDA is available via PyTorch
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return False

    # Check if faiss has GPU support by looking for GPU-specific functions
    # faiss-cpu doesn't have StandardGpuResources, faiss-gpu does
    return hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")


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
            msg = f"FAISS index not found at {self.index_path}"
            raise FileNotFoundError(msg)

        logger.info("Loading FAISS index from %s", self.index_path)
        try:
            io_flags = 0
            if getattr(self.settings, "faiss_use_mmap", False):
                io_flags = faiss.IO_FLAG_MMAP
                logger.info("FAISS mmap mode enabled; index will be memory-mapped")

            if io_flags:
                self._index = faiss.read_index(str(self.index_path), io_flags)
            else:
                # Default path eagerly loads the entire index into RAM
                self._index = faiss.read_index(str(self.index_path))

            # Move to GPU if available (FAISS GPU only supports NVIDIA GPUs, not MPS)
            # Requires faiss-gpu package: pip install faiss-gpu (Linux only)
            if not self.settings.only_cpu and _is_faiss_gpu_available():
                try:
                    logger.info("Moving FAISS index to GPU (faiss-gpu detected)...")
                    res = faiss.StandardGpuResources()
                    self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                    logger.info("FAISS index moved to GPU successfully")
                except Exception as e:
                    logger.warning("Failed to move FAISS index to GPU, falling back to CPU: %s", e)
            elif not self.settings.only_cpu and torch.cuda.is_available():
                logger.info(
                    "CUDA available but faiss-gpu not installed. "
                    "Install with: pip uninstall faiss-cpu && pip install faiss-gpu"
                )

            self._is_loaded = True

            # Log index info
            index_size = self._index.ntotal if self._index is not None else 0
            logger.info(
                "FAISS index loaded successfully: %d vectors, dimension=%d",
                index_size,
                self.settings.faiss_dim,
            )

            # Set nprobe for IVF indices to balance speed vs accuracy
            # Higher nprobe = more accurate but slower
            # nprobe should match or be close to what was used during index creation
            if hasattr(self._index, "nprobe"):
                # With nlist=4096, nprobe=64 searches ~1.5% of clusters
                # This matches the nprobe value used when creating the index
                self._index.nprobe = 64
                logger.info("Set FAISS nprobe to %d for IVF index", self._index.nprobe)

            # Warmup
            if index_size > 0 and self._index is not None:
                logger.info("Warming up FAISS index...")
                dummy_vector = np.zeros((1, self.settings.faiss_dim), dtype=np.float32)
                self._index.search(dummy_vector, 1)
                logger.info("FAISS index warmup complete")
        except Exception as e:
            logger.exception("Failed to load FAISS index")
            msg = f"FAISS index loading failed: {e}"
            raise RuntimeError(msg) from e

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
            msg = "FAISS index not loaded. Call load() first."
            raise RuntimeError(msg)

        # Validate embedding shape
        if embeddings.ndim != 2:
            msg = f"Embeddings must be 2D array, got shape {embeddings.shape}"
            raise ValueError(msg)

        if embeddings.shape[1] != self.settings.faiss_dim:
            msg = (
                f"Embedding dimension mismatch: expected {self.settings.faiss_dim}, "
                f"got {embeddings.shape[1]}"
            )
            raise ValueError(msg)

        # Ensure correct dtype for FAISS
        embeddings = embeddings.astype("float32")

        logger.debug("Searching FAISS index with %d queries, k=%d", embeddings.shape[0], k)

        try:
            distances, indices = self._index.search(embeddings, k)
            logger.debug("FAISS search completed: found %s results", indices.shape)
        except Exception:
            logger.exception("FAISS search failed")
            raise
        else:
            return distances, indices

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
        return int(self._index.ntotal)

    def __repr__(self) -> str:
        """String representation of the store."""
        status = "loaded" if self._is_loaded else "not loaded"
        size = self.index_size if self._is_loaded else "unknown"
        return f"FAISSStore(path={self.index_path}, status={status}, size={size})"
