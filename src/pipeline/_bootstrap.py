"""
Bootstrap module for early environment and library configuration.
"""

import importlib
import os
import types

import torch

from .config import get_settings


# CRITICAL: Set these environment variables BEFORE importing any libraries
# that use OpenMP (faiss, torch, etc.) to prevent crashes from duplicate
# OpenMP runtime initialization.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# CRITICAL: Configure torch threads BEFORE importing faiss
# On macOS, calling torch.set_num_threads() AFTER faiss is imported
# causes a segmentation fault due to OpenMP runtime conflicts.
_settings = get_settings()
torch.set_num_threads(_settings.cpu_inference_threads)
torch.set_num_interop_threads(max(2, _settings.cpu_inference_threads // 2))

# Now it's safe to import faiss using importlib to avoid E402
faiss: types.ModuleType = importlib.import_module("faiss")

# Re-export for convenience
__all__ = ["faiss", "torch"]
