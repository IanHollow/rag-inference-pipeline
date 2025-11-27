"""
Reranker module for the generation service.

Loads and manages the BAAI/bge-reranker-base model for document reranking.
"""

import logging
import time
from typing import TYPE_CHECKING, cast

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

from ..config import PipelineSettings
from .schemas import Document, RerankedDocument

logger = logging.getLogger(__name__)


class Reranker:
    """
    Document reranker using BAAI/bge-reranker-base.

    Loads the model once and keeps it resident in memory.
    Scores query-document pairs and returns top-N documents with normalized scores.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the reranker.

        Args:
            settings: Pipeline configuration settings
        """
        self.settings = settings
        self.model_name = settings.reranker_model_name

        if settings.only_cpu:
            device_name = "cpu"
        elif torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"

        self.device = torch.device(device_name)
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None
        self._loaded = False

        logger.info("Reranker initialized (device: %s)", self.device)

    def load(self) -> None:
        """Load the reranker model and tokenizer into memory."""
        if self._loaded:
            logger.warning("Reranker already loaded")
            return

        logger.info("Loading reranker model: %s", self.model_name)
        start_time = time.time()

        try:
            # Load tokenizer
            self.tokenizer = cast(
                "PreTrainedTokenizerBase",
                AutoTokenizer.from_pretrained(self.model_name),
            )

            # Load model with appropriate dtype
            # Use float16 for CUDA and MPS (Apple Silicon), float32 for CPU
            use_fp16 = self.device.type in ("cuda", "mps")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                dtype=torch.float16 if use_fp16 else torch.float32,
            )
            model = model.to(self.device)
            model.eval()

            # Apply torch.compile for faster inference (PyTorch 2.0+)
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, mode="max-autotune", fullgraph=True)
                    logger.info("Applied torch.compile optimization")
                except Exception as e:
                    logger.warning("torch.compile failed, using eager mode: %s", e)

            self.model = cast("PreTrainedModel", model)
            self._loaded = True

            # Warmup
            logger.info("Warming up reranker...")
            if self.tokenizer:
                dummy_query = "query " * 10
                dummy_doc = "document " * 100  # Approx 100-200 tokens
                dummy_inputs = self.tokenizer(
                    [dummy_query], [dummy_doc], return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
                with torch.no_grad():
                    self.model(**dummy_inputs)
            logger.info("Reranker warmup complete")

            elapsed = time.time() - start_time
            logger.info("Reranker model loaded in %.2f seconds", elapsed)

        except Exception as e:
            logger.exception("Failed to load reranker model: %s", e)
            raise

    def unload(self) -> None:
        """Unload the model from memory."""
        if not self._loaded:
            return

        logger.info("Unloading reranker model")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        self._loaded = False
        logger.info("Reranker model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[RerankedDocument]:
        """
        Rerank documents for a single query.

        Args:
            query: The query string
            documents: List of documents to rerank
            top_n: Number of top documents to return (default: all documents)

        Returns:
            List of reranked documents with scores, sorted by score (descending)

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded or self.model is None or self.tokenizer is None:
            msg = "Reranker model not loaded"
            raise RuntimeError(msg)

        if not documents:
            return []

        if top_n is None:
            top_n = len(documents)

        # Prepare query-document pairs
        pairs = [[query, doc.content] for doc in documents]

        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.settings.truncate_length,
        ).to(self.device)

        with torch.inference_mode():
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()

        # Normalize scores to [0, 1] using sigmoid and convert to numpy
        # Using torch sigmoid before cpu transfer can be faster on GPU
        normalized_scores = torch.sigmoid(scores).cpu().numpy()

        # Create reranked documents with scores
        scored_docs = [
            RerankedDocument(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                category=doc.category,
                score=float(normalized_scores[idx]),
            )
            for idx, doc in enumerate(documents)
        ]

        # Sort by score descending and return top-N
        scored_docs.sort(key=lambda x: x.score, reverse=True)

        return scored_docs[:top_n]

    def rerank_batch(
        self,
        queries: list[str],
        documents_batch: list[list[Document]],
        top_n: int | None = None,
    ) -> list[list[RerankedDocument]]:
        """
        Rerank documents for a batch of queries.

        Args:
            queries: List of query strings
            documents_batch: List of document lists (one per query)
            top_n: Number of top documents to return per query

        Returns:
            List of reranked document lists with scores

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If queries and documents_batch have different lengths
        """
        if not self._loaded or self.model is None or self.tokenizer is None:
            msg = "Reranker model not loaded"
            raise RuntimeError(msg)

        if len(queries) != len(documents_batch):
            msg = f"Queries ({len(queries)}) and documents ({len(documents_batch)}) must have same length"
            raise ValueError(msg)

        results = []
        for query, documents in zip(queries, documents_batch, strict=False):
            reranked = self.rerank(query, documents, top_n)
            results.append(reranked)

        return results
