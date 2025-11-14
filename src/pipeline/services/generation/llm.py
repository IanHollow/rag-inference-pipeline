"""
LLM module for the generation service.

Manages Qwen/Qwen2.5-0.5B-Instruct for response generation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from ...config import PipelineSettings
    from .schemas import RerankedDocument


logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    LLM response generator using Qwen/Qwen2.5-0.5B-Instruct.

    Loads the model once and keeps it resident in memory.
    Generates responses with max_tokens=128 constraint.
    Uses torch autocast and tokenizer caching to stay within 16GB RAM.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the LLM generator.

        Args:
            settings: Pipeline configuration settings
        """
        self.settings = settings
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None
        self._loaded = False

        logger.info("LLM Generator initialized (device: %s)", self.device)

    def load(self) -> None:
        """Load the LLM model and tokenizer into memory."""
        if self._loaded:
            logger.warning("LLM model already loaded")
            return

        logger.info("Loading LLM model: %s", self.model_name)
        start_time = time.time()

        try:
            # Load tokenizer with caching
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )

            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map=str(self.device),
            )
            self.model.eval()
            self._loaded = True

            elapsed = time.time() - start_time
            logger.info("LLM model loaded in %.2f seconds", elapsed)

        except Exception as e:
            logger.exception("Failed to load LLM model: %s", e)
            raise

    def unload(self) -> None:
        """Unload the model from memory."""
        if not self._loaded:
            return

        logger.info("Unloading LLM model")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("LLM model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded

    def _build_prompt(
        self, query: str, reranked_docs: list[RerankedDocument]
    ) -> list[dict[str, str]]:
        """
        Build the prompt from query and reranked documents.

        Args:
            query: The user query
            reranked_docs: List of reranked documents (top-N)

        Returns:
            Formatted messages list for chat template
        """
        # Use top 3 documents for context
        top_docs = reranked_docs[:3]
        context = "\n".join([f"- {doc.title}: {doc.content[:200]}" for doc in top_docs])

        messages = [
            {
                "role": "system",
                "content": "When given Context and Question, reply as 'Answer: <final answer>' only.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

        return messages

    def generate(
        self,
        query: str,
        reranked_docs: list[RerankedDocument],
    ) -> str:
        """
        Generate a response for a single query.

        Args:
            query: The user query
            reranked_docs: List of reranked documents

        Returns:
            Generated response string

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded or self.model is None or self.tokenizer is None:
            msg = "LLM model not loaded"
            raise RuntimeError(msg)

        # Explicitly assert types for mypy
        model: PreTrainedModel = self.model
        tokenizer: PreTrainedTokenizerBase = self.tokenizer

        # Build prompt
        messages = self._build_prompt(query, reranked_docs)

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        tokenizer_output = tokenizer([text], return_tensors="pt")
        model_inputs = tokenizer_output.to(self.device)

        # Get generate method explicitly to work around transformers type stub issues
        # The generate attribute is incorrectly typed in transformers stubs
        generate_fn = model.generate

        # Generate with autocast for memory efficiency
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    generated_ids = generate_fn(  # type: ignore[operator]
                        input_ids=model_inputs["input_ids"],
                        attention_mask=model_inputs.get("attention_mask"),
                        max_new_tokens=self.settings.max_tokens,
                        temperature=0.01,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                    )
            else:
                generated_ids = generate_fn(  # type: ignore[operator]
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs.get("attention_mask"),
                    max_new_tokens=self.settings.max_tokens,
                    temperature=0.01,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )

        # Extract only the generated tokens and remove input
        # generated_ids is a tensor of shape (batch_size, seq_len)
        input_length = model_inputs.input_ids.shape[1]
        generated_ids_list = [output_ids[input_length:] for output_ids in generated_ids]

        # Decode
        response = tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True)[0]

        return response

    def generate_batch(
        self,
        queries: list[str],
        reranked_docs_batch: list[list[RerankedDocument]],
    ) -> list[str]:
        """
        Generate responses for a batch of queries.

        Args:
            queries: List of query strings
            reranked_docs_batch: List of reranked document lists

        Returns:
            List of generated response strings

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If queries and reranked_docs_batch have different lengths
        """
        if not self._loaded or self.model is None or self.tokenizer is None:
            msg = "LLM model not loaded"
            raise RuntimeError(msg)

        if len(queries) != len(reranked_docs_batch):
            msg = f"Queries ({len(queries)}) and reranked_docs ({len(reranked_docs_batch)}) must have same length"
            raise ValueError(msg)

        responses = []
        for query, reranked_docs in zip(queries, reranked_docs_batch, strict=False):
            response = self.generate(query, reranked_docs)
            responses.append(response)

        return responses
