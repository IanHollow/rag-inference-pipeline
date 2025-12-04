"""
Generation Service Layer.

Encapsulates the business logic for the generation pipeline:
Reranking -> LLM Generation -> Sentiment Analysis -> Toxicity Filtering.
"""

import json
import logging
import time

import lz4.frame  # type: ignore[import-untyped]
import msgspec
from opentelemetry import trace

from pipeline.components.document_store import DocumentStore
from pipeline.components.llm import LLMGenerator
from pipeline.components.reranker import Reranker
from pipeline.components.sentiment import SentimentAnalyzer
from pipeline.components.toxicity import ToxicityFilter
from pipeline.config import get_settings
from pipeline.telemetry import (
    SampledStageProfiler,
    batch_size_histogram as pipeline_batch_size_histogram,
    latency_histogram as pipeline_latency_histogram,
    request_counter as pipeline_request_counter,
    stage_duration_gauge,
)

from .metrics import (
    generation_batch_size,
    generation_request_duration,
    generation_requests_total,
    llm_generation_duration,
    rerank_duration,
    sentiment_duration,
    toxicity_duration,
)
from .schemas import (
    Document,
    DocumentStruct,
    GenerationRequest,
    GenerationRequestItem,
    GenerationResponse,
    GenerationResponseItem,
    RerankedDocument,
)


logger = logging.getLogger(__name__)
settings = get_settings()
NODE_LABEL = str(settings.node_number)
tracer = trace.get_tracer(__name__)


class GenerationService:
    def __init__(
        self,
        reranker: Reranker | None,
        llm_generator: LLMGenerator,
        sentiment_analyzer: SentimentAnalyzer | None,
        toxicity_filter: ToxicityFilter | None,
        document_store: DocumentStore | None = None,
    ) -> None:
        self.reranker = reranker
        self.llm_generator = llm_generator
        self.sentiment_analyzer = sentiment_analyzer
        self.toxicity_filter = toxicity_filter
        self.document_store = document_store

        # Cache settings for hot path
        self._enable_profiling = settings.enable_profiling
        self._profiling_sample_rate = settings.profiling_sample_rate
        self._profiling_run_id = settings.profiling_run_id

        # Validate configuration: ID-only handoff requires a document store
        payload_mode = settings.documents_payload_mode
        if payload_mode == "id_only" and self.document_store is None:
            msg = (
                "Configuration Error: DOCUMENTS_PAYLOAD_MODE='id_only' but no DocumentStore "
                "is configured on this node. Documents will reach the LLM empty. "
                "Please add a document_store to the generation profile or use 'full' payload mode."
            )
            logger.error(msg)
            raise ValueError(msg)

    def process_batch(self, generation_request: GenerationRequest) -> GenerationResponse:
        """
        Process a batch of generation requests.
        Optimized for low latency by processing sequentially - GPU/MPS models
        don't benefit from thread parallelism due to GIL and single-device execution.
        """
        start_time = time.time()
        batch_size = len(generation_request.items)

        # Fast path metrics (avoid repeated dict lookups)
        generation_requests_total.inc()
        generation_batch_size.observe(batch_size)

        run_id = self._profiling_run_id
        pipeline_batch_size_histogram.labels(
            run_id=run_id, node=NODE_LABEL, service="generation"
        ).observe(batch_size)

        # Only create profiler if profiling is enabled
        profiler: SampledStageProfiler | None = None
        if self._enable_profiling:
            profiler = SampledStageProfiler(
                enabled=True,
                sample_rate=self._profiling_sample_rate,
                logger=logger,
            )

        logger.info(
            "Processing generation batch: %s with %d items", generation_request.batch_id, batch_size
        )

        # Cache component references for hot loop
        reranker = self.reranker
        llm_generator = self.llm_generator
        sentiment_analyzer = self.sentiment_analyzer
        toxicity_filter = self.toxicity_filter
        document_store = self.document_store

        response_items: list[GenerationResponseItem] = []

        with tracer.start_as_current_span(
            "generation.batch",
            attributes={
                "pipeline.batch_id": generation_request.batch_id,
                "pipeline.service": "generation",
                "pipeline.node": settings.node_number,
            },
        ):
            for item in generation_request.items:
                response_item = self._process_single_item(
                    item=item,
                    reranker=reranker,
                    llm_generator=llm_generator,
                    sentiment_analyzer=sentiment_analyzer,
                    toxicity_filter=toxicity_filter,
                    document_store=document_store,
                    profiler=profiler,
                    run_id=run_id,
                )
                response_items.append(response_item)

        # Record total duration
        total_duration = time.time() - start_time
        generation_request_duration.observe(total_duration)
        pipeline_latency_histogram.labels(
            run_id=run_id, node=NODE_LABEL, service="generation"
        ).observe(total_duration)
        pipeline_request_counter.labels(
            run_id=run_id,
            node=NODE_LABEL,
            service="generation",
            status="success",
        ).inc()

        if profiler and (summary := profiler.summary()):
            logger.info(
                json.dumps(
                    {
                        "event": "generation_profile",
                        "batch_id": generation_request.batch_id,
                        "summary": summary,
                    }
                )
            )

        logger.info(
            "Generation batch %s completed in %.3fs (avg %.3fms per item)",
            generation_request.batch_id,
            total_duration,
            (total_duration / batch_size) * 1000 if batch_size > 0 else 0,
        )

        return GenerationResponse(
            batch_id=generation_request.batch_id,
            items=response_items,
            processing_time=total_duration,
        )

    def _process_single_item(
        self,
        item: GenerationRequestItem,
        reranker: Reranker | None,
        llm_generator: LLMGenerator,
        sentiment_analyzer: SentimentAnalyzer | None,
        toxicity_filter: ToxicityFilter | None,
        document_store: DocumentStore | None,
        profiler: SampledStageProfiler | None,
        run_id: str,
    ) -> GenerationResponseItem:
        """Process a single generation request item. Optimized hot path."""

        # Step -1: Decompress documents if needed
        if item.compressed_docs:
            try:
                decompressed_data = lz4.frame.decompress(item.compressed_docs)
                docs_structs = msgspec.json.decode(decompressed_data, type=list[DocumentStruct])
                item.docs = [
                    Document(
                        doc_id=d.doc_id,
                        title=d.title,
                        content=d.content,
                        category=d.category,
                    )
                    for d in docs_structs
                ]
            except Exception:
                logger.exception(
                    "Failed to decompress documents for request %s",
                    item.request_id,
                )

        # Step 0: Fetch documents if needed (Doc-ID handoff)
        docs = item.docs
        if document_store and docs and not docs[0].content:
            doc_ids = [d.doc_id for d in docs]
            fetched_docs = document_store.fetch_documents(doc_ids)
            docs = [
                Document(
                    doc_id=d.doc_id,
                    title=d.title,
                    content=d.content,
                    category=d.category or "",
                )
                for d in fetched_docs
            ]

        # Step 1: Rerank documents
        rerank_start = time.time()

        if reranker:
            with tracer.start_as_current_span("generation.rerank"):
                if profiler:
                    with profiler.track("generation.rerank"):
                        reranked_docs = reranker.rerank(item.query, docs)
                else:
                    reranked_docs = reranker.rerank(item.query, docs)
        else:
            # Fast path: no reranking needed
            reranked_docs = [
                RerankedDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    content=doc.content,
                    category=doc.category,
                    score=1.0,
                )
                for doc in docs
            ]

        rerank_elapsed = time.time() - rerank_start
        rerank_duration.observe(rerank_elapsed)
        stage_duration_gauge.labels(run_id=run_id, node=NODE_LABEL, stage="generation.rerank").set(
            rerank_elapsed
        )

        # Step 2: Generate text with LLM (use top 3 docs)
        context_docs = reranked_docs[:3]
        llm_start = time.time()

        with tracer.start_as_current_span("generation.llm"):
            if profiler:
                with profiler.track("generation.llm"):
                    generated_text = llm_generator.generate(item.query, context_docs)
            else:
                generated_text = llm_generator.generate(item.query, context_docs)

        llm_elapsed = time.time() - llm_start
        llm_generation_duration.observe(llm_elapsed)
        stage_duration_gauge.labels(run_id=run_id, node=NODE_LABEL, stage="generation.llm").set(
            llm_elapsed
        )

        # Step 3: Sentiment Analysis
        sentiment_score: str | None = None
        if sentiment_analyzer:
            sentiment_start = time.time()

            with tracer.start_as_current_span("generation.sentiment"):
                if profiler:
                    with profiler.track("generation.sentiment"):
                        sentiment_score = sentiment_analyzer.analyze(generated_text)
                else:
                    sentiment_score = sentiment_analyzer.analyze(generated_text)

            sentiment_elapsed = time.time() - sentiment_start
            sentiment_duration.observe(sentiment_elapsed)
            stage_duration_gauge.labels(
                run_id=run_id, node=NODE_LABEL, stage="generation.sentiment"
            ).set(sentiment_elapsed)

        # Step 4: Toxicity Filter
        is_toxic = False
        if toxicity_filter:
            toxicity_start = time.time()

            with tracer.start_as_current_span("generation.toxicity"):
                if profiler:
                    with profiler.track("generation.toxicity"):
                        is_toxic, _ = toxicity_filter.check(generated_text)
                else:
                    is_toxic, _ = toxicity_filter.check(generated_text)

            toxicity_elapsed = time.time() - toxicity_start
            toxicity_duration.observe(toxicity_elapsed)
            stage_duration_gauge.labels(
                run_id=run_id, node=NODE_LABEL, stage="generation.toxicity"
            ).set(toxicity_elapsed)

        # Filter if toxic
        final_text = "[Content Filtered due to toxicity]" if is_toxic else generated_text

        return GenerationResponseItem(
            request_id=item.request_id,
            generated_response=final_text,
            sentiment=sentiment_score,
            is_toxic="true" if is_toxic else "false" if toxicity_filter else None,
        )
