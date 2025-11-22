"""
Generation Service Layer.

Encapsulates the business logic for the generation pipeline:
Reranking -> LLM Generation -> Sentiment Analysis -> Toxicity Filtering.
"""

import json
import logging
import time

from opentelemetry import trace

from ...components.llm import LLMGenerator
from ...components.reranker import Reranker
from ...components.sentiment import SentimentAnalyzer
from ...components.toxicity import ToxicityFilter
from ...config import get_settings
from ...telemetry import (
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
    GenerationRequest,
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
        sentiment_analyzer: SentimentAnalyzer,
        toxicity_filter: ToxicityFilter,
    ) -> None:
        self.reranker = reranker
        self.llm_generator = llm_generator
        self.sentiment_analyzer = sentiment_analyzer
        self.toxicity_filter = toxicity_filter

    def process_batch(self, generation_request: GenerationRequest) -> GenerationResponse:
        """
        Process a batch of generation requests.
        """
        profiler = SampledStageProfiler(
            enabled=settings.enable_profiling,
            sample_rate=settings.profiling_sample_rate,
            logger=logger,
        )

        start_time = time.time()
        batch_size = len(generation_request.items)
        generation_requests_total.inc()
        generation_batch_size.observe(batch_size)
        pipeline_batch_size_histogram.labels(
            run_id=settings.profiling_run_id, node=NODE_LABEL, service="generation"
        ).observe(batch_size)

        logger.info(
            "Processing generation batch: %s with %d items", generation_request.batch_id, batch_size
        )

        with tracer.start_as_current_span(
            "generation.batch",
            attributes={
                "pipeline.batch_id": generation_request.batch_id,
                "pipeline.service": "generation",
                "pipeline.node": settings.node_number,
            },
        ):
            response_items: list[GenerationResponseItem] = []

            for item in generation_request.items:
                # Step 1: Rerank documents
                rerank_start = time.time()
                with (
                    tracer.start_as_current_span("generation.rerank"),
                    profiler.track("generation.rerank"),
                ):
                    if self.reranker:
                        # Convert RetrievalDocument to dict or whatever reranker expects
                        reranked_docs = self.reranker.rerank(item.query, item.docs)
                    else:
                        # Convert Document to RerankedDocument with default score
                        reranked_docs = [
                            RerankedDocument(
                                doc_id=doc.doc_id,
                                title=doc.title,
                                content=doc.content,
                                category=doc.category,
                                score=1.0,
                            )
                            for doc in item.docs
                        ]

                rerank_elapsed = time.time() - rerank_start
                rerank_duration.observe(rerank_elapsed)
                stage_duration_gauge.labels(
                    run_id=settings.profiling_run_id, node=NODE_LABEL, stage="generation.rerank"
                ).set(rerank_elapsed)

                # Step 2: Generate text with LLM
                llm_start = time.time()
                with (
                    tracer.start_as_current_span("generation.llm"),
                    profiler.track("generation.llm"),
                ):
                    # Use top k documents for context
                    context_docs = reranked_docs[:3]  # Top 3
                    generated_text = self.llm_generator.generate(item.query, context_docs)

                llm_elapsed = time.time() - llm_start
                llm_generation_duration.observe(llm_elapsed)
                stage_duration_gauge.labels(
                    run_id=settings.profiling_run_id, node=NODE_LABEL, stage="generation.llm"
                ).set(llm_elapsed)

                # Step 3: Sentiment Analysis
                sentiment_start = time.time()
                with (
                    tracer.start_as_current_span("generation.sentiment"),
                    profiler.track("generation.sentiment"),
                ):
                    sentiment_score = self.sentiment_analyzer.analyze(generated_text)

                sentiment_elapsed = time.time() - sentiment_start
                sentiment_duration.observe(sentiment_elapsed)
                stage_duration_gauge.labels(
                    run_id=settings.profiling_run_id, node=NODE_LABEL, stage="generation.sentiment"
                ).set(sentiment_elapsed)

                # Step 4: Toxicity Filter
                toxicity_start = time.time()
                with (
                    tracer.start_as_current_span("generation.toxicity"),
                    profiler.track("generation.toxicity"),
                ):
                    is_toxic, _ = self.toxicity_filter.check(generated_text)

                toxicity_elapsed = time.time() - toxicity_start
                toxicity_duration.observe(toxicity_elapsed)
                stage_duration_gauge.labels(
                    run_id=settings.profiling_run_id, node=NODE_LABEL, stage="generation.toxicity"
                ).set(toxicity_elapsed)

                # Filter if toxic
                final_text = generated_text
                if is_toxic:
                    final_text = "[Content Filtered due to toxicity]"

                response_items.append(
                    GenerationResponseItem(
                        request_id=item.request_id,
                        generated_response=final_text,
                        sentiment=sentiment_score,
                        is_toxic="true" if is_toxic else "false",
                    )
                )

        # Record total duration
        total_duration = time.time() - start_time
        generation_request_duration.observe(total_duration)
        pipeline_latency_histogram.labels(
            run_id=settings.profiling_run_id, node=NODE_LABEL, service="generation"
        ).observe(total_duration)
        pipeline_request_counter.labels(
            run_id=settings.profiling_run_id,
            node=NODE_LABEL,
            service="generation",
            status="success",
        ).inc()

        if summary := profiler.summary():
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
