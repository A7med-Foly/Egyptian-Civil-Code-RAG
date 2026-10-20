"""End-to-end RAG pipeline orchestration for Egyptian Civil Code Q&A."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from legal_rag.config import settings
from legal_rag.indexing.vector_store import LegalVectorStore
from legal_rag.ingestion.normalizer import clean_legal_text
from legal_rag.rag.generator import LegalGenerator
from legal_rag.rag.prompts import (
    ARABIC_LEGAL_SYSTEM_PROMPT,
    ARABIC_USER_PROMPT_TEMPLATE,
    format_context_from_chunks,
)

logger = logging.getLogger("legal_rag.rag.pipeline")


class LegalRAGPipeline:
    """Coordinates retrieval, context construction, prompt formatting, and LLM generation."""

    def __init__(
        self,
        vector_store: LegalVectorStore | None = None,
        generator: LegalGenerator | None = None,
    ):
        self.vector_store = vector_store or LegalVectorStore()
        self.generator = generator or LegalGenerator()
        logger.info("LegalRAGPipeline initialized.")

    async def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Process question through RAG pipeline and return answer with verified citations."""
        start_time = time.time()
        k = top_k or settings.RETRIEVAL_TOP_K
        threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD

        clean_query = clean_legal_text(question, is_arabic=True)
        logger.info(f"Processing query: '{clean_query}' (top_k={k})")

        # 1. Hybrid retrieval
        t_retrieval_start = time.time()
        retrieved_results = self.vector_store.hybrid_search(clean_query, top_k=k)
        retrieval_latency = time.time() - t_retrieval_start

        # Filter by threshold if applicable
        relevant_chunks = [
            (chunk, score) for chunk, score in retrieved_results if score >= threshold
        ]

        # Extract unique citations
        sources: list[str] = []
        seen = set()
        for chunk, score in relevant_chunks:
            cit = chunk.citation
            if cit not in seen:
                seen.add(cit)
                sources.append(cit)

        # 2. Context construction
        if relevant_chunks:
            context_str = format_context_from_chunks(relevant_chunks)
            user_prompt = ARABIC_USER_PROMPT_TEMPLATE.format(
                context=context_str, question=clean_query
            )
        else:
            logger.warning(
                f"No relevant documents found above threshold {threshold} for query: {clean_query}"
            )
            context_str = ""
            user_prompt = (
                f"سؤال المستخدم: {clean_query}\n\n"
                "لا توجد مواد قانونية في السياق مطابقة لهذا السؤال بدقة كافية."
            )

        # 3. Generation
        t_gen_start = time.time()
        answer = await self.generator.generate_response(
            prompt=user_prompt, system_prompt=ARABIC_LEGAL_SYSTEM_PROMPT
        )
        generation_latency = time.time() - t_gen_start

        total_latency = time.time() - start_time
        logger.info(
            f"Query answered in {total_latency * 1000:.1f}ms "
            f"(retrieval={retrieval_latency * 1000:.1f}ms, generation={generation_latency * 1000:.1f}ms) "
            f"with {len(sources)} citations."
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_documents": [chunk for chunk, _ in relevant_chunks],
            "retrieval_latency_ms": round(retrieval_latency * 1000, 2),
            "generation_latency_ms": round(generation_latency * 1000, 2),
            "total_latency_ms": round(total_latency * 1000, 2),
        }

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Synchronous convenience method to answer a question."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run, self.answer_question(question, top_k, similarity_threshold)
                    ).result()
            return loop.run_until_complete(
                self.answer_question(question, top_k, similarity_threshold)
            )
        except RuntimeError:
            return asyncio.run(self.answer_question(question, top_k, similarity_threshold))

    async def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated response tokens."""
        k = top_k or settings.RETRIEVAL_TOP_K
        clean_query = clean_legal_text(question, is_arabic=True)
        retrieved_results = self.vector_store.hybrid_search(clean_query, top_k=k)

        context_str = format_context_from_chunks(retrieved_results)
        user_prompt = ARABIC_USER_PROMPT_TEMPLATE.format(context=context_str, question=clean_query)

        async for token in self.generator.generate_stream(
            prompt=user_prompt, system_prompt=ARABIC_LEGAL_SYSTEM_PROMPT
        ):
            yield token
