"""BentoML Production Service for Arabic Legal Document Q&A.

Implements adaptive micro-batching with separate worker processes,
decoupling ML inference from ASGI event loop, with health and Prometheus metrics endpoints.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import bentoml
from pydantic import BaseModel, Field

from legal_rag.indexing.vector_store import LegalVectorStore
from legal_rag.rag.pipeline import LegalRAGPipeline

logger = logging.getLogger("bentoml.legal_rag")


class AskInput(BaseModel):
    question: str = Field(..., min_length=3, description="Arabic or English legal question")
    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of statutory articles to retrieve"
    )


class AskOutput(BaseModel):
    answer: str
    sources: list[str]
    model_version: str
    latency_ms: float


@bentoml.service(
    name="legal_rag_service",
    resources={"cpu": "2"},
    traffic={"timeout": 60},
)
class LegalRAGService:
    """Production BentoML Service for Egyptian Legal RAG with adaptive micro-batching."""

    def __init__(self):
        logger.info("Initializing LegalRAGService models and vector store...")
        self.vector_store = LegalVectorStore()
        self.pipeline = LegalRAGPipeline(vector_store=self.vector_store)
        self.api_version = "v0.3.0"
        logger.info(
            f"LegalRAGService ready with {self.vector_store.documents_indexed} indexed articles."
        )

    @bentoml.api(route="/ask")
    async def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Interactive legal question answering."""
        start_time = time.perf_counter()
        result = await self.pipeline.answer_question(question=question, top_k=top_k)
        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "model_version": self.api_version,
            "latency_ms": total_latency_ms,
        }

    @bentoml.api(
        route="/ask_batch",
        batchable=True,
        batch_dim=0,
        max_batch_size=16,
        max_latency_ms=50,
    )
    def ask_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        """Adaptive micro-batching endpoint: aggregates concurrent requests into a single batch."""
        results = []
        for q in questions:
            # Dense + sparse hybrid retrieval
            retrieved = self.vector_store.search(q, top_k=3)
            sources = [chunk.citation for chunk, _ in retrieved]
            article_numbers = [chunk.article_number for chunk, _ in retrieved]
            results.append(
                {
                    "question": q,
                    "sources": sources,
                    "article_numbers": article_numbers,
                    "status": "retrieved",
                    "model_version": self.api_version,
                }
            )
        return results

    @bentoml.api(route="/health")
    def health(self) -> dict[str, Any]:
        """Health and readiness probe."""
        return {
            "status": "healthy",
            "service": "legal_rag_bento",
            "documents_indexed": self.vector_store.documents_indexed,
            "version": self.api_version,
        }
