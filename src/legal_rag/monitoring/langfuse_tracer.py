"""Langfuse tracing integration for Arabic Legal Document Q&A RAG.

Provides end-to-end tracing across query ingestion, retrieval chunks,
prompt construction, and LLM answer generation with token accounting,
TTFT (Time To First Token) observation, and graceful local fallback.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("legal_rag.monitoring.langfuse")

# Try importing Langfuse client
try:
    from langfuse import Langfuse

    LANGFUSE_SDK_AVAILABLE = True
except ImportError:
    Langfuse = None  # type: ignore
    LANGFUSE_SDK_AVAILABLE = False


class LegalRAGTracer:
    """Manages LLM traces, spans, and metrics with Langfuse and fallback logger."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ) -> None:
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "http://localhost:3001")
        self.enabled = bool(LANGFUSE_SDK_AVAILABLE and self.public_key and self.secret_key)
        self.client: Any | None = None
        self.active_traces: list[dict[str, Any]] = []

        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
                logger.info("Langfuse tracer initialized successfully pointing to %s", self.host)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to initialize Langfuse client (%s). Using fallback tracer.", e
                )
                self.enabled = False
        else:
            logger.info("Langfuse keys not configured; operating in local fallback telemetry mode.")

    @contextmanager
    def trace_query(
        self,
        query: str,
        correlation_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager to trace an end-to-end Legal RAG query."""
        start_time = time.perf_counter()
        trace_data: dict[str, Any] = {
            "name": "arabic-legal-rag-query",
            "input": query,
            "correlation_id": correlation_id,
            "user_id": user_id or "anonymous",
            "metadata": metadata or {},
            "spans": [],
            "generations": [],
            "start_time": time.time(),
        }

        langfuse_trace = None
        if self.enabled and self.client:
            try:
                langfuse_trace = self.client.trace(
                    name=trace_data["name"],
                    input=trace_data["input"],
                    user_id=trace_data["user_id"],
                    metadata={"correlation_id": correlation_id, **(metadata or {})},
                )
                trace_data["_langfuse_trace"] = langfuse_trace
            except Exception as e:  # noqa: BLE001
                logger.debug("Langfuse trace creation failed: %s", e)

        try:
            yield trace_data
        finally:
            total_duration = time.perf_counter() - start_time
            trace_data["duration_seconds"] = total_duration
            self.active_traces.append(trace_data)

            if langfuse_trace:
                try:
                    langfuse_trace.update(
                        output=trace_data.get("output"),
                        metadata={"total_duration_sec": total_duration},
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("Langfuse trace update failed: %s", e)

    def record_retrieval_span(
        self,
        trace_data: dict[str, Any],
        query: str,
        retrieved_documents: list[Any],
        latency_ms: float,
    ) -> dict[str, Any]:
        """Record the dense/BM25 retrieval stage."""
        docs_summary = [
            {
                "article_number": getattr(doc, "article_number", None)
                or (doc.get("article_number") if isinstance(doc, dict) else None),
                "score": getattr(doc, "score", None)
                or (doc.get("score") if isinstance(doc, dict) else None),
                "preview": (
                    getattr(doc, "text", "")
                    if hasattr(doc, "text")
                    else (doc.get("text", "") if isinstance(doc, dict) else str(doc))
                )[:120],
            }
            for doc in retrieved_documents
        ]

        span_record = {
            "name": "retrieval",
            "query": query,
            "num_documents": len(retrieved_documents),
            "documents": docs_summary,
            "latency_ms": latency_ms,
        }
        trace_data["spans"].append(span_record)

        lf_trace = trace_data.get("_langfuse_trace")
        if lf_trace:
            try:
                lf_trace.span(
                    name="legal-retrieval",
                    input={"query": query},
                    output={"count": len(retrieved_documents), "documents": docs_summary},
                    metadata={"latency_ms": latency_ms},
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Langfuse retrieval span failed: %s", e)

        return span_record

    def record_generation(
        self,
        trace_data: dict[str, Any],
        prompt: str,
        output_text: str,
        model_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ttft_ms: float | None = None,
        latency_ms: float | None = None,
    ) -> dict[str, Any]:
        """Record LLM prompt, response, token usage, and Time to First Token."""
        gen_record = {
            "model": model_name,
            "prompt": prompt,
            "output": output_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "ttft_ms": ttft_ms,
            "latency_ms": latency_ms,
        }
        trace_data["generations"].append(gen_record)
        trace_data["output"] = output_text

        lf_trace = trace_data.get("_langfuse_trace")
        if lf_trace:
            try:
                lf_trace.generation(
                    name="legal-llm-generation",
                    model=model_name,
                    input=prompt,
                    output=output_text,
                    usage={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": prompt_tokens + completion_tokens,
                    },
                    metadata={
                        "ttft_ms": ttft_ms,
                        "latency_ms": latency_ms,
                    },
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("Langfuse generation logging failed: %s", e)

        return gen_record

    def flush(self) -> None:
        """Flush queued traces to Langfuse backend if connected."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:  # noqa: BLE001
                logger.debug("Langfuse flush failed: %s", e)


# Global tracer singleton
tracer = LegalRAGTracer()
