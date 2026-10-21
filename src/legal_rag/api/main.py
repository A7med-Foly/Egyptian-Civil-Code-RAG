"""FastAPI application for Arabic Legal Document Q&A."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from legal_rag.api.middleware import CorrelationIdMiddleware
from legal_rag.api.rate_limiter import rate_limiter
from legal_rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetadataResponse,
)
from legal_rag.config import settings
from legal_rag.guardrails.input_guardrails import detect_prompt_injection, redact_pii
from legal_rag.guardrails.output_guardrails import validate_citation_grounding
from legal_rag.indexing.vector_store import LegalVectorStore
from legal_rag.logging_conf import get_correlation_id, setup_logging
from legal_rag.monitoring.langfuse_tracer import tracer
from legal_rag.monitoring.metrics import (
    INFLIGHT_REQUESTS,
    PREDICTION_LATENCY_SECONDS,
    get_latest_metrics,
    record_prediction,
    track_stage_latency,
)
from legal_rag.rag.pipeline import LegalRAGPipeline

# Initialize structured logging
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger("legal_rag.api.main")

# Global pipeline instance initialized during lifespan
pipeline_instance: LegalRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models, embeddings, and vector index once at startup; clean up at shutdown."""
    global pipeline_instance
    logger.info("Initializing legal RAG models and index during application lifespan startup...")
    try:
        vector_store = LegalVectorStore()
        pipeline_instance = LegalRAGPipeline(vector_store=vector_store)
        logger.info(
            f"Startup complete. Vector store loaded with {vector_store.documents_indexed} documents in memory."
        )
    except Exception:
        logger.exception("Fatal error loading models at startup")
        pipeline_instance = None
    yield
    logger.info("Application shutting down, releasing resources...")
    pipeline_instance = None


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Production-grade Arabic Legal Document Q&A RAG system for the Egyptian Civil Code.",
    lifespan=lifespan,
)

# Attach request tracing middleware
app.add_middleware(CorrelationIdMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return clean 422 with structured error message."""
    logger.warning(
        f"Validation error on {request.url.path}: {exc.errors()}",
        extra={"errors": exc.errors()},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unprocessable Entity",
            "message": "Input validation failed. Please provide a valid, non-empty legal question.",
            "details": jsonable_encoder(exc.errors()),
            "correlation_id": get_correlation_id(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unexpected exceptions: log full traceback internally, return sanitized 500."""
    logger.exception(f"Unhandled internal server error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing your legal query.",
            "correlation_id": get_correlation_id(),
        },
    )


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Liveness & readiness probe: returns 200 only if model and index are in memory."""
    if pipeline_instance is None or pipeline_instance.vector_store is None:
        logger.error("Health check failed: models not loaded in memory")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model and vector store are not loaded in memory.",
        )
    return HealthResponse(
        status="healthy",
        documents_indexed=pipeline_instance.vector_store.documents_indexed,
    )


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus exposition format endpoint for scraping."""
    metrics_bytes = get_latest_metrics()
    return Response(
        content=metrics_bytes,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/metadata", response_model=MetadataResponse, status_code=status.HTTP_200_OK)
async def get_metadata():
    """Return system configuration, versions, and indexed article counts."""
    count = pipeline_instance.vector_store.documents_indexed if pipeline_instance else 0
    return MetadataResponse(
        api_version=settings.API_VERSION,
        model_name=settings.LLM_MODEL_NAME,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        documents_indexed=count,
        chunking_strategy=settings.CHUNKING_STRATEGY,
    )


@app.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_question(payload: AskRequest, request: Request):
    """Answer an Arabic or English legal question with verified Egyptian Civil Code citations."""
    if pipeline_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is warming up or models are not loaded.",
        )

    # 1. Rate Limiting Check (Token Bucket)
    client_ip = request.client.host if request.client else "anonymous"
    rate_res = rate_limiter.check_rate_limit(client_ip)
    if not rate_res.allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(rate_res.retry_after_seconds)},
            content={
                "error": "Too Many Requests",
                "message": f"Rate limit exceeded. Please retry after {rate_res.retry_after_seconds} seconds.",
                "retry_after": rate_res.retry_after_seconds,
                "correlation_id": get_correlation_id(),
            },
        )

    # 2. Input Guardrails: Prompt Injection Filter & PII Redaction
    with track_stage_latency("guardrails"):
        injection_res = detect_prompt_injection(payload.question)
        if not injection_res.passed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Security Guardrail: {injection_res.reason}",
            )
        pii_res = redact_pii(payload.question)
        sanitized_question = pii_res.sanitized_text

    # 3. Instrumented Execution with Stage Timings, Langfuse Tracing & In-flight Saturation
    INFLIGHT_REQUESTS.inc()
    status_label = "success"
    with tracer.trace_query(
        query=sanitized_question,
        correlation_id=get_correlation_id(),
        user_id=client_ip,
    ) as trace:
        try:
            with track_stage_latency("total"):
                result = await pipeline_instance.answer_question(
                    question=sanitized_question,
                    top_k=payload.top_k,
                )

            # Record internal pipeline stage latencies
            retrieval_sec = float(result.get("retrieval_latency_ms", 0.0)) / 1000.0
            generation_sec = float(result.get("generation_latency_ms", 0.0)) / 1000.0
            PREDICTION_LATENCY_SECONDS.labels(stage="retrieval").observe(retrieval_sec)
            PREDICTION_LATENCY_SECONDS.labels(stage="generation").observe(generation_sec)

            # Record Langfuse spans
            tracer.record_retrieval_span(
                trace_data=trace,
                query=sanitized_question,
                retrieved_documents=result.get("retrieved_documents", []),
                latency_ms=result.get("retrieval_latency_ms", 0.0),
            )
            tracer.record_generation(
                trace_data=trace,
                prompt=sanitized_question,
                output_text=result["answer"],
                model_name=settings.LLM_MODEL_NAME,
                prompt_tokens=len(sanitized_question.split()),
                completion_tokens=len(result["answer"].split()),
                latency_ms=result.get("generation_latency_ms", 0.0),
            )

            # 4. Output Guardrail: Citation Grounding Check
            with track_stage_latency("guardrails"):
                retrieved_texts = [
                    (
                        getattr(d, "text", "")
                        if hasattr(d, "text")
                        else (d.get("text", "") if isinstance(d, dict) else str(d))
                    )
                    for d in result.get("retrieved_documents", [])
                ]
                validate_citation_grounding(
                    answer=result["answer"],
                    sources=result["sources"],
                    retrieved_texts=retrieved_texts,
                )

            record_prediction(
                model_version=settings.API_VERSION,
                status="success",
                provider=settings.LLM_PROVIDER,
                score=0.85,
            )

            return AskResponse(
                answer=result["answer"],
                sources=result["sources"],
                model_version=settings.API_VERSION,
                correlation_id=get_correlation_id(),
                latency_ms=result["total_latency_ms"],
            )
        except Exception:
            status_label = "error"
            record_prediction(
                model_version=settings.API_VERSION,
                status=status_label,
                provider=settings.LLM_PROVIDER,
            )
            raise
        finally:
            INFLIGHT_REQUESTS.dec()


@app.post("/ask/stream")
async def ask_question_stream(payload: AskRequest, request: Request):
    """Stream answer tokens progressively."""
    if pipeline_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is warming up or models are not loaded.",
        )

    # Rate Limiting
    client_ip = request.client.host if request.client else "anonymous"
    rate_res = rate_limiter.check_rate_limit(client_ip)
    if not rate_res.allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(rate_res.retry_after_seconds)},
            content={
                "error": "Too Many Requests",
                "message": f"Rate limit exceeded. Please retry after {rate_res.retry_after_seconds} seconds.",
                "retry_after": rate_res.retry_after_seconds,
            },
        )

    # Input guardrails
    injection_res = detect_prompt_injection(payload.question)
    if not injection_res.passed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security Guardrail: {injection_res.reason}",
        )

    pii_res = redact_pii(payload.question)
    sanitized_question = pii_res.sanitized_text

    async def token_stream():
        INFLIGHT_REQUESTS.inc()
        try:
            with track_stage_latency("generation"):
                async for token in pipeline_instance.answer_stream(
                    sanitized_question, top_k=payload.top_k
                ):
                    yield token
            record_prediction(
                model_version=settings.API_VERSION,
                status="success",
                provider=settings.LLM_PROVIDER,
            )
        finally:
            INFLIGHT_REQUESTS.dec()

    return StreamingResponse(token_stream(), media_type="text/plain")


def start():
    """CLI entry point to run uvicorn server."""
    uvicorn.run(
        "legal_rag.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    start()
