"""FastAPI application for Arabic Legal Document Q&A."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from legal_rag.api.middleware import CorrelationIdMiddleware
from legal_rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetadataResponse,
)
from legal_rag.config import settings
from legal_rag.indexing.vector_store import LegalVectorStore
from legal_rag.logging_conf import get_correlation_id, setup_logging
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
async def ask_question(payload: AskRequest):
    """Answer an Arabic or English legal question with verified Egyptian Civil Code citations."""
    if pipeline_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is warming up or models are not loaded.",
        )

    result = await pipeline_instance.answer_question(
        question=payload.question,
        top_k=payload.top_k,
    )

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        model_version=settings.API_VERSION,
        correlation_id=get_correlation_id(),
        latency_ms=result["total_latency_ms"],
    )


@app.post("/ask/stream")
async def ask_question_stream(payload: AskRequest):
    """Stream answer tokens progressively."""
    if pipeline_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is warming up or models are not loaded.",
        )

    async def token_stream():
        async for token in pipeline_instance.answer_stream(payload.question, top_k=payload.top_k):
            yield token

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
