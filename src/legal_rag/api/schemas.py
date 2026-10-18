"""Pydantic schemas for Arabic Legal Q&A API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Request schema for asking a legal question."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The legal question to answer, in Arabic or English.",
    )
    top_k: int | None = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant statutory articles to retrieve.",
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Question cannot be empty or whitespace only.")
        return clean

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "ما هي شروط العقد والقوة الملزمة له؟",
                "top_k": 3,
            }
        }
    }


class AskResponse(BaseModel):
    """Response schema containing grounded answer and verified statutory citations."""

    answer: str = Field(..., description="Grounded legal answer.")
    sources: list[str] = Field(
        ...,
        description="List of statutory article citations (e.g. 'Egyptian Civil Code, Article 147').",
    )
    model_version: str = Field(default="0.1.0", description="Serving model version.")
    correlation_id: str = Field(..., description="Unique request tracing ID.")
    latency_ms: float = Field(..., description="End-to-end processing latency in milliseconds.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "وفقاً للمادة 147 من القانون المدني المصري، فإن العقد شريعة المتعاقدين...",
                "sources": ["Egyptian Civil Code, Article 147"],
                "model_version": "0.1.0",
                "correlation_id": "c1f7a3e2-8b9a-4c12-98ab-34567890abcd",
                "latency_ms": 42.5,
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response confirming in-memory readiness and indexed count."""

    status: str = Field(..., description="Service status: healthy or degraded.")
    documents_indexed: int = Field(..., description="Count of articles/chunks currently indexed.")


class MetadataResponse(BaseModel):
    """System metadata response detailing models, versions, and configurations."""

    api_version: str
    model_name: str
    embedding_model: str
    documents_indexed: int
    chunking_strategy: str
    corpus_name: str = "Egyptian Civil Code (Law of 1948)"
