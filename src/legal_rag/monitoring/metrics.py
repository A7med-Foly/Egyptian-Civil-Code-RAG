"""Prometheus metrics contract and multiprocess instrumentation for Legal RAG API.

Implements all 8 required metrics according to the Module 5 specification,
and resolves the multiprocess trap via prometheus_client MultiProcessCollector.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from prometheus_client import (
    REGISTRY as DEFAULT_REGISTRY,
)
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

# ---------------------------------------------------------------------------
# Registry Setup & Multiprocess Trap Resolution
# ---------------------------------------------------------------------------
MULTIPROC_DIR = os.getenv("PROMETHEUS_MULTIPROC_DIR")

if MULTIPROC_DIR:
    Path(MULTIPROC_DIR).mkdir(parents=True, exist_ok=True)
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
else:
    registry = DEFAULT_REGISTRY

# ---------------------------------------------------------------------------
# Metric Contract Definitions
# ---------------------------------------------------------------------------
# 1. Total event counter
PREDICTIONS_TOTAL = Counter(
    "legal_rag_predictions_total",
    "Total number of statutory queries processed by the Legal RAG service.",
    ["model_version", "status", "provider"],
    registry=registry,
)
# Pre-initialize standard status labels so series are created and exposed at 0
PREDICTIONS_TOTAL.labels(model_version="0.1.0", status="success", provider="groq")
PREDICTIONS_TOTAL.labels(model_version="0.1.0", status="error", provider="groq")

# 2. Stage-by-stage latency histogram (SLA buckets aligned to sub-second / sub-2.5s targets)
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0)

PREDICTION_LATENCY_SECONDS = Histogram(
    "legal_rag_prediction_latency_seconds",
    "Latency distribution partitioned by pipeline execution stage.",
    ["stage"],
    buckets=LATENCY_BUCKETS,
    registry=registry,
)

# 3. Output prediction score distribution (cheapest continuous drift indicator)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)

PREDICTION_SCORE = Histogram(
    "legal_rag_prediction_score",
    "Distribution of top retrieval similarity scores.",
    buckets=SCORE_BUCKETS,
    registry=registry,
)

# 4. Effective batch size
BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)

INFERENCE_BATCH_SIZE = Histogram(
    "legal_rag_inference_batch_size",
    "Effective request micro-batch size processed by inference engines.",
    buckets=BATCH_SIZE_BUCKETS,
    registry=registry,
)

# 5. In-flight request gauge (instantaneous saturation)
INFLIGHT_REQUESTS = Gauge(
    "legal_rag_inflight_requests",
    "Number of requests currently executing across worker processes.",
    multiprocess_mode="livesum" if MULTIPROC_DIR else "all",
    registry=registry,
)

# 6. Static info gauge (version, git commit, framework)
MODEL_INFO = Gauge(
    "legal_rag_model_info",
    "Active model metadata and runtime framework.",
    ["version", "framework", "commit"],
    registry=registry,
)
# Initialize info gauge with current release
MODEL_INFO.labels(version="v0.3.0", framework="onnx_openvino_groq", commit="c96199e").set(1)

# 7. Guardrail decision counters
GUARDRAIL_DECISIONS_TOTAL = Counter(
    "legal_rag_guardrail_decisions_total",
    "Decisions made by input/output guardrails.",
    ["guardrail", "decision"],
    registry=registry,
)

# 8. Rate limiter counter
RATE_LIMIT_HITS_TOTAL = Counter(
    "legal_rag_rate_limit_hits_total",
    "Rate limiting decisions (allowed vs throttled).",
    ["client_id", "status"],
    registry=registry,
)


# ---------------------------------------------------------------------------
# Convenience Instrumentation Helpers
# ---------------------------------------------------------------------------
@contextmanager
def track_stage_latency(stage: str) -> Generator[None, None, None]:
    """Measure wall-clock latency of an execution stage and record to Prometheus histogram."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - t0
        PREDICTION_LATENCY_SECONDS.labels(stage=stage).observe(duration)


def record_prediction(
    model_version: str,
    status: str,
    provider: str,
    score: float | None = None,
    batch_size: int = 1,
) -> None:
    """Record query metrics on completion."""
    PREDICTIONS_TOTAL.labels(
        model_version=model_version,
        status=status,
        provider=provider,
    ).inc()

    if score is not None:
        PREDICTION_SCORE.observe(float(score))

    INFERENCE_BATCH_SIZE.observe(batch_size)


def record_guardrail(guardrail: str, decision: str) -> None:
    """Record input or output guardrail evaluation."""
    GUARDRAIL_DECISIONS_TOTAL.labels(guardrail=guardrail, decision=decision).inc()


def record_rate_limit(client_id: str, status: str) -> None:
    """Record token bucket rate limit decision ('allowed' or 'throttled')."""
    RATE_LIMIT_HITS_TOTAL.labels(client_id=client_id, status=status).inc()


def get_latest_metrics() -> bytes:
    """Return Prometheus exposition format bytes for the /metrics route."""
    return generate_latest(registry)
