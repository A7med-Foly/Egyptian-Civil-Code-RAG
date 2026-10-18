# Module 1 Report: From Notebook to Production-Ready Service

**Project:** Project 2 — LLM / RAG: Arabic Legal Document Q&A  
**Corpus:** Egyptian Civil Code (*القانون المدني المصري*, Law of 1948)  
**Deliverable:** Pip-installable package `legal-rag`, tested FastAPI service, ONNX serialized model, and multi-stage Docker container.

---

## 1. Corpus Extraction & Ingestion Verification

The raw bilingual PDF (`Egyptian-Civil-Code.pdf`, 170 pages) was extracted using a custom column-aware, layout-preserving extraction pipeline (`legal_rag.ingestion.extractor` and `normalizer`).

| Metric | Target Requirement | Measured Result | Status |
|---|---|---|---|
| **Total Articles Extracted** | 1,149 | 1,149 | PASS |
| **Active Articles** | 1,093 | 1,093 | PASS |
| **Repealed Articles Flagged** | 56 (54–80, 389–417) | 56 | PASS |
| **Missing / Unexplained Gaps** | 0 | 0 | PASS |
| **Empty Arabic Records (`text_ar`)** | 0 | 0 | PASS |
| **Citation Format Compliance** | 100% | 100% (`Egyptian Civil Code, Article {N}`) | PASS |

### Arabic Parsing Pitfalls Resolved:
1. **Arabic-Indic Numeral Reversal:** Digit sequences extracted in reverse visual order (e.g. `٩٤١١`) are byte-reversed and normalized to Western integer digits (`1149`).
2. **Column De-interleaving:** Page midpoint thresholding ($x \approx 285$) cleanly separates the English left column from the Arabic right column, eliminating word bleed.
3. **Repealed Provisions Preserved:** Articles 54–80 (Decree Law 384/1956) and 389–417 (Evidence Law 25/1968) are flagged with statutory repeal reasons rather than omitted, preventing hallucinations.

---

## 2. Serialization Format Comparison & ONNX Parity

### Format Evaluation Matrix

| Format | Human-Readable | Cross-Language | Schema-Enforced | Safe from Untrusted Source? | Primary Use in System |
|---|---|---|---|---|---|
| **JSON** | Yes | Yes | Yes (via Pydantic) | Yes | Structured corpus & API payload |
| **Protobuf** | No (binary) | Yes | Yes (`.proto`) | Yes | gRPC & IPC communications |
| **Pickle** | No (binary) | Python-only | No | **NO (Arbitrary code exec)** | Strictly forbidden in production |
| **ONNX** | No (graph) | Yes (C++, Java, Rust, JS) | Yes (ONNX Spec) | Yes | High-performance embedding inference |

> [!CAUTION]
> **The Golden Security Rule:** Pickle executes arbitrary bytecode on `pickle.load()`. Never load an unverified `.pkl` file in a production environment. Our application uses **ONNX Runtime** for model inference and strict **Pydantic schemas** for data transfer.

### PyTorch vs. ONNX Runtime Parity & Benchmark

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Hardware:** Intel x86_64 CPU (CPUExecutionProvider)
- **Batch Size:** Dynamic (tested on 4 multilingual legal queries)

| Runtime | Mean Latency | Max Absolute Error vs PyTorch | Parity Passed? | Speedup Factor |
|---|---|---|---|---|
| **PyTorch Eager Mode** | 62.98 ms | Baseline (0.0) | N/A | 1.0x |
| **ONNX Runtime (CPU)** | **39.47 ms** | **2.71 × 10⁻⁶** | **YES (`atol=1e-3`)** | **1.60x faster** |

---

## 3. Containerization & Production Packaging

The service was containerized using a multi-stage Docker build running as a non-privileged `appuser`.

| Build Type | Total Virtual Size | Compressed Transfer Size | Security Context |
|---|---|---|---|
| **Single-Stage Build (Naive)** | ~3.8 GB (build tools + caches) | ~1.1 GB | root (uid 0) |
| **Multi-Stage Build (`Dockerfile`)** | **2.05 GB** | **434 MB** | **`appuser` (uid 1000)** |

### Container Verification:
- **Non-Root Execution:** `docker exec test-legal-api whoami` → `appuser`.
- **Preloaded Weights & Index:** All sentence-transformer weights and the 1,149-article ChromaDB vector index are baked into `/home/appuser/.cache` and `/app/data/index/`, enabling 100% offline startup in under 2 seconds.
- **Docker Healthcheck:** Automated probe on `/health` reporting 200 in 2.3 ms.

---

## 4. API Endpoints & Structured Logging

The service exposes 4 endpoints:
- `GET /health`: Returns `{status: "healthy", documents_indexed: 1149}`.
- `GET /metadata`: Returns model versions and architecture configurations.
- `POST /ask`: Returns grounded legal answers with statutory citations.
- `POST /ask/stream`: Server-sent token streaming.

### Structured JSON Logging Sample:
```json
{"timestamp": "2026-10-18 17:52:06,023", "level": "INFO", "logger": "legal_rag.api.middleware", "message": "Incoming request: GET /health", "correlation_id": "70b293be-0ccd-419c-a452-a9acb7acabeb", "path": "/health", "method": "GET"}
{"timestamp": "2026-10-18 17:52:06,026", "level": "INFO", "logger": "legal_rag.api.middleware", "message": "Completed request: GET /health - status=200 in 3.29ms", "correlation_id": "70b293be-0ccd-419c-a452-a9acb7acabeb", "status_code": 200, "duration_ms": 3.29}
```

---

## 5. MLOps Maturity Self-Assessment

### Current Standing: Level 1 (ML Pipeline & Service Automation)
The repository has evolved beyond Level 0 (ad-hoc notebooks):
- Codebase is structured as an installable Python package (`legal_rag`) with `pyproject.toml`.
- Data ingestion and normalization are fully scripted, tested, and reproducible.
- Production serving uses async FastAPI with startup model loading (lifespan) and Pydantic validation.
- Unit and integration tests enforce a **71.64% test coverage gate**.
- Containerization runs in multi-stage Docker as non-root.

### What is Missing to Reach Level 2?
To advance to Level 2 (Continuous Delivery & Automated Pipeline Retraining), the system requires:
1. Automated experiment tracking and model registration via **MLflow** to systematically log chunking and embedding hyperparameter variations against RAGAS faithfulness.
2. Data and pipeline versioning via **DVC** coupled with **GitHub Actions CI/CD** that enforces automated quality gates blocking pull requests when retrieval faithfulness degrades.

*(These components are the explicit focus of Module 2).*
