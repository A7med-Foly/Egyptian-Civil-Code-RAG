# Arabic Legal Document Q&A — Egyptian Civil Code RAG

An enterprise-grade, production-oriented Retrieval-Augmented Generation (RAG) system for querying the bilingual Egyptian Civil Code (*القانون المدني المصري*, Law of 1948) with verified statutory article citations.

Developed as the graduation project for the **MLOps Practitioner Bootcamp** (ITI × MLOps MENA Community).

---

## 3-Command Quickstart (Zero to Prediction)

Any engineer or reviewer can run and query the complete containerized service in three commands without installing Python or cloning weights:

```bash
# 1. Start the containerized service
docker compose -f docker/docker-compose.yml up -d

# 2. Check service health and indexed corpus count
curl -s http://localhost:8000/health

# 3. Query the legal RAG service
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي شروط العقد والقوة الملزمة له؟"}'
```

Example response:
```json
{
  "answer": "وفقاً لما ورد في Egyptian Civil Code, Article 147 من القانون المدني المصري: تنص الأحكام القانونية ذات الصلة على أن العقد شريعة المتعاقدين...",
  "sources": ["Egyptian Civil Code, Article 147"],
  "model_version": "0.1.0",
  "correlation_id": "0a93b788-6afd-4bfc-81fc-f7c3a1a19dad",
  "latency_ms": 38.7
}
```

---

## Local Developer Workflow

```bash
# 1. Install package in editable mode with development dependencies
uv sync --dev
# OR: pip install -e ".[dev]"

# 2. Lint and format code
ruff check src tests && black --check src tests

# 3. Run complete test suite with coverage enforcement (>= 70%)
pytest -v --cov=src/legal_rag --cov-fail-under=70

# 4. Ingest raw PDF and build vector index
python -m legal_rag.ingestion.extractor
python -m legal_rag.indexing.vector_store

# 5. Start development API server
uvicorn legal_rag.api.main:app --reload --port 8000
```

---

## Repository Layout

```
.
├── README.md                      # 3-command quickstart, API documentation, workflow
├── pyproject.toml                 # Package metadata, CPU-torch index, dependencies
├── .gitignore                     # Data, cache, and credential filtering
├── .dockerignore                  # Minimal container build context
├── .pre-commit-config.yaml        # Ruff, Black, and git hooks
│
├── data/
│   ├── raw/                       # Raw PDF (Egyptian-Civil-Code.pdf, tracked by DVC)
│   ├── processed/                 # egyptian_civil_code.json (1,149 verified articles)
│   └── index/                     # Persistent ChromaDB vector store
│
├── src/legal_rag/                 # The installable Python package
│   ├── config.py                  # Centralized Pydantic-settings
│   ├── logging_conf.py            # Structured JSON formatter & correlation IDs
│   ├── ingestion/
│   │   ├── extractor.py           # Column-aware bilingual PDF extractor
│   │   ├── normalizer.py          # Arabic text & Arabic-Indic numeral cleaner
│   │   └── validator.py           # Corpus continuity & repeal verification
│   ├── indexing/
│   │   ├── chunker.py             # Article-aware legal chunker
│   │   ├── embedder.py            # Multilingual sentence-transformers wrapper
│   │   ├── export.py              # ONNX model export & parity checker
│   │   └── vector_store.py        # ChromaDB + BM25 hybrid search engine
│   ├── rag/
│   │   ├── generator.py           # LLM client (Mock, local, OpenAI-compatible)
│   │   ├── pipeline.py            # Complete RAG pipeline orchestrator
│   │   └── prompts.py             # Versioned Arabic legal prompts & disclaimers
│   └── api/
│       ├── main.py                # FastAPI app with lifespan model loading
│       ├── schemas.py             # Pydantic v2 schemas (/ask, /health, /metadata)
│       └── middleware.py          # X-Request-ID & latency logging middleware
│
├── tests/                         # Pytest suite (33 passed, 71.6% coverage)
│   ├── conftest.py                # Session-scoped fixtures and test client
│   ├── test_api.py                # API endpoints and 422 error handling
│   ├── test_chunker.py            # Article preservation and paragraph splitting
│   ├── test_config_and_logging.py # Config defaults and correlation ID context
│   ├── test_extraction.py         # 1,149 articles continuity & spot-checks
│   ├── test_normalizer.py         # Tashkeel, numeral reversal, orthography
│   ├── test_retrieval.py          # Hybrid ChromaDB + BM25 search
│   └── test_serialization.py      # ONNX Runtime parity verification
│
├── docker/
│   ├── Dockerfile                 # Multi-stage non-root container (434MB compressed)
│   └── docker-compose.yml         # API service deployment
│
└── reports/
    └── module-1.md                # Module 1 lab report & MLOps maturity self-assessment
```

---

## Module 1 Highlights

- **Extraction Fidelity:** 1,149/1,149 articles parsed, 1,093 active, 56 repealed (Articles 54–80 & 389–417 flagged). Zero missing articles, zero empty Arabic bodies.
- **ONNX Speedup:** ONNX Runtime provides a **1.60x CPU speedup** (39.5ms vs 63.0ms) over PyTorch eager mode with maximum absolute error $\le 2.71 \times 10^{-6}$.
- **Security & Multi-Stage Docker:** Container runs under unprivileged user `appuser` (UID 1000) with baked weights and precomputed vector index.
- **Enforced Code Standards:** 100% compliance with `ruff` and `black`; strict **$\ge 70\%$ test coverage gate** passed.
