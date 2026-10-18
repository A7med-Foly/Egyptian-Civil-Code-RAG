# Arabic Legal Document Q&A — Egyptian Civil Code RAG

An enterprise-grade, production-oriented Retrieval-Augmented Generation (RAG) system for querying the bilingual Egyptian Civil Code (القانون المدني المصري, Law of 1948) with verified article citations.

## Quickstart (3 Commands)

```bash
# 1. Install package in editable mode
pip install -e ".[dev]"

# 2. Run unit and integration tests with coverage
pytest -v --cov=src/legal_rag --cov-fail-under=70

# 3. Start the production FastAPI service
uvicorn legal_rag.api.main:app --host 0.0.0.0 --port 8000
```

## Docker Quickstart

```bash
docker compose -f docker/docker-compose.yml up -d
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "ما هي شروط العقد؟"}'
```

Developed as the graduation project for the **MLOps Practitioner Bootcamp** (ITI × MLOps MENA Community).
