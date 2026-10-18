# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies into /install
RUN uv pip install --no-cache --system --prefix=/install .

# Preload embedding model into cache using the installed libraries
ENV HF_HOME=/build/hf_cache
RUN PYTHONPATH="/install/lib/python3.11/site-packages" python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Stage 2: Minimal runtime image
FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy preloaded model cache to appuser's home
COPY --from=builder --chown=appuser:appuser /build/hf_cache /home/appuser/.cache/huggingface

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/processed/ ./data/processed/
COPY --chown=appuser:appuser data/index/ ./data/index/

USER appuser
ENV PYTHONPATH="/app/src"
ENV HF_HOME="/home/appuser/.cache/huggingface"
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "legal_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
