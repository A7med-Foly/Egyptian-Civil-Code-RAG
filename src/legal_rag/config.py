"""Centralized application configuration using pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override and sane defaults."""

    # Project paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw" / "Egyptian-Civil-Code.pdf"
    PROCESSED_DATA_PATH: Path = DATA_DIR / "processed" / "egyptian_civil_code.json"
    INDEX_DIR: Path = DATA_DIR / "index" / "chroma_db"
    REPORTS_DIR: Path = BASE_DIR / "reports"

    # API configuration
    API_TITLE: str = "Arabic Legal Document Q&A API"
    API_VERSION: str = "0.1.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORRELATION_HEADER: str = "X-Request-ID"

    # Model & Retrieval configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CHUNKING_STRATEGY: str = "article"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.35

    # LLM configuration (Groq, OpenAI-compatible, Mock)
    LLM_PROVIDER: str = "groq"  # "groq", "openai_compatible", "local", "mock"
    LLM_MODEL_NAME: str = "openai/gpt-oss-120b"
    LLM_API_BASE: str = "https://api.groq.com/openai/v1"
    LLM_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    LLM_TEMPERATURE: float = 0.0

    def get_api_key(self) -> str | None:
        """Retrieve either GROQ_API_KEY or generic LLM_API_KEY."""
        return self.GROQ_API_KEY or self.LLM_API_KEY

    # MLflow configuration (defaults to local SQLite with full Registry support, overridable by env)
    MLFLOW_TRACKING_URI: str = "sqlite:///data/mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "arabic-legal-rag"
    MLFLOW_S3_ENDPOINT_URL: str = "http://localhost:9000"

    # S3 / MinIO configuration
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"
    AWS_ENDPOINT_URL: str = "http://localhost:9000"

    # Langfuse configuration
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
