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

    # LLM configuration (Dual local/API adapter)
    LLM_PROVIDER: str = "mock"  # "mock", "local", "openai_compatible"
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"
    LLM_API_BASE: str | None = None
    LLM_API_KEY: str | None = None
    LLM_TEMPERATURE: float = 0.0

    # MLflow configuration
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "arabic-legal-rag"

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
