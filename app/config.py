from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "RAG Backend"
    api_v1_prefix: str = "/api/v1"
    environment: str = "development"

    # LLM
    openai_api_key: str
    openai_base_url: str
    llm_model: str = "openai/gpt-oss-20b:fastest"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "document_chunks"

    # SQL
    database_url: str = "sqlite+aiosqlite:///./rag_backend.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_chat_ttl_seconds: int = 86400
    redis_max_history_messages: int = 12

    # Chunking
    fixed_chunk_size: int = 800
    fixed_chunk_overlap: int = 120
    semantic_chunk_target_size: int = 900
    semantic_chunk_max_size: int = 1400

    # Retrieval
    retrieval_top_k: int = 4
    retrieval_score_threshold: float = 0.0

    # Upload
    max_upload_size_mb: int = 25


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()