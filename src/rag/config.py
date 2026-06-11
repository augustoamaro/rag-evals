from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    llm_model: str = "claude-opus-4-8"
    anthropic_api_key: str | None = None


def get_settings() -> Settings:
    return Settings()
