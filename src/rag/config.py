from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    llm_model: str = "claude-opus-4-8"
    anthropic_api_key: str | None = None
    dataset_path: Path = Path(__file__).resolve().parents[2] / "corpus" / "benchmark.yaml"
    index_id: str | None = None
    chunk_target_words: int = Field(default=180, gt=0)
    chunk_overlap_words: int = Field(default=30, ge=0)

    @model_validator(mode="after")
    def valid_chunk_window(self) -> Settings:
        if self.chunk_overlap_words >= self.chunk_target_words:
            raise ValueError(
                "chunk_overlap_words must be smaller than chunk_target_words"
            )
        return self


def get_settings() -> Settings:
    return Settings()
