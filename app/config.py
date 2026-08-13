"""Central configuration. Every value is overridable via environment or .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    # --- Generation ---
    # ANTHROPIC_API_KEY has no RAG_ prefix, so it is read explicitly below.
    llm_model: str = "claude-opus-5"
    llm_effort: str = "low"
    llm_max_tokens: int = 1024
    llm_thinking: str = "disabled"  # "disabled" for latency | "adaptive" for hard questions

    # --- Embeddings ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_backend: str = "onnx_quantized"
    embed_batch_size: int = 64
    embed_cache_size: int = 1024

    # --- Vector store ---
    chroma_path: Path = Path("./data/chroma")
    collection: str = "documents"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    text_columns: str = ""

    # --- Retrieval ---
    top_k: int = 5
    candidate_k: int = 20
    mmr_lambda: float = 0.5
    max_subqueries: int = 3

    # --- Service ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    @field_validator("embed_backend")
    @classmethod
    def _valid_backend(cls, v: str) -> str:
        allowed = {"onnx_quantized", "sentence_transformers"}
        if v not in allowed:
            raise ValueError(f"embed_backend must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("llm_effort")
    @classmethod
    def _valid_effort(cls, v: str) -> str:
        allowed = {"low", "medium", "high", "xhigh", "max"}
        if v not in allowed:
            raise ValueError(f"llm_effort must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("llm_thinking")
    @classmethod
    def _valid_thinking(cls, v: str, info) -> str:
        allowed = {"disabled", "adaptive"}
        if v not in allowed:
            raise ValueError(f"llm_thinking must be one of {sorted(allowed)}, got {v!r}")
        # The API rejects disabled thinking above "high" effort, so catch the
        # bad combination at startup rather than on the first request.
        if v == "disabled" and info.data.get("llm_effort") in {"xhigh", "max"}:
            raise ValueError(
                "llm_thinking='disabled' is not permitted with llm_effort='xhigh' or 'max'; "
                "use llm_thinking='adaptive' or lower the effort"
            )
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_fits(cls, v: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @property
    def text_column_list(self) -> list[str]:
        """Explicit CSV/JSON text columns, or [] to auto-detect."""
        return [c.strip() for c in self.text_columns.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
