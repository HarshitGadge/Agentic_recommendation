"""Pydantic request/response models -- the public API contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(
        ..., description="File or directory to ingest, read from the server's filesystem."
    )
    recursive: bool = Field(True, description="Walk subdirectories when path is a directory.")
    source_tag: str | None = Field(
        None, description="Optional label stored on every chunk, e.g. a Kaggle dataset slug."
    )


class IngestResponse(BaseModel):
    files_seen: int
    files_ingested: int
    files_skipped: list[str] = Field(default_factory=list)
    chunks_written: int
    duplicates_dropped: int
    elapsed_ms: float


class Citation(BaseModel):
    chunk_id: str
    source: str
    score: float
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageTimings(BaseModel):
    """Per-stage wall-clock, in milliseconds. This is what the benchmark reads."""

    plan_ms: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(None, ge=1, le=50)
    decompose: bool = Field(True, description="Let the planner split multi-part questions.")
    generate: bool = Field(
        True, description="Set false to get retrieval only, no answer synthesis."
    )
    source_filter: str | None = Field(None, description="Restrict retrieval to one source_tag.")


class QueryResponse(BaseModel):
    question: str
    sub_queries: list[str]
    answer: str
    answer_mode: Literal["llm", "extractive", "none"]
    citations: list[Citation]
    timings: StageTimings
    request_id: str


class StatsResponse(BaseModel):
    collection: str
    chunk_count: int
    sources: dict[str, int]
    embed_model: str
    embed_backend: str
    embed_dimension: int
    llm_model: str | None
    answer_mode: Literal["llm", "extractive"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    checks: dict[str, str]
