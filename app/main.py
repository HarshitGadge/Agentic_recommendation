"""FastAPI service exposing ingestion, retrieval, and grounded answering."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app import __version__
from app.agents.orchestrator import RetrievalAgent
from app.config import Settings, get_settings
from app.generation.answerer import Answerer, build_answerer
from app.ingest.pipeline import ingest_path
from app.obs.logging_conf import configure_logging
from app.obs.metrics import registry
from app.retrieval.embeddings import CachedEmbedder, Embedder, build_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.store import VectorStore
from app.schemas import (
    Citation,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StageTimings,
    StatsResponse,
)

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 320


@dataclass
class AppState:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    retriever: Retriever
    agent: RetrievalAgent
    answerer: Answerer


def build_state(settings: Settings) -> AppState:
    """Wire the object graph once, at startup."""
    embedder = build_embedder(
        backend=settings.embed_backend,
        model_name=settings.embed_model,
        batch_size=settings.embed_batch_size,
        cache_size=settings.embed_cache_size,
    )
    # Load the model and initialise the compute graph before serving, so the
    # first real request doesn't pay for it.
    embedder.warmup()

    store = VectorStore(
        embedder=embedder,
        path=settings.chroma_path,
        collection_name=settings.collection,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    retriever = Retriever(
        store=store,
        top_k=settings.top_k,
        candidate_k=settings.candidate_k,
        mmr_lambda=settings.mmr_lambda,
    )
    agent = RetrievalAgent(retriever=retriever, max_subqueries=settings.max_subqueries)
    answerer = build_answerer(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        effort=settings.llm_effort,
        thinking=settings.llm_thinking,
    )
    return AppState(
        settings=settings,
        embedder=embedder,
        store=store,
        retriever=retriever,
        agent=agent,
        answerer=answerer,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting agentic-rag v%s", __version__)
    app.state.rag = build_state(settings)
    logger.info(
        "Ready: %d chunks indexed, embed=%s(%s), answers=%s",
        app.state.rag.store.count(),
        app.state.rag.embedder.name,
        app.state.rag.embedder.model_name,
        app.state.rag.answerer.mode,
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Agentic RAG Service",
    version=__version__,
    description=(
        "Retrieval-augmented generation over a local corpus. LangChain chunking, "
        "ChromaDB vector search, an adaptive retrieval agent, and grounded answers "
        "with citations."
    ),
    lifespan=lifespan,
)


def get_state(request: Request) -> AppState:
    state = getattr(request.app.state, "rag", None)
    if state is None:  # pragma: no cover - only if lifespan failed
        raise HTTPException(status_code=503, detail="Service not initialised")
    return state


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    registry.increment("errors")
    logger.exception("Unhandled error (request_id=%s)", request_id)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# --------------------------------------------------------------------------
# Health / introspection
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    checks: dict[str, str] = {}
    status = "ok"

    try:
        count = state.store.count()
        checks["vector_store"] = f"ok ({count} chunks)"
        if count == 0:
            checks["vector_store"] = "ok (empty - nothing ingested yet)"
    except Exception as exc:
        checks["vector_store"] = f"error: {exc}"
        status = "degraded"

    checks["embedder"] = f"ok ({state.embedder.name}/{state.embedder.model_name})"
    checks["answerer"] = state.answerer.mode
    if state.answerer.mode == "extractive":
        checks["answerer"] = "extractive (ANTHROPIC_API_KEY not set)"

    return HealthResponse(status=status, version=__version__, checks=checks)


@app.get("/stats", response_model=StatsResponse, tags=["ops"])
def stats(state: AppState = Depends(get_state)) -> StatsResponse:
    return StatsResponse(
        collection=state.settings.collection,
        chunk_count=state.store.count(),
        sources=state.store.source_breakdown(),
        embed_model=state.embedder.model_name,
        embed_backend=state.embedder.name,
        embed_dimension=state.embedder.dimension,
        llm_model=state.settings.llm_model if state.answerer.mode == "llm" else None,
        answer_mode="llm" if state.answerer.mode == "llm" else "extractive",
    )


@app.get("/metrics", tags=["ops"])
def metrics(state: AppState = Depends(get_state)) -> dict[str, Any]:
    payload = registry.snapshot()
    if isinstance(state.embedder, CachedEmbedder):
        payload["embed_cache"] = state.embedder.cache_stats()
    payload["chunk_count"] = state.store.count()
    return payload


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
def ingest(body: IngestRequest, state: AppState = Depends(get_state)) -> IngestResponse:
    """Ingest a file or directory already present on the server's filesystem."""
    try:
        result = ingest_path(
            path=body.path,
            store=state.store,
            chunk_size=state.settings.chunk_size,
            chunk_overlap=state.settings.chunk_overlap,
            recursive=body.recursive,
            source_tag=body.source_tag,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    registry.increment("ingests")
    return IngestResponse(**result.__dict__)


@app.post("/ingest/upload", response_model=IngestResponse, tags=["ingest"])
async def ingest_upload(
    files: list[UploadFile] = File(...),
    source_tag: str | None = None,
    state: AppState = Depends(get_state),
) -> IngestResponse:
    """Ingest uploaded files without needing filesystem access on the server."""
    staging = Path(tempfile.mkdtemp(prefix="rag-upload-"))
    try:
        for upload in files:
            if not upload.filename:
                continue
            # Strip any directory component -- never trust a client filename.
            target = staging / Path(upload.filename).name
            with target.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)

        result = ingest_path(
            path=staging,
            store=state.store,
            chunk_size=state.settings.chunk_size,
            chunk_overlap=state.settings.chunk_overlap,
            recursive=False,
            source_tag=source_tag,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    registry.increment("ingests")
    return IngestResponse(**result.__dict__)


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(
    body: QueryRequest, request: Request, state: AppState = Depends(get_state)
) -> QueryResponse:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    started = time.perf_counter()

    outcome = state.agent.run(
        question=body.question,
        top_k=body.top_k,
        decompose=body.decompose,
        source_filter=body.source_filter,
    )

    generate_ms = 0.0
    if body.generate:
        t0 = time.perf_counter()
        answer = state.answerer.answer(body.question, outcome.hits)
        generate_ms = (time.perf_counter() - t0) * 1000
        answer_text, answer_mode = answer.text, answer.mode
    else:
        answer_text, answer_mode = "", "none"

    total_ms = (time.perf_counter() - started) * 1000
    timings = StageTimings(
        plan_ms=round(outcome.plan_ms, 2),
        embed_ms=round(outcome.embed_ms, 2),
        search_ms=round(outcome.search_ms, 2),
        rerank_ms=round(outcome.rerank_ms, 2),
        generate_ms=round(generate_ms, 2),
        total_ms=round(total_ms, 2),
    )
    registry.record_query(timings.model_dump())

    citations = [
        Citation(
            chunk_id=hit.chunk_id,
            source=hit.source,
            score=hit.score,
            snippet=hit.text[:SNIPPET_CHARS].strip(),
            metadata=hit.metadata,
        )
        for hit in outcome.hits
    ]

    logger.info(
        "query id=%s rounds=%d hits=%d total=%.0fms (embed %.0f / search %.0f / gen %.0f)",
        request_id,
        outcome.rounds,
        len(outcome.hits),
        total_ms,
        timings.embed_ms,
        timings.search_ms,
        timings.generate_ms,
    )

    return QueryResponse(
        question=body.question,
        sub_queries=outcome.plan.sub_queries,
        answer=answer_text,
        answer_mode=answer_mode,  # type: ignore[arg-type]
        citations=citations,
        timings=timings,
        request_id=request_id,
    )


def run() -> None:  # pragma: no cover - entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
