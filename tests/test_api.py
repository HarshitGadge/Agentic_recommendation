"""HTTP surface: health, ingest, query, stats, metrics."""

from __future__ import annotations

import io

import pytest

from app.generation.answerer import NO_CONTEXT_MESSAGE, ExtractiveAnswerer
from app.obs.metrics import LatencyRegistry, percentile
from app.retrieval.store import SearchHit


# --------------------------------------------------------------------------
# Ops endpoints
# --------------------------------------------------------------------------
def test_health_is_ok_on_a_cold_index(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "vector_store" in body["checks"]


def test_health_reports_extractive_mode_without_a_key(client):
    assert "extractive" in client.get("/health").json()["checks"]["answerer"]


def test_request_id_is_echoed(client):
    response = client.get("/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


def test_request_id_is_generated_when_absent(client):
    assert client.get("/health").headers["x-request-id"]


def test_stats_reflects_ingested_content(client):
    client.post("/ingest", json={"path": client.corpus_path})
    body = client.get("/stats").json()
    assert body["chunk_count"] > 0
    assert body["sources"]
    assert body["answer_mode"] == "extractive"
    assert body["embed_dimension"] == 64


def test_metrics_tracks_queries(client):
    client.post("/ingest", json={"path": client.corpus_path})
    client.post("/query", json={"question": "What is photosynthesis?"})
    body = client.get("/metrics").json()
    assert body["counts"]["queries"] >= 1
    assert "total" in body["latency_ms"]


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------
def test_ingest_directory(client):
    response = client.post("/ingest", json={"path": client.corpus_path})
    assert response.status_code == 200
    body = response.json()
    assert body["files_ingested"] == 4
    assert body["chunks_written"] > 0
    assert body["elapsed_ms"] > 0


def test_ingest_missing_path_returns_404(client):
    response = client.post("/ingest", json={"path": "/nonexistent/path/xyz"})
    assert response.status_code == 404


def test_ingest_with_source_tag_is_filterable(client):
    client.post("/ingest", json={"path": client.corpus_path, "source_tag": "biology"})
    response = client.post(
        "/query", json={"question": "cellular respiration", "source_filter": "biology"}
    )
    citations = response.json()["citations"]
    assert citations
    assert all(c["metadata"]["source_tag"] == "biology" for c in citations)


def _upload(name: str, body: bytes) -> list[tuple[str, tuple[str, io.BytesIO, str]]]:
    return [("files", (name, io.BytesIO(body), "text/plain"))]


def test_ingest_upload_accepts_files(client):
    files = _upload(
        "upload.txt",
        b"Ribosomes assemble proteins from amino acids following the "
        b"instructions carried by messenger RNA.",
    )
    response = client.post("/ingest/upload", files=files)
    assert response.status_code == 200
    assert response.json()["chunks_written"] >= 1


def test_upload_strips_directory_traversal(client):
    files = _upload(
        "../../escape.txt",
        b"Some content that is long enough to be chunked into a retrievable passage here.",
    )
    response = client.post("/ingest/upload", files=files)
    assert response.status_code == 200
    sources = client.get("/stats").json()["sources"]
    assert not any(".." in source for source in sources)


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------
def test_query_returns_answer_and_citations(client):
    client.post("/ingest", json={"path": client.corpus_path})
    response = client.post("/query", json={"question": "How does photosynthesis work?"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert body["answer"]
    assert body["answer_mode"] == "extractive"
    assert body["request_id"]


def test_query_reports_stage_timings(client):
    client.post("/ingest", json={"path": client.corpus_path})
    timings = client.post("/query", json={"question": "mitochondria"}).json()["timings"]
    assert timings["total_ms"] > 0
    assert timings["embed_ms"] >= 0
    assert timings["search_ms"] >= 0
    # Stage times must not exceed the wall-clock total.
    stages = sum(
        timings[k] for k in ("plan_ms", "embed_ms", "search_ms", "generate_ms")
    )
    assert stages <= timings["total_ms"] + 1.0


def test_query_on_empty_index_is_graceful(client):
    body = client.post("/query", json={"question": "anything"}).json()
    assert body["citations"] == []
    assert body["answer"] == NO_CONTEXT_MESSAGE
    assert body["answer_mode"] == "none"


def test_query_respects_top_k(client):
    client.post("/ingest", json={"path": client.corpus_path})
    body = client.post("/query", json={"question": "energy in cells", "top_k": 2}).json()
    assert len(body["citations"]) <= 2


def test_query_generate_false_skips_synthesis(client):
    client.post("/ingest", json={"path": client.corpus_path})
    body = client.post("/query", json={"question": "ATP", "generate": False}).json()
    assert body["answer"] == ""
    assert body["answer_mode"] == "none"
    assert body["citations"]
    assert body["timings"]["generate_ms"] == 0


def test_query_exposes_sub_queries(client):
    client.post("/ingest", json={"path": client.corpus_path})
    body = client.post(
        "/query",
        json={"question": "What is photosynthesis and how does respiration release energy?"},
    ).json()
    assert len(body["sub_queries"]) >= 2


def test_empty_question_is_rejected(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_oversized_top_k_is_rejected(client):
    assert client.post("/query", json={"question": "x", "top_k": 999}).status_code == 422


# --------------------------------------------------------------------------
# Extractive answerer
# --------------------------------------------------------------------------
def test_extractive_answer_cites_passages():
    hits = [
        SearchHit("a", "Photosynthesis converts light energy into chemical energy in plants.",
                  {"source": "a.txt"}, 0.9),
        SearchHit("b", "Chlorophyll is the pigment that absorbs sunlight inside chloroplasts.",
                  {"source": "b.txt"}, 0.7),
    ]
    answer = ExtractiveAnswerer().answer("How does photosynthesis capture light?", hits)
    assert answer.mode == "extractive"
    assert "[1]" in answer.text or "[2]" in answer.text


def test_extractive_answer_without_hits():
    answer = ExtractiveAnswerer().answer("anything", [])
    assert answer.mode == "none"
    assert answer.text == NO_CONTEXT_MESSAGE


# --------------------------------------------------------------------------
# Metrics primitives
# --------------------------------------------------------------------------
def test_percentile_interpolates():
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 100) == pytest.approx(100.0)
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 95) == 7.0


def test_registry_window_bounds_memory():
    registry = LatencyRegistry(window=5)
    for i in range(20):
        registry.record_query({"total_ms": float(i)})
    snapshot = registry.snapshot()
    assert snapshot["counts"]["queries"] == 20
    assert snapshot["latency_ms"]["total"]["count"] == 5  # only the last 5 retained


def test_registry_reset():
    registry = LatencyRegistry()
    registry.record_query({"total_ms": 1.0})
    registry.reset()
    assert registry.snapshot()["counts"]["queries"] == 0
