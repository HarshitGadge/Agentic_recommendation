"""Vector store, MMR, RRF, planner, and the agent loop."""

from __future__ import annotations

import pytest

from app.agents.orchestrator import RetrievalAgent, build_context
from app.agents.planner import broaden, plan_queries
from app.retrieval.embeddings import CachedEmbedder
from app.retrieval.retriever import (
    Retriever,
    cosine,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from app.retrieval.store import SearchHit


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
def test_search_on_empty_collection_returns_nothing(store):
    assert store.search("anything at all", k=5) == []


def test_search_returns_relevant_chunk_first(populated_store):
    hits = populated_store.search("chlorophyll absorbs sunlight photosynthesis", k=3)
    assert hits
    assert "photosynthesis" in hits[0].text.lower() or "chlorophyll" in hits[0].text.lower()


def test_scores_are_similarities_not_distances(populated_store):
    hits = populated_store.search("mitochondria ATP", k=3)
    assert hits
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    # Results must arrive in descending relevance.
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_metadata_filter_restricts_results(store, corpus):
    from app.ingest.pipeline import ingest_path

    ingest_path(path=corpus, store=store, chunk_size=400, chunk_overlap=60, source_tag="bio")
    hits = store.search_by_vector(
        store.embedder.embed_query("energy"), k=5, where={"source_tag": "bio"}
    )
    assert hits
    assert all(h.metadata["source_tag"] == "bio" for h in hits)

    none = store.search_by_vector(
        store.embedder.embed_query("energy"), k=5, where={"source_tag": "absent"}
    )
    assert none == []


def test_reset_clears_the_collection(populated_store):
    assert populated_store.count() > 0
    populated_store.reset()
    assert populated_store.count() == 0


def test_source_breakdown_counts_chunks(populated_store):
    breakdown = populated_store.source_breakdown()
    assert breakdown
    assert sum(breakdown.values()) == populated_store.count()


# --------------------------------------------------------------------------
# Similarity / ranking primitives
# --------------------------------------------------------------------------
def test_cosine_edge_cases():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([], [1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def _hit(chunk_id: str, score: float, vector: list[float]) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id, text=f"text {chunk_id}", metadata={}, score=score, embedding=vector
    )


# 3-D unit vectors against query [1, 0, 0]:
#   a      relevance 0.850
#   a_dup  relevance 0.840, but redundancy vs "a" is 0.999 (near-duplicate)
#   b      relevance 0.800, redundancy vs "a" only 0.680 (independent evidence)
# The third dimension is what lets "b" stay relevant while diverging from "a" --
# in 2-D any vector that differs from "a" also loses relevance in lockstep, which
# makes every MMR score collapse to an uninformative tie.
_MMR_QUERY = [1.0, 0.0, 0.0]
_MMR_HITS = [
    _hit("a", 0.85, [0.850, 0.5268, 0.0]),
    _hit("a_dup", 0.84, [0.8401, 0.5401, 0.05]),
    _hit("b", 0.80, [0.800, 0.0, 0.600]),
]


def test_mmr_drops_a_near_duplicate():
    selected = maximal_marginal_relevance(_MMR_QUERY, _MMR_HITS, k=2, lambda_mult=0.5)
    ids = [h.chunk_id for h in selected]
    assert ids[0] == "a", "the most relevant hit always seeds the selection"
    assert ids[1] == "b", "MMR should prefer the diverse hit over the near-duplicate"


def test_mmr_with_lambda_one_is_pure_relevance():
    # lambda=1.0 zeroes the redundancy penalty, so this must degrade to top-k.
    selected = maximal_marginal_relevance(_MMR_QUERY, _MMR_HITS, k=2, lambda_mult=1.0)
    assert [h.chunk_id for h in selected] == ["a", "a_dup"]


def test_mmr_with_lambda_zero_maximises_diversity():
    selected = maximal_marginal_relevance(_MMR_QUERY, _MMR_HITS, k=2, lambda_mult=0.0)
    assert [h.chunk_id for h in selected] == ["a", "b"]


def test_mmr_without_embeddings_falls_back_to_order():
    hits = [
        SearchHit(chunk_id="a", text="a", metadata={}, score=0.9),
        SearchHit(chunk_id="b", text="b", metadata={}, score=0.8),
    ]
    assert maximal_marginal_relevance([1.0], hits, k=1) == hits[:1]


def test_rrf_rewards_agreement_across_lists():
    # "shared" ranks 2nd in both lists; "top_a" ranks 1st in only one.
    list_a = [_hit("top_a", 0.9, [1.0]), _hit("shared", 0.5, [1.0])]
    list_b = [_hit("top_b", 0.9, [1.0]), _hit("shared", 0.5, [1.0])]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0].chunk_id == "shared"


def test_rrf_deduplicates():
    hits = [_hit("x", 0.5, [1.0])]
    fused = reciprocal_rank_fusion([hits, hits])
    assert len(fused) == 1


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------
def test_single_intent_question_is_not_split():
    plan = plan_queries("What is photosynthesis?")
    assert plan.strategy == "single"
    assert plan.sub_queries == ["What is photosynthesis?"]


def test_compound_question_is_decomposed():
    plan = plan_queries("What is photosynthesis and how does cellular respiration work?")
    assert plan.is_decomposed
    assert len(plan.sub_queries) >= 2
    assert plan.sub_queries[0].startswith("What is photosynthesis")


def test_multiple_question_marks_are_split():
    plan = plan_queries("What is ATP? Where is it produced?")
    assert plan.is_decomposed


def test_decomposition_respects_the_cap():
    question = "What is A and how is B and why is C and when is D and where is E?"
    plan = plan_queries(question, max_subqueries=2)
    assert len(plan.sub_queries) <= 2


def test_decompose_flag_disables_splitting():
    plan = plan_queries("What is A and how is B?", decompose=False)
    assert plan.strategy == "single"


def test_preamble_is_stripped():
    plan = plan_queries("Can you tell me what mitochondria do?")
    assert not plan.sub_queries[0].lower().startswith("can you")


def test_broaden_removes_interrogative_framing():
    assert broaden("What is the Calvin cycle?").lower().startswith("calvin cycle")


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------
def test_agent_returns_hits_and_timings(populated_store):
    agent = RetrievalAgent(Retriever(populated_store, top_k=3, candidate_k=10))
    outcome = agent.run("How does photosynthesis store energy?")
    assert outcome.hits
    assert outcome.embed_ms >= 0
    assert outcome.search_ms >= 0
    assert outcome.rounds in (1, 2)


def test_agent_retries_when_coverage_is_weak(populated_store, monkeypatch):
    from app.agents import orchestrator

    # Force every first-round result to look weak.
    monkeypatch.setattr(orchestrator, "WEAK_COVERAGE_SCORE", 1.5)
    agent = orchestrator.RetrievalAgent(Retriever(populated_store, top_k=3, candidate_k=10))
    outcome = agent.run("What is the Calvin cycle?")
    assert outcome.rounds == 2
    assert outcome.broadened_query
    assert "broadened" in " ".join(outcome.notes)


def test_agent_does_not_retry_on_strong_coverage(populated_store, monkeypatch):
    from app.agents import orchestrator

    monkeypatch.setattr(orchestrator, "WEAK_COVERAGE_SCORE", -1.0)
    agent = orchestrator.RetrievalAgent(Retriever(populated_store, top_k=3, candidate_k=10))
    outcome = agent.run("photosynthesis")
    assert outcome.rounds == 1


def test_agent_handles_empty_index(store):
    agent = RetrievalAgent(Retriever(store, top_k=3))
    outcome = agent.run("anything")
    assert outcome.hits == []


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------
def test_context_is_numbered_for_citation():
    hits = [
        SearchHit("a", "First passage body.", {"source": "a.txt"}, 0.9),
        SearchHit("b", "Second passage body.", {"source": "b.txt", "page": 3}, 0.8),
    ]
    context = build_context(hits)
    assert "[1] source: a.txt" in context
    assert "[2] source: b.txt, page 3" in context


def test_context_respects_the_char_budget():
    hits = [SearchHit(f"c{i}", "x" * 5000, {"source": "big.txt"}, 0.5) for i in range(10)]
    context = build_context(hits, max_chars=2000)
    assert len(context) < 2600  # budget plus one truncated block's header


# --------------------------------------------------------------------------
# Embedding cache
# --------------------------------------------------------------------------
def test_cache_serves_repeat_queries(embedder):
    cached = CachedEmbedder(embedder, max_size=8)
    first = cached.embed_query("repeat me")
    calls_after_first = embedder.call_count
    second = cached.embed_query("repeat me")
    assert first == second
    assert embedder.call_count == calls_after_first  # no re-embed
    assert cached.cache_stats()["hits"] == 1


def test_cache_is_case_insensitive(embedder):
    cached = CachedEmbedder(embedder, max_size=8)
    cached.embed_query("Hello World")
    cached.embed_query("hello world")
    assert cached.cache_stats()["hits"] == 1


def test_cache_evicts_least_recently_used(embedder):
    cached = CachedEmbedder(embedder, max_size=2)
    cached.embed_query("one")
    cached.embed_query("two")
    cached.embed_query("three")
    assert cached.cache_stats()["size"] == 2


def test_cache_does_not_intercept_document_embedding(embedder):
    cached = CachedEmbedder(embedder, max_size=8)
    cached.embed_documents(["same", "same"])
    assert embedder.call_count == 2
