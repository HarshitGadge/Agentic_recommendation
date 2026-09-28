"""BM25 keyword index, hybrid (vector + keyword) retrieval, and fusion scoring."""

from __future__ import annotations

from app.agents.orchestrator import WEAK_COVERAGE_SCORE, RetrievalAgent
from app.ingest.chunker import Chunk
from app.retrieval.lexical import BM25Index, stem, tokenize
from app.retrieval.retriever import Retriever, reciprocal_rank_fusion
from app.retrieval.store import SearchHit, VectorStore


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------
def test_tokenize_lowercases_drops_stopwords_and_folds_plurals():
    assert tokenize("The Mutations of BRCA1 genes") == ["mutation", "brca1", "gene"]


def test_stem_folds_plurals_only():
    assert stem("studies") == "study"
    assert stem("classes") == "class"
    assert stem("virus") == "virus"  # -us is not a plural
    assert stem("gas") == "gas"  # too short to strip
    assert stem("binding") == "binding"


def test_hyphenated_terms_stay_whole():
    assert "covid-19" in tokenize("COVID-19 outcomes")


# --------------------------------------------------------------------------
# BM25 index
# --------------------------------------------------------------------------
def _index() -> BM25Index:
    index = BM25Index()
    index.add(
        ["a", "b", "c"],
        [
            "Aspirin lowers the risk of myocardial infarction in older adults.",
            "Statins reduce LDL cholesterol and cardiovascular events.",
            "Aspirin aspirin aspirin: dosing guidance for aspirin therapy.",
        ],
        [{"source_tag": "cardio"}, {"source_tag": "cardio"}, {"source_tag": "dosing"}],
    )
    return index


def test_bm25_ranks_documents_containing_the_term():
    ranked = [doc for doc, _ in _index().search("statins cholesterol", k=3)]
    assert ranked[0] == "b"
    assert "a" not in ranked  # no shared terms, so no score


def test_bm25_term_frequency_saturates():
    scores = dict(_index().search("aspirin", k=3))
    # Four mentions score higher than one, but far less than four times as high.
    assert scores["c"] > scores["a"]
    assert scores["c"] < 4 * scores["a"]


def test_bm25_where_filter():
    ranked = [doc for doc, _ in _index().search("aspirin", k=3, where={"source_tag": "cardio"})]
    assert ranked == ["a"]


def test_bm25_readding_an_id_replaces_it():
    index = _index()
    index.add(["b"], ["Completely different text about volcanoes."])
    assert len(index) == 3
    assert index.search("statins", k=3) == []
    assert index.search("volcanoes", k=3)[0][0] == "b"


def test_bm25_empty_query_or_index():
    assert BM25Index().search("anything", k=5) == []
    assert _index().search("the of and", k=5) == []  # stopwords only


# --------------------------------------------------------------------------
# Store integration
# --------------------------------------------------------------------------
def _chunks() -> list[Chunk]:
    texts = {
        "p1": "Photosynthesis converts light energy into glucose in the chloroplast.",
        "p2": "Cellular respiration releases energy from glucose in mitochondria.",
        "p3": "Part number XJ9000 is the replacement valve for the cooling pump.",
    }
    return [Chunk(id=k, text=v, metadata={"source": f"{k}.txt"}) for k, v in texts.items()]


def test_store_keeps_bm25_in_sync_on_upsert_and_reset(embedder, tmp_path):
    store = VectorStore(embedder, path=tmp_path / "db", collection_name="hybrid_test", lexical=True)
    store.upsert(_chunks())
    assert len(store.lexical) == store.count() == 3
    store.reset()
    assert len(store.lexical) == 0


def test_store_rebuilds_bm25_from_disk_at_startup(embedder, tmp_path):
    first = VectorStore(embedder, path=tmp_path / "db", collection_name="hybrid_test")
    first.upsert(_chunks())
    assert first.lexical is None

    restarted = VectorStore(
        embedder, path=tmp_path / "db", collection_name="hybrid_test", lexical=True
    )
    assert len(restarted.lexical) == 3
    hits = restarted.search_lexical("XJ9000 valve", k=2, include_embeddings=True)
    assert hits[0].chunk_id == "p3"
    assert hits[0].embedding is not None  # fetched from Chroma so MMR can use it


def test_search_lexical_without_index_is_empty(store):
    assert store.search_lexical("anything", k=3) == []


# --------------------------------------------------------------------------
# Hybrid retrieval
# --------------------------------------------------------------------------
def test_hybrid_retrieval_scores_are_cosine_and_fusion_is_recorded(embedder, tmp_path):
    store = VectorStore(embedder, path=tmp_path / "db", collection_name="hybrid_test", lexical=True)
    store.upsert(_chunks())
    retriever = Retriever(store, top_k=3, candidate_k=3, mmr_lambda=1.0, hybrid=True)

    hits, _, _ = retriever.retrieve(["replacement valve XJ9000"])
    assert hits[0].chunk_id == "p3"
    # Scores stay on the cosine scale (not tiny RRF values), so the agent's
    # weak-coverage check still means something.
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    assert hits[0].score > 0.3
    assert all(h.fused_score is not None for h in hits)


def test_hybrid_is_disabled_without_a_keyword_index(store):
    assert Retriever(store, hybrid=True).hybrid is False


def test_rrf_keeps_the_best_cosine_score_and_sets_fused_score():
    vec = [1.0, 0.0]
    list_a = [SearchHit("x", "x", {}, 0.82, vec), SearchHit("y", "y", {}, 0.40, vec)]
    list_b = [SearchHit("y", "y", {}, 0.61, vec)]
    fused = {h.chunk_id: h for h in reciprocal_rank_fusion([list_a, list_b])}
    assert fused["x"].score == 0.82
    assert fused["y"].score == 0.61  # best score across lists, not an RRF value
    assert fused["y"].fused_score > fused["x"].fused_score  # agreement across lists wins


def test_decomposed_question_does_not_trigger_a_spurious_retry(populated_store):
    """Regression: fusion used to overwrite scores with RRF values (~0.03), which are
    always below the weak-coverage threshold, so every multi-part question retried."""
    agent = RetrievalAgent(Retriever(populated_store, top_k=3, candidate_k=10), max_subqueries=3)
    outcome = agent.run("what is photosynthesis and how does cellular respiration release energy")
    assert outcome.plan.is_decomposed
    assert outcome.best_score >= WEAK_COVERAGE_SCORE
    assert outcome.rounds == 1


# --------------------------------------------------------------------------
# MMR: the vectorised implementation matches the straightforward definition
# --------------------------------------------------------------------------
def _reference_mmr(query, hits, k, lambda_mult):
    """The textbook greedy MMR loop, written for clarity rather than speed."""
    import math

    from app.retrieval.retriever import cosine

    if len(hits) <= k:
        return hits
    relevance = {h.chunk_id: cosine(query, h.embedding) for h in hits}
    selected = [max(hits, key=lambda h: relevance[h.chunk_id])]
    remaining = [h for h in hits if h is not selected[0]]
    while remaining and len(selected) < k:
        best, best_score = None, -math.inf
        for cand in remaining:
            redundancy = max(cosine(cand.embedding, s.embedding) for s in selected)
            score = lambda_mult * relevance[cand.chunk_id] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best, best_score = cand, score
        selected.append(best)
        remaining.remove(best)
    return selected


def test_vectorised_mmr_matches_the_reference_loop():
    import random

    from app.retrieval.retriever import maximal_marginal_relevance

    rng = random.Random(7)
    for _ in range(200):
        dim = rng.choice([3, 8, 32])
        hits = [
            SearchHit(str(i), "", {}, 0.0, [rng.gauss(0, 1) for _ in range(dim)])
            for i in range(rng.randint(1, 30))
        ]
        query = [rng.gauss(0, 1) for _ in range(dim)]
        k = rng.randint(1, 12)
        lam = rng.choice([0.0, 0.3, 0.5, 0.8, 1.0])
        expected = [h.chunk_id for h in _reference_mmr(query, hits, k, lam)]
        actual = [h.chunk_id for h in maximal_marginal_relevance(query, hits, k, lam)]
        assert actual == expected
