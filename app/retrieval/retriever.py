"""Retrieval strategy: candidate fetch, cross-query fusion, MMR diversification.

Plain top-k over a chunked corpus tends to return several near-identical
chunks from the same passage. These two steps fix that:

* **Reciprocal rank fusion** merges several ranked lists -- one per sub-query, and
  a vector list plus a BM25 keyword list when hybrid search is on -- without
  needing their scores to be on a comparable scale.
* **Maximal Marginal Relevance** then trades a little relevance for diversity
  so the context window holds distinct evidence rather than one idea repeated.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from app.retrieval.store import SearchHit, VectorStore


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Vectors are normalised, but don't assume it."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def maximal_marginal_relevance(
    query_vector: list[float],
    hits: list[SearchHit],
    k: int,
    lambda_mult: float = 0.5,
    relevance: dict[str, float] | None = None,
) -> list[SearchHit]:
    """Greedily pick ``k`` hits balancing query relevance against redundancy.

    ``lambda_mult`` = 1.0 is pure relevance (identical to top-k); 0.0 is pure
    diversity. Hits missing embeddings fall through to relevance order.

    ``relevance`` overrides the per-hit relevance term (default: cosine similarity
    to the query). After rank fusion it carries the fused score, so the keyword
    signal is not thrown away when MMR re-orders the candidates.
    """
    usable = [h for h in hits if h.embedding]
    if not usable or k <= 0:
        return hits[:k]
    if len(usable) <= k:
        return usable

    if relevance is None:
        relevance = {h.chunk_id: cosine(query_vector, h.embedding or []) for h in usable}
    else:
        relevance = {h.chunk_id: relevance.get(h.chunk_id, 0.0) for h in usable}
    rel = np.array([relevance[h.chunk_id] for h in usable])

    if lambda_mult >= 1.0:
        # No diversity term: this is plain top-k by relevance (stable for ties).
        order = np.argsort(-rel, kind="stable")[:k]
        return [usable[i] for i in order]

    # Pairwise cosine similarities in one matrix product, then an incremental
    # "max similarity to anything already selected" vector: O(n^2 d + k n)
    # instead of re-scoring every candidate against every pick.
    matrix = np.array([h.embedding for h in usable], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    similarity = matrix @ matrix.T

    chosen = [int(np.argmax(rel))]  # seed with the single most relevant hit
    redundancy = similarity[:, chosen[0]].copy()
    available = np.ones(len(usable), dtype=bool)
    available[chosen[0]] = False

    while available.any() and len(chosen) < k:
        scores = lambda_mult * rel - (1 - lambda_mult) * redundancy
        scores[~available] = -np.inf
        pick = int(np.argmax(scores))
        chosen.append(pick)
        available[pick] = False
        redundancy = np.maximum(redundancy, similarity[:, pick])

    return [usable[i] for i in chosen]


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]],
    k_constant: int = 60,
    weights: list[float] | None = None,
) -> list[SearchHit]:
    """Merge several ranked lists into one.

    RRF scores by rank position rather than raw similarity, so sub-queries that
    return systematically higher or lower scores don't dominate the merge.
    Each document's fused score is the sum of ``weight / (k_constant + rank)`` across
    the lists it appears in, which rewards agreement between lists. ``weights``
    (default all 1.0) lets a weaker list, such as keyword search, count for less.

    ``score`` keeps the best cosine similarity seen for the hit (so downstream
    coverage checks and citations stay on one scale); the fusion score goes in
    ``fused_score`` and decides the order.
    """
    fused: dict[str, float] = {}
    best_hit: dict[str, SearchHit] = {}

    weights = weights or [1.0] * len(ranked_lists)
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, hit in enumerate(ranked, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + weight / (k_constant + rank)
            previous = best_hit.get(hit.chunk_id)
            if previous is None or hit.score > previous.score:
                best_hit[hit.chunk_id] = hit

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [replace(best_hit[chunk_id], fused_score=round(score, 6)) for chunk_id, score in ordered]


class Retriever:
    """Fetch a wide candidate set, fuse across sub-queries (and keyword search), then diversify."""

    def __init__(
        self,
        store: VectorStore,
        top_k: int = 5,
        candidate_k: int = 20,
        mmr_lambda: float = 0.5,
        hybrid: bool = False,
        rrf_k: int = 60,
        keyword_weight: float = 1.0,
    ):
        self.store = store
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.mmr_lambda = mmr_lambda
        # Hybrid needs the store's BM25 index; quietly fall back to vector-only without it.
        self.hybrid = hybrid and store.lexical is not None
        self.rrf_k = rrf_k
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> tuple[list[SearchHit], float, float]:
        """Run every sub-query and return ``(hits, embed_ms, search_ms)``."""
        import time

        k = top_k or self.top_k
        where = {"source_tag": source_filter} if source_filter else None

        embed_ms = 0.0
        search_ms = 0.0
        ranked_lists: list[list[SearchHit]] = []
        list_weights: list[float] = []
        query_vectors: list[list[float]] = []

        for query in queries:
            t0 = time.perf_counter()
            vector = self.store.embedder.embed_query(query)
            embed_ms += (time.perf_counter() - t0) * 1000
            query_vectors.append(vector)

            t1 = time.perf_counter()
            hits = self.store.search_by_vector(
                vector, k=self.candidate_k, where=where, include_embeddings=True
            )
            if hits:
                ranked_lists.append(hits)
                list_weights.append(1.0)
            if self.hybrid:
                keyword_hits = self.store.search_lexical(
                    query, k=self.candidate_k, where=where, include_embeddings=True
                )
                # Report every hit's score as cosine similarity, whichever list found it.
                for hit in keyword_hits:
                    hit.score = round(cosine(vector, hit.embedding or []), 6)
                if keyword_hits:
                    ranked_lists.append(keyword_hits)
                    list_weights.append(self.keyword_weight)
            search_ms += (time.perf_counter() - t1) * 1000

        if not ranked_lists:
            return [], embed_ms, search_ms

        if len(ranked_lists) == 1:
            candidates, relevance = ranked_lists[0], None
        else:
            candidates = reciprocal_rank_fusion(
                ranked_lists, k_constant=self.rrf_k, weights=list_weights
            )
            # Rescale fused scores to [0, 1] so MMR's trade-off weight means the same thing.
            top = max(h.fused_score or 0.0 for h in candidates) or 1.0
            relevance = {h.chunk_id: (h.fused_score or 0.0) / top for h in candidates}

        # Diversify against the primary query's vector.
        selected = maximal_marginal_relevance(
            query_vectors[0], candidates, k=k, lambda_mult=self.mmr_lambda, relevance=relevance
        )
        return selected, embed_ms, search_ms
