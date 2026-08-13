"""Retrieval strategy: candidate fetch, cross-query fusion, MMR diversification.

Plain top-k over a chunked corpus tends to return several near-identical
chunks from the same passage. These two steps fix that:

* **Reciprocal rank fusion** merges the result lists of several sub-queries
  without needing their scores to be on a comparable scale.
* **Maximal Marginal Relevance** then trades a little relevance for diversity
  so the context window holds distinct evidence rather than one idea repeated.
"""

from __future__ import annotations

import math
from dataclasses import replace

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
) -> list[SearchHit]:
    """Greedily pick ``k`` hits balancing query relevance against redundancy.

    ``lambda_mult`` = 1.0 is pure relevance (identical to top-k); 0.0 is pure
    diversity. Hits missing embeddings fall through to relevance order.
    """
    usable = [h for h in hits if h.embedding]
    if not usable or k <= 0:
        return hits[:k]
    if len(usable) <= k:
        return usable

    relevance = {h.chunk_id: cosine(query_vector, h.embedding or []) for h in usable}

    selected: list[SearchHit] = []
    remaining = list(usable)

    # Seed with the single most relevant hit.
    first = max(remaining, key=lambda h: relevance[h.chunk_id])
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < k:
        best_hit = None
        best_score = -math.inf
        for candidate in remaining:
            redundancy = max(
                cosine(candidate.embedding or [], chosen.embedding or []) for chosen in selected
            )
            score = lambda_mult * relevance[candidate.chunk_id] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_hit = candidate
        if best_hit is None:
            break
        selected.append(best_hit)
        remaining.remove(best_hit)

    return selected


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]], k_constant: int = 60
) -> list[SearchHit]:
    """Merge several ranked lists into one.

    RRF scores by rank position rather than raw similarity, so sub-queries that
    return systematically higher or lower scores don't dominate the merge.
    Each document's fused score is the sum of ``1 / (k_constant + rank)`` across
    the lists it appears in, which rewards agreement between sub-queries.
    """
    fused: dict[str, float] = {}
    best_hit: dict[str, SearchHit] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k_constant + rank)
            previous = best_hit.get(hit.chunk_id)
            if previous is None or hit.score > previous.score:
                best_hit[hit.chunk_id] = hit

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [replace(best_hit[chunk_id], score=round(score, 6)) for chunk_id, score in ordered]


class Retriever:
    """Fetch a wide candidate set, fuse across sub-queries, then diversify."""

    def __init__(
        self,
        store: VectorStore,
        top_k: int = 5,
        candidate_k: int = 20,
        mmr_lambda: float = 0.5,
    ):
        self.store = store
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.mmr_lambda = mmr_lambda

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
            search_ms += (time.perf_counter() - t1) * 1000
            if hits:
                ranked_lists.append(hits)

        if not ranked_lists:
            return [], embed_ms, search_ms

        candidates = (
            ranked_lists[0] if len(ranked_lists) == 1 else reciprocal_rank_fusion(ranked_lists)
        )

        # Diversify against the primary query's vector.
        selected = maximal_marginal_relevance(
            query_vectors[0], candidates, k=k, lambda_mult=self.mmr_lambda
        )
        return selected, embed_ms, search_ms
