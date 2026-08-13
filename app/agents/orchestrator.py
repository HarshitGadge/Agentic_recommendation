"""The retrieval agent loop.

One pass of naive top-k is not enough when a question is multi-part or phrased
unlike the corpus. This orchestrator runs a small, bounded control loop:

    plan -> retrieve -> assess coverage -> (broaden and retry once) -> assemble

The retry is what makes it adaptive rather than a fixed pipeline: the agent
inspects its own retrieval quality and decides whether another round is
warranted. The loop is capped at two rounds so worst-case latency stays
predictable -- an unbounded agent loop is a latency bug waiting to happen.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.agents.planner import QueryPlan, broaden, plan_queries
from app.retrieval.retriever import Retriever
from app.retrieval.store import SearchHit

logger = logging.getLogger(__name__)

# If the best hit scores below this, the first round probably missed.
WEAK_COVERAGE_SCORE = 0.35


@dataclass
class RetrievalOutcome:
    plan: QueryPlan
    hits: list[SearchHit]
    rounds: int = 1
    broadened_query: str | None = None
    plan_ms: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def best_score(self) -> float:
        return max((h.score for h in self.hits), default=0.0)


class RetrievalAgent:
    """Plans, retrieves, self-assesses, and assembles grounded context."""

    def __init__(self, retriever: Retriever, max_subqueries: int = 3):
        self.retriever = retriever
        self.max_subqueries = max_subqueries

    def run(
        self,
        question: str,
        top_k: int | None = None,
        decompose: bool = True,
        source_filter: str | None = None,
    ) -> RetrievalOutcome:
        t_plan = time.perf_counter()
        plan = plan_queries(question, max_subqueries=self.max_subqueries, decompose=decompose)
        plan_ms = (time.perf_counter() - t_plan) * 1000

        t_retrieve = time.perf_counter()
        hits, embed_ms, search_ms = self.retriever.retrieve(
            plan.sub_queries, top_k=top_k, source_filter=source_filter
        )
        rerank_ms = max((time.perf_counter() - t_retrieve) * 1000 - embed_ms - search_ms, 0.0)

        outcome = RetrievalOutcome(
            plan=plan,
            hits=hits,
            plan_ms=plan_ms,
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
        )

        if plan.is_decomposed:
            outcome.notes.append(f"decomposed into {len(plan.sub_queries)} sub-queries")

        # --- Self-assessment: was that good enough? ---
        if self._coverage_is_weak(hits):
            widened = broaden(question)
            if widened and widened.lower() != question.strip().lower():
                logger.info(
                    "Weak coverage (best=%.3f); retrying with %r", outcome.best_score, widened
                )
                t_retry = time.perf_counter()
                retry_hits, retry_embed_ms, retry_search_ms = self.retriever.retrieve(
                    [widened], top_k=top_k, source_filter=source_filter
                )
                outcome.rerank_ms += max(
                    (time.perf_counter() - t_retry) * 1000 - retry_embed_ms - retry_search_ms, 0.0
                )
                outcome.embed_ms += retry_embed_ms
                outcome.search_ms += retry_search_ms
                outcome.rounds = 2
                outcome.broadened_query = widened
                outcome.hits = self._merge(hits, retry_hits, top_k or self.retriever.top_k)
                outcome.notes.append("broadened query after weak first-round coverage")

        return outcome

    def _coverage_is_weak(self, hits: list[SearchHit]) -> bool:
        if not hits:
            return True
        return max(h.score for h in hits) < WEAK_COVERAGE_SCORE

    def _merge(self, first: list[SearchHit], second: list[SearchHit], k: int) -> list[SearchHit]:
        """Union both rounds, best score wins per chunk, truncate to k."""
        by_id: dict[str, SearchHit] = {}
        for hit in [*first, *second]:
            existing = by_id.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                by_id[hit.chunk_id] = hit
        return sorted(by_id.values(), key=lambda h: h.score, reverse=True)[:k]


def build_context(hits: list[SearchHit], max_chars: int = 12_000) -> str:
    """Render hits as a numbered context block the model can cite by index.

    Numbering is what makes citations verifiable: the model refers to [1], and
    the caller can map that straight back to a chunk ID and source file.
    """
    blocks: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        source = hit.source
        page = hit.metadata.get("page")
        row = hit.metadata.get("row")
        locator = f", page {page}" if page else (f", row {row}" if row else "")
        header = f"[{index}] source: {source}{locator}"
        body = hit.text.strip()

        block = f"{header}\n{body}"
        if used + len(block) > max_chars:
            remaining = max_chars - used - len(header) - 2
            if remaining < 200:
                break
            block = f"{header}\n{body[:remaining]}..."
            blocks.append(block)
            break
        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)
