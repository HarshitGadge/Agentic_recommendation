"""Query planning: turn one user question into the search queries to run.

This is deliberately rule-based rather than LLM-driven. Decomposition sits on
the hot path of every request, so paying a model round-trip here would cost
more latency than the extra recall is worth -- and it keeps the retrieval layer
working with no API key configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Conjunctions that usually join two genuinely separate asks.
_SPLIT_PATTERN = re.compile(
    r"\s+(?:and also|as well as|and then|;|\band\b(?=\s+(?:what|how|why|when|where|who|which"
    r"|is|are|does|do|did|can)\b))\s*",
    re.IGNORECASE,
)

# Leading filler that adds no retrieval signal.
_PREAMBLE = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|i(?:'d| would) like to know\s+|tell me\s+|explain\s+)+",
    re.IGNORECASE,
)

_QUESTION_WORDS = ("what", "how", "why", "when", "where", "who", "which", "compare", "list")


@dataclass
class QueryPlan:
    original: str
    sub_queries: list[str]
    strategy: str  # "single" | "decomposed"

    @property
    def is_decomposed(self) -> bool:
        return self.strategy == "decomposed"


def plan_queries(question: str, max_subqueries: int = 3, decompose: bool = True) -> QueryPlan:
    """Produce the ordered list of queries to search with.

    The original question is always first, so single-intent questions behave
    exactly as they would without a planner.
    """
    cleaned = _clean(question)
    if not decompose or max_subqueries <= 1:
        return QueryPlan(original=question, sub_queries=[cleaned], strategy="single")

    parts = _split(cleaned)
    if len(parts) <= 1:
        return QueryPlan(original=question, sub_queries=[cleaned], strategy="single")

    sub_queries = [cleaned]
    for raw_part in parts:
        part = _clean(raw_part)
        if len(part) < 8:
            continue
        if part.lower() == cleaned.lower():
            continue
        if part not in sub_queries:
            sub_queries.append(part)
        if len(sub_queries) >= max_subqueries:
            break

    if len(sub_queries) == 1:
        return QueryPlan(original=question, sub_queries=sub_queries, strategy="single")
    return QueryPlan(original=question, sub_queries=sub_queries, strategy="decomposed")


def broaden(question: str) -> str:
    """Widen a query for a follow-up pass when the first round retrieved little.

    Drops interrogative framing and keeps the content words, which matches
    corpus prose more closely than a fully-formed question does.
    """
    text = _clean(question).rstrip("?").strip()
    words = text.split()
    if words and words[0].lower() in _QUESTION_WORDS:
        words = words[1:]
    # Drop leading auxiliaries left behind by removing the question word.
    _AUXILIARIES = {"is", "are", "was", "were", "do", "does", "did", "the", "a", "an"}
    while words and words[0].lower() in _AUXILIARIES:
        words = words[1:]
    broadened = " ".join(words).strip()
    return broadened or text


def _clean(text: str) -> str:
    text = text.strip()
    text = _PREAMBLE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _split(text: str) -> list[str]:
    """Split on conjunctions and multi-question punctuation."""
    # Multiple explicit questions take priority over conjunction splitting.
    if text.count("?") > 1:
        parts = [p.strip() + "?" for p in text.split("?") if p.strip()]
        if len(parts) > 1:
            return parts
    return [p for p in _SPLIT_PATTERN.split(text) if p and p.strip()]
