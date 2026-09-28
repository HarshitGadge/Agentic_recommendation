"""In-memory BM25 keyword index, used alongside vector search for hybrid retrieval.

Dense embeddings are good at paraphrase ("heart attack" ~ "myocardial infarction") but
weak on exact tokens a user types verbatim: gene names, drug names, error codes, IDs.
BM25 is the opposite. Fusing the two ranked lists with reciprocal rank fusion gets
most of the benefit of each; ``eval/retrieval_eval.py`` measures the difference.

The index lives in memory and is rebuilt from the vector store at startup. With
posting lists in plain dicts it handles tens of thousands of chunks comfortably;
beyond a few hundred thousand, a dedicated engine (Elasticsearch, OpenSearch, or
a database full-text index) is the better home for it.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# A short English stopword list. Removing these keeps posting lists small and stops
# very common words from contributing noise to the score.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "he",
        "her", "his", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our", "she",
        "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
        "to", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will",
        "with",
        "would", "you", "your", "do", "does", "did", "not", "no", "can", "could", "should", "may",
        "might", "also", "been", "being", "about", "over", "under", "between", "after", "before",
    }
)


# Plural folding, so "genes" matches "gene" and "studies" matches "study". This is the
# plural step of the Porter stemmer only: a full stemmer conflates more word forms but
# also more unrelated words. On the SciFact benchmark it adds 0.6 points of nDCG@10
# and 1.9 points of recall@100 over no stemming (eval/retrieval_eval.py).
def stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (hyphenated terms kept whole), no stopwords, plurals folded."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """Okapi BM25 over chunk texts, keyed by chunk ID.

    ``k1`` controls term-frequency saturation and ``b`` document-length
    normalisation. The defaults (0.9, 0.4) are the Anserini/Pyserini defaults
    that BEIR-style evaluations commonly use.
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self._lock = threading.Lock()
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)  # term -> {doc: tf}
        self._doc_len: dict[str, int] = {}
        self._doc_terms: dict[str, Counter] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._text: dict[str, str] = {}
        self._total_len = 0

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]] | None = None):
        """Add or replace documents. Re-adding an ID replaces it (upsert semantics)."""
        metadatas = metadatas or [{} for _ in ids]
        with self._lock:
            for doc_id, text, metadata in zip(ids, texts, metadatas, strict=True):
                if doc_id in self._doc_len:
                    self._remove(doc_id)
                terms = Counter(tokenize(text))
                for term, tf in terms.items():
                    self._postings[term][doc_id] = tf
                length = sum(terms.values())
                self._doc_len[doc_id] = length
                self._doc_terms[doc_id] = terms
                self._metadata[doc_id] = dict(metadata or {})
                self._text[doc_id] = text
                self._total_len += length

    def _remove(self, doc_id: str) -> None:
        for term in self._doc_terms.pop(doc_id, {}):
            postings = self._postings.get(term)
            if postings is not None:
                postings.pop(doc_id, None)
                if not postings:
                    del self._postings[term]
        self._total_len -= self._doc_len.pop(doc_id, 0)
        self._metadata.pop(doc_id, None)
        self._text.pop(doc_id, None)

    def clear(self) -> None:
        with self._lock:
            self._postings.clear()
            self._doc_len.clear()
            self._doc_terms.clear()
            self._metadata.clear()
            self._text.clear()
            self._total_len = 0

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._doc_len)

    def search(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(doc_id, score)`` pairs, best first.

        ``where`` supports the same simple equality filter the service uses for
        vector search (for example ``{"source_tag": "my-dataset"}``).
        """
        terms = tokenize(query)
        n_docs = len(self._doc_len)
        if not terms or n_docs == 0 or k <= 0:
            return []
        avg_len = self._total_len / n_docs

        scores: dict[str, float] = defaultdict(float)
        with self._lock:
            for term in set(terms):
                postings = self._postings.get(term)
                if not postings:
                    continue
                df = len(postings)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                for doc_id, tf in postings.items():
                    norm = self.k1 * (1 - self.b + self.b * self._doc_len[doc_id] / avg_len)
                    scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)

            if where:
                scores = {
                    d: s
                    for d, s in scores.items()
                    if all(
                        self._metadata.get(d, {}).get(key) == value for key, value in where.items()
                    )
                }

        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def document(self, doc_id: str) -> tuple[str, dict[str, Any]]:
        return self._text.get(doc_id, ""), self._metadata.get(doc_id, {})
