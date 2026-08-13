"""ChromaDB persistent vector store with tuned HNSW parameters.

Embeddings are computed by our own ``Embedder`` rather than by a Chroma
embedding function, so the backend stays swappable and the benchmark can time
the embedding stage in isolation.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.ingest.chunker import Chunk
from app.retrieval.embeddings import Embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity in [0, 1]; higher is better
    embedding: list[float] | None = None

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))


class VectorStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(
        self,
        embedder: Embedder,
        path: str | Path = "./data/chroma",
        collection_name: str = "documents",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ):
        self.embedder = embedder
        self.collection_name = collection_name
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

        # Cosine space matches the normalised vectors both backends produce.
        # M and construction_ef trade index build time for recall; search_ef
        # trades query latency for recall and is the main runtime knob.
        self._index_metadata = {
            "hnsw:space": "cosine",
            "hnsw:M": hnsw_m,
            "hnsw:construction_ef": hnsw_ef_construction,
            "hnsw:search_ef": hnsw_ef_search,
        }
        self._collection = self._get_or_create()

    def _get_or_create(self):
        try:
            return self._client.get_or_create_collection(
                name=self.collection_name, metadata=dict(self._index_metadata)
            )
        except Exception as exc:
            # Older/newer Chroma builds reject unknown hnsw:* keys. Falling back
            # to defaults is better than refusing to start.
            logger.warning("HNSW tuning rejected by Chroma (%s); using defaults", exc)
            return self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk]) -> int:
        """Embed and write chunks. Repeated IDs overwrite rather than duplicate."""
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata or {"source": "unknown"} for c in chunks],
        )
        return len(chunks)

    def reset(self) -> None:
        """Drop and recreate the collection. Used by tests and re-index flows."""
        with contextlib.suppress(Exception):  # collection may not exist yet
            self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def search_by_vector(
        self,
        vector: list[float],
        k: int,
        where: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[SearchHit]:
        """Nearest-neighbour search against a precomputed query vector."""
        if self.count() == 0:
            return []

        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(k, max(self.count(), 1)),
            where=where or None,
            include=include,
        )
        return self._to_hits(result, include_embeddings=include_embeddings)

    def search(
        self,
        query: str,
        k: int,
        where: dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[SearchHit]:
        vector = self.embedder.embed_query(query)
        return self.search_by_vector(vector, k, where=where, include_embeddings=include_embeddings)

    def _to_hits(self, result: dict[str, Any], include_embeddings: bool) -> list[SearchHit]:
        ids = (result.get("ids") or [[]])[0]
        if not ids:
            return []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        embeddings = (result.get("embeddings") or [[]])[0] if include_embeddings else None

        hits: list[SearchHit] = []
        for i, chunk_id in enumerate(ids):
            # Chroma returns cosine *distance* (1 - similarity) for this space.
            distance = float(distances[i]) if i < len(distances) else 1.0
            embedding = None
            if embeddings is not None and i < len(embeddings) and embeddings[i] is not None:
                embedding = list(embeddings[i])
            hits.append(
                SearchHit(
                    chunk_id=str(chunk_id),
                    text=documents[i] if i < len(documents) else "",
                    metadata=dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {},
                    score=round(1.0 - distance, 6),
                    embedding=embedding,
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def source_breakdown(self, limit: int = 10_000) -> dict[str, int]:
        """Chunk counts per source file. Sampled -- not exact on huge corpora."""
        total = self.count()
        if total == 0:
            return {}
        try:
            payload = self._collection.get(limit=min(limit, total), include=["metadatas"])
        except Exception as exc:
            logger.warning("Could not read metadata for stats: %s", exc)
            return {}

        counts: dict[str, int] = {}
        for metadata in payload.get("metadatas") or []:
            entry = metadata or {}
            key = str(entry.get("source_tag") or entry.get("source", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
