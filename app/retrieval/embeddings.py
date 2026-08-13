"""Pluggable Hugging Face embedding backends.

Two implementations of the same protocol, which is what makes the latency
benchmark in ``bench/benchmark.py`` an apples-to-apples comparison:

``sentence_transformers``
    The baseline. Runs the model through PyTorch in fp32. This is the
    conventional way to serve a Hugging Face embedding model.

``onnx_quantized``
    The optimisation. Runs the *same* model architecture through ONNX Runtime
    with int8-quantised weights (via ``fastembed``). Same vector space, same
    retrieval quality in practice, materially less CPU work per call.

A small LRU cache sits in front of query embedding, because repeated and
near-repeated queries are the norm in a served system and re-encoding them is
pure waste.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Common interface for every embedding backend."""

    name: str
    model_name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus text for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width. Needed to size the Chroma collection."""

    def warmup(self) -> None:
        """Pay one-time model-load and graph-init cost up front.

        Without this the first served request absorbs several hundred
        milliseconds of lazy initialisation, which would also poison the first
        benchmark sample.
        """
        self.embed_query("warmup")


class OnnxQuantizedEmbedder(Embedder):
    """int8-quantised ONNX Runtime backend (fastembed). The served default."""

    name = "onnx_quantized"

    def __init__(self, model_name: str, batch_size: int = 64, threads: int | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, threads=threads)
        self._dimension: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.embed(texts, batch_size=self.batch_size)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        # query_embed applies the model's query prefix where one is defined
        # (bge-style models expect it); it falls back to plain embed otherwise.
        try:
            vector = next(iter(self._model.query_embed(text)))
        except (AttributeError, NotImplementedError):
            vector = next(iter(self._model.embed([text])))
        return vector.tolist()

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension


class SentenceTransformerEmbedder(Embedder):
    """PyTorch fp32 backend. The benchmark baseline, not the served path."""

    name = "sentence_transformers"

    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The sentence_transformers backend requires the optional 'baseline' "
                "extra. Install it with:  pip install -e '.[baseline]'"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return vector.tolist()

    @property
    def dimension(self) -> int:
        # Renamed in sentence-transformers 5.x; support both spellings.
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        return int(getter())


class CachedEmbedder(Embedder):
    """Wraps any embedder with a bounded LRU cache over query embeddings.

    Document embedding is deliberately not cached: it runs once per chunk at
    ingest time, so a cache would only consume memory.
    """

    def __init__(self, inner: Embedder, max_size: int = 1024):
        self._inner = inner
        self._max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        key = text.strip().lower()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return cached
            self.misses += 1

        vector = self._inner.embed_query(text)

        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return vector

    def warmup(self) -> None:
        self._inner.warmup()

    def cache_stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._cache),
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


def build_embedder(
    backend: str,
    model_name: str,
    batch_size: int = 64,
    cache_size: int = 1024,
) -> Embedder:
    """Construct the configured backend, optionally wrapped in a query cache."""
    if backend == "onnx_quantized":
        embedder: Embedder = OnnxQuantizedEmbedder(model_name, batch_size=batch_size)
    elif backend == "sentence_transformers":
        embedder = SentenceTransformerEmbedder(model_name, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown embedding backend: {backend!r}")

    logger.info("Embedding backend=%s model=%s", backend, model_name)
    if cache_size > 0:
        return CachedEmbedder(embedder, max_size=cache_size)
    return embedder
