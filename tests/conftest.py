"""Shared fixtures.

Tests use a deterministic hashing embedder rather than a real model: the suite
must run offline, in CI, in under a second, and must not be measuring a
Hugging Face download. It is a real bag-of-words vector space, so similarity
between related texts still behaves sensibly.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from app.retrieval.embeddings import Embedder

DIM = 64


class HashingEmbedder(Embedder):
    """Deterministic hashed bag-of-words vectors. No model, no network."""

    name = "hashing_test"

    def __init__(self, dim: int = DIM):
        self.model_name = "test/hashing"
        self._dim = dim
        self.call_count = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.call_count += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.call_count += 1
        return self._vector(text)

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(embedder, tmp_path):
    from app.retrieval.store import VectorStore

    return VectorStore(
        embedder=embedder,
        path=tmp_path / "chroma",
        collection_name="test_docs",
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A small mixed-format corpus covering every loader."""
    root = tmp_path / "corpus"
    root.mkdir()

    (root / "photosynthesis.md").write_text(
        "# Photosynthesis\n\n"
        "Photosynthesis is the process by which green plants convert light energy "
        "into chemical energy stored as glucose. It takes place in the chloroplasts "
        "of plant cells, where chlorophyll absorbs sunlight.\n\n"
        "The light-dependent reactions occur in the thylakoid membrane and produce "
        "ATP and NADPH. The Calvin cycle then uses these to fix carbon dioxide into "
        "sugar molecules in the stroma.\n",
        encoding="utf-8",
    )

    (root / "respiration.txt").write_text(
        "Cellular respiration is the metabolic process that releases energy from "
        "glucose. In eukaryotes it happens largely in the mitochondria. Glycolysis "
        "splits glucose into pyruvate, and the citric acid cycle then oxidises that "
        "pyruvate to carbon dioxide while generating electron carriers.\n",
        encoding="utf-8",
    )

    (root / "notes.csv").write_text(
        "id,topic,summary\n"
        "1,mitochondria,"
        '"The mitochondrion is the organelle responsible for producing most of the '
        'cell ATP through oxidative phosphorylation across the inner membrane."\n'
        "2,chloroplast,"
        '"The chloroplast houses the pigment chlorophyll and is the site where '
        'photosynthesis converts light into stored chemical energy."\n',
        encoding="utf-8",
    )

    (root / "records.jsonl").write_text(
        '{"title": "Enzymes", "body": "Enzymes are protein catalysts that lower the '
        'activation energy of biochemical reactions, letting cells run metabolism at '
        'temperatures that would otherwise be far too low."}\n'
        '{"title": "ATP", "body": "Adenosine triphosphate is the primary energy '
        'currency of the cell, releasing usable energy when its terminal phosphate '
        'bond is hydrolysed."}\n',
        encoding="utf-8",
    )

    return root


@pytest.fixture
def populated_store(store, corpus):
    from app.ingest.pipeline import ingest_path

    ingest_path(path=corpus, store=store, chunk_size=400, chunk_overlap=60)
    return store


@pytest.fixture
def client(monkeypatch, embedder, tmp_path, corpus):
    """TestClient wired to the hashing embedder and a temp vector store."""
    from fastapi.testclient import TestClient

    from app import main
    from app.agents.orchestrator import RetrievalAgent
    from app.config import Settings
    from app.generation.answerer import ExtractiveAnswerer
    from app.retrieval.retriever import Retriever
    from app.retrieval.store import VectorStore

    settings = Settings(chroma_path=tmp_path / "chroma", collection="test_api", top_k=3)

    def fake_build_state(_settings: Settings) -> main.AppState:
        vector_store = VectorStore(
            embedder=embedder,
            path=settings.chroma_path,
            collection_name=settings.collection,
        )
        retriever = Retriever(store=vector_store, top_k=3, candidate_k=10)
        return main.AppState(
            settings=settings,
            embedder=embedder,
            store=vector_store,
            retriever=retriever,
            agent=RetrievalAgent(retriever=retriever, max_subqueries=3),
            answerer=ExtractiveAnswerer(),
        )

    monkeypatch.setattr(main, "build_state", fake_build_state)
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    with TestClient(main.app) as test_client:
        test_client.corpus_path = str(corpus)  # type: ignore[attr-defined]
        yield test_client
