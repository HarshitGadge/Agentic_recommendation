"""Retrieval-quality evaluation on a labelled benchmark (BEIR SciFact by default).

The latency benchmark (``bench/benchmark.py``) says how *fast* retrieval is. This
says how *good* it is: it runs the service's own chunker, embedder, vector store,
BM25 index, retriever and agent over a corpus with human relevance judgements,
and reports standard IR metrics per configuration.

    python -m eval.download_scifact                 # ~4.6 MB into eval/data/scifact
    python -m eval.retrieval_eval                   # uses the service's configured model

Metrics are computed at the document level: chunks are mapped back to the document
they came from, and a document counts once, at the rank of its best chunk.

    nDCG@10    ranking quality of the top 10, rewarding relevant documents near the top
    Recall@10  share of relevant documents found in the top 10
    MRR@10     1 / rank of the first relevant document (0 if none in the top 10)
    Recall@100 share found in the top 100; only for single ranked lists (vector, BM25, hybrid)

Every configuration returns ``--top-k`` chunks (default 10) from ``--candidate-k``
candidates (default 50). The service defaults are top_k=5 and candidate_k=20; the
evaluation uses more so that nDCG@10 is defined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.agents.orchestrator import RetrievalAgent
from app.config import get_settings
from app.ingest.chunker import Chunk, chunk_documents
from app.retrieval.embeddings import Embedder, build_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.store import SearchHit, VectorStore

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "scifact"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class Benchmark:
    name: str
    documents: list  # langchain Documents, metadata["source"] = document id
    queries: dict[str, str]  # query id -> text, only queries with judgements
    qrels: dict[str, set[str]]  # query id -> relevant document ids


def load_beir(folder: Path) -> Benchmark:
    """Load a BEIR-format dataset: corpus.jsonl, queries.jsonl, qrels/test.tsv."""
    from langchain_core.documents import Document

    def read_jsonl(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    corpus = read_jsonl(folder / "corpus.jsonl")
    all_queries = {str(q["_id"]): q["text"] for q in read_jsonl(folder / "queries.jsonl")}

    qrels: dict[str, set[str]] = {}
    with (folder / "qrels" / "test.tsv").open(encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            query_id, doc_id, score = line.strip().split("\t")
            if int(score) > 0:
                qrels.setdefault(query_id, set()).add(doc_id)

    documents = [
        Document(
            page_content=f"{d.get('title', '')}\n\n{d.get('text', '')}".strip(),
            metadata={"source": str(d["_id"])},
        )
        for d in corpus
    ]
    queries = {qid: all_queries[qid] for qid in qrels}
    return Benchmark(folder.name, documents, queries, qrels)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def to_document_ranking(hits: list[SearchHit]) -> list[str]:
    ranking: list[str] = []
    for hit in hits:
        doc_id = str(hit.metadata.get("source"))
        if doc_id not in ranking:
            ranking.append(doc_id)
    return ranking


def ndcg_at_k(ranking: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranking[:k]) & relevant) / len(relevant)


def mrr_at_k(ranking: list[str], relevant: set[str], k: int = 10) -> float:
    return next((1 / (i + 1) for i, d in enumerate(ranking[:k]) if d in relevant), 0.0)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


# --------------------------------------------------------------------------
# Embedding with a disk cache (embedding the corpus is the slow part)
# --------------------------------------------------------------------------
class PrecomputedEmbedder(Embedder):
    """Serves cached document vectors; delegates queries to the real model."""

    def __init__(self, inner: Embedder, vectors: dict[str, list[float]]):
        self._inner = inner
        self._vectors = vectors
        self.name = inner.name
        self.model_name = inner.model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    @property
    def dimension(self) -> int:
        return self._inner.dimension


def embed_corpus(
    embedder: Embedder, chunks: list[Chunk], cache_dir: Path, tag: str
) -> dict[str, list[float]]:
    import numpy as np

    key = hashlib.sha256("".join(c.id for c in chunks).encode()).hexdigest()[:12]
    cache = cache_dir / f"{tag}-{key}.npy"
    if cache.exists():
        matrix = np.load(cache)
        print(f"  loaded cached embeddings: {cache.name}", flush=True)
    else:
        print(f"  embedding {len(chunks):,} chunks (cached to {cache.name}) ...", flush=True)
        started = time.perf_counter()
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), 512):
            vectors.extend(embedder.embed_documents([c.text for c in chunks[i : i + 512]]))
            print(f"    {min(i + 512, len(chunks)):,}/{len(chunks):,}", flush=True)
        matrix = np.asarray(vectors, dtype=np.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, matrix)
        print(f"  done in {time.perf_counter() - started:.0f}s", flush=True)
    return {c.text: matrix[i].tolist() for i, c in enumerate(chunks)}


def build_store(embedder: Embedder, chunks: list[Chunk], workdir: Path, name: str) -> VectorStore:
    store = VectorStore(embedder, path=workdir / name, collection_name=f"eval-{name}", lexical=True)
    for i in range(0, len(chunks), 1000):
        store.upsert(chunks[i : i + 1000])
    return store


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(
    name: str, bench: Benchmark, search: Callable[[str], list[SearchHit]], deep: bool
) -> dict:
    ndcg, rec10, mrr, rec100, latency = [], [], [], [], []
    for query_id, text in bench.queries.items():
        started = time.perf_counter()
        hits = search(text)
        latency.append((time.perf_counter() - started) * 1000)
        ranking = to_document_ranking(hits)
        relevant = bench.qrels[query_id]
        ndcg.append(ndcg_at_k(ranking, relevant))
        rec10.append(recall_at_k(ranking, relevant, 10))
        mrr.append(mrr_at_k(ranking, relevant))
        if deep:
            rec100.append(recall_at_k(ranking, relevant, 100))
    row = {
        "config": name,
        "ndcg@10": round(statistics.mean(ndcg), 4),
        "recall@10": round(statistics.mean(rec10), 4),
        "mrr@10": round(statistics.mean(mrr), 4),
        "recall@100": round(statistics.mean(rec100), 4) if deep else None,
        "latency_p50_ms": round(percentile(latency, 50), 2),
        "latency_p95_ms": round(percentile(latency, 95), 2),
        "per_query_ndcg": ndcg,
    }
    r100 = f"{row['recall@100']:.4f}" if deep else "   -  "
    print(
        f"  {name:<44} nDCG@10 {row['ndcg@10']:.4f}  R@10 {row['recall@10']:.4f}  "
        f"MRR@10 {row['mrr@10']:.4f}  R@100 {r100}  p50 {row['latency_p50_ms']:.1f} ms",
        flush=True,
    )
    return row


def paired_bootstrap(a: list[float], b: list[float], rounds: int = 2000, seed: int = 0) -> tuple:
    """95% CI for mean(b - a) by resampling queries."""
    import random

    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return statistics.mean(diffs), means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def run_configs(store: VectorStore, bench: Benchmark, top_k: int, candidate_k: int) -> list[dict]:
    def retriever(**kwargs) -> Retriever:
        options = {"top_k": top_k, "candidate_k": candidate_k, "mmr_lambda": 1.0} | kwargs
        return Retriever(store, **options)

    deep = {"top_k": 100, "candidate_k": 100}
    hybrid_equal = retriever(hybrid=True, keyword_weight=1.0, **deep)
    hybrid_half = retriever(hybrid=True, keyword_weight=0.5, **deep)
    previous = retriever(mmr_lambda=0.5)
    current = retriever()
    hybrid_top = retriever(hybrid=True, keyword_weight=0.5)

    return [
        # Single ranked lists, 100 deep: ranking quality plus recall@100.
        evaluate("vector", bench, lambda q: store.search(q, 100), deep=True),
        evaluate("BM25 keyword", bench, lambda q: store.search_lexical(q, 100), deep=True),
        evaluate(
            "hybrid, BM25 weight 1.0", bench, lambda q: hybrid_equal.retrieve([q])[0], deep=True
        ),
        evaluate(
            "hybrid, BM25 weight 0.5", bench, lambda q: hybrid_half.retrieve([q])[0], deep=True
        ),
        # The service's retrieval path (agent: plan, retrieve, fuse, diversify, retry),
        # returning top_k chunks.
        evaluate(
            "service, previous defaults (MMR 0.5)",
            bench,
            lambda q: RetrievalAgent(previous).run(q, top_k=top_k).hits,
            deep=False,
        ),
        evaluate(
            "service, new defaults (no MMR)",
            bench,
            lambda q: RetrievalAgent(current).run(q, top_k=top_k).hits,
            deep=False,
        ),
        evaluate(
            "service, hybrid on (BM25 weight 0.5)",
            bench,
            lambda q: RetrievalAgent(hybrid_top).run(q, top_k=top_k).hits,
            deep=False,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="BEIR-format dataset folder"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--cache", type=Path, default=Path(tempfile.gettempdir()) / "rag-eval-cache"
    )
    parser.add_argument(
        "--compare-model-path",
        type=Path,
        default=None,
        help="Second local model directory (e.g. fp32 ONNX) to compare vector quality and latency",
    )
    parser.add_argument(
        "--compare-model-name",
        default=None,
        help="fastembed model name for --compare-model-path (see README)",
    )
    args = parser.parse_args(argv)

    if not (args.data / "corpus.jsonl").exists():
        print(f"No dataset at {args.data}. Run:  python -m eval.download_scifact", file=sys.stderr)
        return 1

    settings = get_settings()
    bench = load_beir(args.data)
    chunks = chunk_documents(bench.documents, settings.chunk_size, settings.chunk_overlap)
    print(
        f"{bench.name}: {len(bench.documents):,} documents -> {len(chunks):,} chunks, "
        f"{len(bench.queries)} test queries",
        flush=True,
    )

    base = build_embedder(
        settings.embed_backend,
        settings.embed_model,
        cache_size=0,
        model_path=settings.embed_model_path or None,
    )
    base.warmup()
    vectors = embed_corpus(base, chunks, args.cache, f"{bench.name}-{base.name}")

    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp:
        store = build_store(PrecomputedEmbedder(base, vectors), chunks, Path(tmp), "main")
        print(
            f"\nConfigurations (top_k={args.top_k}, candidate_k={args.candidate_k}), "
            f"model {base.model_name} via {base.name}:",
            flush=True,
        )
        rows = run_configs(store, bench, args.top_k, args.candidate_k)

        by_name = {r["config"]: r for r in rows}
        comparisons = {}
        for label, a, b in [
            (
                "new vs previous service defaults",
                "service, previous defaults (MMR 0.5)",
                "service, new defaults (no MMR)",
            ),
            ("hybrid (weight 0.5) vs vector", "vector", "hybrid, BM25 weight 0.5"),
            (
                "hybrid on vs off in the service",
                "service, new defaults (no MMR)",
                "service, hybrid on (BM25 weight 0.5)",
            ),
        ]:
            mean, lo, hi = paired_bootstrap(
                by_name[a]["per_query_ndcg"], by_name[b]["per_query_ndcg"]
            )
            comparisons[label] = {
                "metric": "ndcg@10",
                "diff": round(mean, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
            }
            print(f"  {label}: nDCG@10 {mean:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})", flush=True)

        precision = None
        if args.compare_model_path:
            precision = compare_models(args, bench, chunks, base, store, Path(tmp))

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "dataset": bench.name,
        "documents": len(bench.documents),
        "chunks": len(chunks),
        "queries": len(bench.queries),
        "model": base.model_name,
        "backend": base.name,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "configs": [{k: v for k, v in r.items() if k != "per_query_ndcg"} for r in rows],
        "comparisons": comparisons,
        "model_comparison": precision,
    }
    out = RESULTS_DIR / f"{bench.name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


def compare_models(args, bench, chunks, base, base_store, workdir: Path) -> dict:
    """Vector-only quality and query-embedding latency: served model vs a second model."""
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    from app.retrieval.embeddings import OnnxEmbedder

    name = args.compare_model_name or "local/bge-small-en-v1.5-fp32"
    if name.startswith("local/"):
        TextEmbedding.add_custom_model(
            model=name,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf="BAAI/bge-small-en-v1.5"),
            dim=384,
            model_file="onnx/model.onnx",
        )
    other = OnnxEmbedder(name, model_path=str(args.compare_model_path))
    other.warmup()
    other_vectors = embed_corpus(
        other, chunks, args.cache, f"{bench.name}-{name.replace('/', '_')}"
    )
    other_store = build_store(PrecomputedEmbedder(other, other_vectors), chunks, workdir, "compare")

    print(f"\nModel comparison (vector top-k): {base.model_name} [served] vs {name}", flush=True)
    results = {}
    for label, embedder, store in [("served", base, base_store), ("compare", other, other_store)]:
        row = evaluate(f"vector top-k [{label}]", bench, lambda q, s=store: s.search(q, 100), True)
        samples = []
        for text in list(bench.queries.values())[:200]:
            started = time.perf_counter()
            embedder.embed_query(text)
            samples.append((time.perf_counter() - started) * 1000)
        row["query_embed_p50_ms"] = round(percentile(samples, 50), 2)
        row["query_embed_p95_ms"] = round(percentile(samples, 95), 2)
        p50, p95 = row["query_embed_p50_ms"], row["query_embed_p95_ms"]
        print(f"    query embedding p50 {p50} ms, p95 {p95} ms")
        results[label] = {k: v for k, v in row.items() if k != "per_query_ndcg"}
    return results


if __name__ == "__main__":
    raise SystemExit(main())
