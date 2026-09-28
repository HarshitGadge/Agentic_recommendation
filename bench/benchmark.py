"""Latency benchmark: PyTorch fp32 baseline vs the graph-optimised ONNX Runtime path.

This exists so the performance claim about this project is a *measurement*
rather than an assertion. It builds a separate index per backend over the same
corpus, replays the same queries against each, and reports p50/p95/p99 for the
embedding stage, the vector-search stage, and their total -- plus the reduction
between baseline and optimised.

    python -m bench.benchmark --corpus ./data/docs --runs 30

If ``sentence-transformers`` is not installed the baseline is skipped and only
the optimised path is profiled. Install it with:  pip install -e '.[baseline]'

Numbers are hardware-specific. Report the ones your machine produces; do not
copy someone else's.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.ingest.chunker import chunk_documents
from app.ingest.loaders import discover_files, iter_documents
from app.obs.logging_conf import configure_logging
from app.obs.metrics import percentile
from app.retrieval.embeddings import build_embedder
from app.retrieval.store import VectorStore

DEFAULT_RUNS = 30
DEFAULT_WARMUP = 5

# Below this, HNSW query time is dominated by fixed overhead rather than by the
# index, so the search-stage comparison carries no signal.
MIN_MEANINGFUL_CHUNKS = 1000


@dataclass
class StageStats:
    p50: float
    p95: float
    p99: float
    mean: float
    stdev: float
    samples: int

    @classmethod
    def from_samples(cls, values: list[float]) -> StageStats:
        ordered = sorted(values)
        return cls(
            p50=round(percentile(ordered, 50), 3),
            p95=round(percentile(ordered, 95), 3),
            p99=round(percentile(ordered, 99), 3),
            mean=round(statistics.fmean(ordered), 3),
            stdev=round(statistics.stdev(ordered), 3) if len(ordered) > 1 else 0.0,
            samples=len(ordered),
        )


@dataclass
class BackendResult:
    backend: str
    model: str
    dimension: int
    chunks_indexed: int
    index_build_s: float
    embed: StageStats
    search: StageStats
    total: StageStats
    cache_hit_p50: float | None = None
    notes: list[str] = field(default_factory=list)


def load_corpus_chunks(corpus: Path, chunk_size: int, chunk_overlap: int) -> list:
    files = discover_files(corpus, recursive=True)
    if not files:
        raise SystemExit(
            f"No loadable files under {corpus}. Supported: .txt .md .pdf .csv .tsv .json .jsonl"
        )
    documents = []
    for _path, docs in iter_documents(files):
        documents.extend(docs)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise SystemExit(f"{len(files)} file(s) found under {corpus} but they produced no chunks.")
    return chunks


def derive_queries(chunks: list, count: int, seed: int = 7) -> list[str]:
    """Sample realistic queries from the corpus itself.

    Using corpus-derived text keeps the benchmark honest about the query
    distribution: these are strings that genuinely have matches in the index.
    """
    rng = random.Random(seed)
    pool = [c.text for c in chunks if len(c.text) > 80]
    if not pool:
        pool = [c.text for c in chunks]
    picks = rng.sample(pool, min(count, len(pool)))

    queries = []
    for text in picks:
        # First sentence, trimmed -- approximates a natural-language question
        # better than a whole chunk does.
        sentence = text.split(". ")[0].strip()
        queries.append(sentence[:180] if len(sentence) > 20 else text[:180])
    return queries


def benchmark_backend(
    backend: str,
    model: str,
    chunks: list,
    queries: list[str],
    runs: int,
    warmup: int,
    ef_search: int,
    cache_size: int,
    store_root: Path,
) -> BackendResult:
    print(f"\n=== {backend} / {model} ===", flush=True)

    # Cache disabled for the main measurement: a cache would measure the cache,
    # not the model. Its effect is measured separately at the end.
    embedder = build_embedder(backend=backend, model_name=model, cache_size=0)
    t0 = time.perf_counter()
    embedder.warmup()
    print(f"  model load + warmup: {time.perf_counter() - t0:.2f}s", flush=True)

    store = VectorStore(
        embedder=embedder,
        path=store_root / backend,
        collection_name=f"bench_{backend}",
        hnsw_ef_search=ef_search,
    )
    store.reset()

    print(f"  indexing {len(chunks)} chunks...", flush=True)
    t_index = time.perf_counter()
    for start in range(0, len(chunks), 256):
        store.upsert(chunks[start : start + 256])
    index_build_s = time.perf_counter() - t_index
    print(f"  indexed in {index_build_s:.2f}s", flush=True)

    # Warmup rounds are discarded: the first few queries pay for lazily
    # allocated buffers and a cold HNSW page cache.
    for i in range(warmup):
        store.search(queries[i % len(queries)], k=5)

    embed_samples: list[float] = []
    search_samples: list[float] = []
    total_samples: list[float] = []

    print(f"  timing {runs} queries...", flush=True)
    for i in range(runs):
        query = queries[i % len(queries)]

        t_embed = time.perf_counter()
        vector = embedder.embed_query(query)
        embed_ms = (time.perf_counter() - t_embed) * 1000

        t_search = time.perf_counter()
        store.search_by_vector(vector, k=5)
        search_ms = (time.perf_counter() - t_search) * 1000

        embed_samples.append(embed_ms)
        search_samples.append(search_ms)
        total_samples.append(embed_ms + search_ms)

    result = BackendResult(
        backend=backend,
        model=model,
        dimension=embedder.dimension,
        chunks_indexed=len(chunks),
        index_build_s=round(index_build_s, 2),
        embed=StageStats.from_samples(embed_samples),
        search=StageStats.from_samples(search_samples),
        total=StageStats.from_samples(total_samples),
    )

    # Measure the query cache separately, so its benefit is never conflated
    # with the model's own speed.
    if cache_size > 0:
        cached = build_embedder(backend=backend, model_name=model, cache_size=cache_size)
        cached.warmup()
        repeated = queries[0]
        cached.embed_query(repeated)  # prime
        hit_samples = []
        for _ in range(runs):
            t = time.perf_counter()
            cached.embed_query(repeated)
            hit_samples.append((time.perf_counter() - t) * 1000)
        result.cache_hit_p50 = round(percentile(sorted(hit_samples), 50), 4)

    return result


def print_table(results: list[BackendResult]) -> None:
    print("\n" + "=" * 78)
    print("RESULTS (milliseconds per query)")
    print("=" * 78)
    header = f"{'backend':<24}{'stage':<10}{'p50':>10}{'p95':>10}{'p99':>10}{'mean':>10}"
    print(header)
    print("-" * 78)
    for result in results:
        for stage_name in ("embed", "search", "total"):
            stage: StageStats = getattr(result, stage_name)
            print(
                f"{result.backend:<24}{stage_name:<10}"
                f"{stage.p50:>10.2f}{stage.p95:>10.2f}{stage.p99:>10.2f}{stage.mean:>10.2f}"
            )
        print("-" * 78)


def print_comparison(baseline: BackendResult, optimized: BackendResult) -> dict:
    print("\n" + "=" * 78)
    print(f"REDUCTION: {optimized.backend} vs {baseline.backend}")
    print("=" * 78)

    deltas: dict[str, dict[str, float]] = {}
    for stage_name in ("embed", "search", "total"):
        base: StageStats = getattr(baseline, stage_name)
        opt: StageStats = getattr(optimized, stage_name)
        stage_deltas = {}
        for metric in ("p50", "p95", "p99"):
            b = getattr(base, metric)
            o = getattr(opt, metric)
            reduction = ((b - o) / b * 100) if b > 0 else 0.0
            stage_deltas[metric] = round(reduction, 1)
            print(
                f"  {stage_name:<8} {metric:<5} {b:>9.2f} ms -> {o:>9.2f} ms   "
                f"{reduction:+6.1f}%"
            )
        deltas[stage_name] = stage_deltas
        print()

    headline = deltas["total"]["p50"]
    print(f"  Headline: {headline:.1f}% p50 reduction in embed + search latency.")
    print("  (Hardware-specific. Quote the number your machine produced.)")
    return deltas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=Path("./data/docs"),
                        help="Directory of documents to index for the benchmark.")
    parser.add_argument("--queries", type=Path, default=None,
                        help="Newline-separated query file. Defaults to corpus-derived queries.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--ef-search", type=int, default=64, help="HNSW ef_search for both runs.")
    parser.add_argument("--cache-size", type=int, default=1024, help="0 to skip the cache probe.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Profile only the optimised path.")
    parser.add_argument("--out", type=Path, default=Path("bench/results/latest.json"))
    args = parser.parse_args(argv)

    # Keep the console clean: model-download chatter would drown the results.
    configure_logging("WARNING", fmt="%(levelname)-8s %(message)s")

    print(f"Loading corpus from {args.corpus} ...")
    chunks = load_corpus_chunks(args.corpus, args.chunk_size, args.chunk_overlap)
    print(f"  {len(chunks)} chunks")

    if len(chunks) < MIN_MEANINGFUL_CHUNKS:
        print(
            f"\n  NOTE: {len(chunks)} chunks is too small for the search-stage numbers to\n"
            f"  mean anything -- at this size HNSW query time is pure constant overhead\n"
            f"  and can even read as slower. The embedding-stage comparison is still\n"
            f"  valid (it does not depend on corpus size). For a representative\n"
            f"  search-stage figure, benchmark against {MIN_MEANINGFUL_CHUNKS}+ chunks."
        )

    if args.queries and args.queries.exists():
        queries = [q.strip() for q in args.queries.read_text().splitlines() if q.strip()]
        print(f"  {len(queries)} queries from {args.queries}")
    else:
        queries = derive_queries(chunks, count=max(args.runs, 20))
        print(f"  {len(queries)} queries derived from the corpus")

    results: list[BackendResult] = []
    with tempfile.TemporaryDirectory(prefix="rag-bench-") as tmp:
        store_root = Path(tmp)

        if not args.skip_baseline:
            try:
                results.append(
                    benchmark_backend(
                        "sentence_transformers", args.model, chunks, queries,
                        args.runs, args.warmup, args.ef_search, args.cache_size, store_root,
                    )
                )
            except ImportError as exc:
                print(f"\n[skipped baseline] {exc}", file=sys.stderr)

        results.append(
            benchmark_backend(
                "onnx_optimized", args.model, chunks, queries,
                args.runs, args.warmup, args.ef_search, args.cache_size, store_root,
            )
        )

    print_table(results)

    payload: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "runs": args.runs,
        "chunks": len(chunks),
        "ef_search": args.ef_search,
        "backends": [asdict(r) for r in results],
    }

    by_name = {r.backend: r for r in results}
    if "sentence_transformers" in by_name and "onnx_optimized" in by_name:
        payload["reduction_pct"] = print_comparison(
            by_name["sentence_transformers"], by_name["onnx_optimized"]
        )
    else:
        print("\nOnly one backend profiled -- no comparison. "
              "Install the baseline with:  pip install -e '.[baseline]'")

    for result in results:
        if result.cache_hit_p50 is not None:
            print(f"\n  {result.backend}: cached query embed p50 = {result.cache_hit_p50:.4f} ms "
                  f"(vs {result.embed.p50:.2f} ms uncached)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
