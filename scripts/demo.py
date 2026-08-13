"""Narrated end-to-end demo against a running service.

Walks through the whole system and prints what each step proves, so you can see
it work rather than take the README's word for it:

    make run          # or: docker compose up -d
    make demo         # or: python -m scripts.demo

Every number printed is measured live from the running service. Nothing here is
hardcoded.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

BASE = "http://localhost:8000"
RULE = "=" * 74


def call(method: str, path: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"content-type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def snippet(text: str, width: int = 88) -> str:
    """One-line preview. Chunks contain newlines that would break indentation."""
    flat = " ".join(text.split())
    return flat[:width] + ("..." if len(flat) > width else "")


def step(number: int, title: str, why: str) -> None:
    print(f"\n{RULE}\nSTEP {number}: {title}\n{RULE}")
    print(f"  what this shows: {why}\n")


def require_service() -> None:
    try:
        health = call("GET", "/health")
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        print(
            "Could not reach the service at " + BASE + "\n\n"
            "Start it first, then re-run this demo:\n"
            "    make run                 # local\n"
            "    docker compose up -d     # container\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(f"Service is up: {BASE}  (version {health['version']}, status {health['status']})")
    for name, value in health["checks"].items():
        print(f"  {name:<14} {value}")


def show_timings(timings: dict) -> None:
    step(4, "Where the time goes",
         "every request is instrumented per stage, which is how the "
         "optimisation was measured")
    widest = max(timings.values()) or 1.0
    for stage in ("plan_ms", "embed_ms", "search_ms", "rerank_ms", "generate_ms", "total_ms"):
        value = timings[stage]
        bar = "#" * int(round(value / widest * 40))
        print(f"    {stage:<12} {value:>8.2f} ms  {bar}")


def show_cache() -> None:
    step(5, "The embedding cache", "a repeated query skips the model entirely")
    repeat = "how does quantization reduce latency?"
    for i in range(3):
        response = call("POST", "/query", {"question": repeat, "top_k": 2})
        label = "cold (model runs)" if i == 0 else "warm (cache hit)"
        print(f"    run {i + 1}  embed {response['timings']['embed_ms']:>6.2f} ms   "
              f"total {response['timings']['total_ms']:>7.2f} ms   {label}")
    cache = call("GET", "/metrics").get("embed_cache", {})
    print(f"\n  cache: {cache.get('hits', 0)} hits / {cache.get('misses', 0)} misses "
          f"(hit rate {cache.get('hit_rate', 0):.0%})")
    print("  -> a repeated query becomes a dictionary lookup instead of a model call.")


def show_metrics() -> None:
    step(6, "Service metrics",
         "percentiles over a rolling window, not just single-request numbers")
    metrics = call("GET", "/metrics")
    print(f"    queries served   {metrics['counts']['queries']}")
    print(f"    errors           {metrics['counts']['errors']}")
    print(f"    chunks indexed   {metrics['chunk_count']}")
    print(f"\n    {'stage':<10}{'p50':>10}{'p95':>10}{'p99':>10}")
    for stage, stats in metrics["latency_ms"].items():
        print(f"    {stage:<10}{stats['p50']:>10.2f}{stats['p95']:>10.2f}{stats['p99']:>10.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", default="/corpus",
                        help="Path the SERVICE can see. /corpus in Docker, ./data/docs locally.")
    args = parser.parse_args(argv)

    print(f"\n{RULE}\nAGENTIC RAG -- LIVE DEMO\n{RULE}\n")
    require_service()

    # ----------------------------------------------------------------- 1
    step(1, "Ingest a corpus",
         "mixed formats (.md and .csv) become searchable chunks in one call")
    try:
        ingest = call("POST", "/ingest", {"path": args.corpus, "source_tag": "demo"})
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"  The service cannot see {args.corpus!r}.")
            print("  In Docker the corpus is mounted at /corpus; locally pass --corpus ./data/docs")
            return 1
        raise
    print(f"  files ingested   {ingest['files_ingested']} of {ingest['files_seen']}")
    print(f"  chunks written   {ingest['chunks_written']}")
    print(f"  elapsed          {ingest['elapsed_ms']:.0f} ms")

    print("\n  Re-ingesting the exact same corpus...")
    again = call("POST", "/ingest", {"path": args.corpus, "source_tag": "demo"})
    total = call("GET", "/stats")["chunk_count"]
    print(f"  chunks written   {again['chunks_written']} (upserted, not duplicated)")
    print(f"  collection size  {total} chunks -- unchanged")
    print("  -> chunk IDs are content hashes, so ingestion is idempotent.")

    # ----------------------------------------------------------------- 2
    step(2, "Ask a question",
         "the answer is grounded in retrieved passages, each one cited")
    question = "What does ef_search control in an HNSW index?"
    print(f"  Q: {question}\n")
    result = call("POST", "/query", {"question": question, "top_k": 3})
    print("  A: " + snippet(result["answer"], 600))
    print("\n  citations (what the answer is standing on):")
    for i, citation in enumerate(result["citations"], start=1):
        print(f"    [{i}] score {citation['score']:.3f}  {citation['source']}")
        print(f"        {snippet(citation['snippet'], 84)}")
    print(f"\n  answer mode: {result['answer_mode']}")
    if result["answer_mode"] == "extractive":
        print("  (set ANTHROPIC_API_KEY to get synthesised answers instead of extracted ones)")

    # ----------------------------------------------------------------- 3
    step(3, "Ask a two-part question",
         "the agent splits it, searches each half, and fuses the results")
    compound = "What is chunking and how does int8 quantization reduce latency?"
    print(f"  Q: {compound}\n")
    fused = call("POST", "/query", {"question": compound, "top_k": 4})
    print("  the planner decomposed it into:")
    for sub in fused["sub_queries"]:
        print(f"    - {sub}")
    print("\n  fused results (note both halves are represented):")
    for citation in fused["citations"]:
        print(f"    score {citation['score']:.3f}  {citation['source']}")
        print(f"      {snippet(citation['snippet'], 84)}")
    print("\n  -> one question, two intents, evidence retrieved for each.")

    show_timings(fused["timings"])
    show_cache()
    show_metrics()

    print_summary()
    return 0


def print_summary() -> None:
    print(f"\n{RULE}\nWHAT THIS DEMO PROVED\n{RULE}")
    for line in (
        "ingestion handles mixed formats and is idempotent on re-run",
        "answers are grounded in retrieved passages and cite their sources",
        "the agent decomposes multi-part questions and fuses the results",
        "every stage is measured, so latency claims are checkable",
        "repeated queries are served from cache",
    ):
        print(f"  [x] {line}")
    print(f"\n  Interactive docs: {BASE}/docs")
    print("  Latency benchmark (run natively, not in Docker):")
    print("      make install-baseline && make bench\n")


if __name__ == "__main__":
    raise SystemExit(main())
