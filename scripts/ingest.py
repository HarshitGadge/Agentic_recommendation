"""CLI ingestion -- index a corpus without starting the API server.

    python -m scripts.ingest ./data/docs --source-tag kaggle-news
    python -m scripts.ingest ./data/docs --reset      # rebuild from scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.ingest.pipeline import ingest_path
from app.obs.logging_conf import configure_logging
from app.retrieval.embeddings import build_embedder
from app.retrieval.store import VectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", type=Path, help="File or directory to ingest.")
    parser.add_argument("--source-tag", default=None,
                        help="Label stored on every chunk (e.g. the Kaggle dataset slug).")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Do not descend into subdirectories.")
    parser.add_argument("--reset", action="store_true",
                        help="Drop the collection before ingesting. Destructive.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(
        "DEBUG" if args.verbose else "INFO", fmt="%(levelname)-8s %(message)s"
    )

    settings = get_settings()
    embedder = build_embedder(
        backend=settings.embed_backend,
        model_name=settings.embed_model,
        batch_size=settings.embed_batch_size,
        cache_size=0,  # ingestion never re-embeds the same query
        model_path=settings.embed_model_path or None,
    )
    store = VectorStore(
        embedder=embedder,
        path=settings.chroma_path,
        collection_name=settings.collection,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )

    if args.reset:
        confirm = input(f"Delete all {store.count()} chunks in '{settings.collection}'? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return 1
        store.reset()
        print("Collection reset.")

    try:
        result = ingest_path(
            path=args.path,
            store=store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            recursive=not args.no_recursive,
            source_tag=args.source_tag,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nFiles seen:      {result.files_seen}\n"
        f"Files ingested:  {result.files_ingested}\n"
        f"Files skipped:   {len(result.files_skipped)}\n"
        f"Chunks written:  {result.chunks_written}\n"
        f"Duplicates:      {result.duplicates_dropped}\n"
        f"Elapsed:         {result.elapsed_ms / 1000:.1f}s\n"
        f"Collection size: {store.count()} chunks"
    )
    if result.files_skipped:
        print("\nSkipped (no extractable text):")
        for path in result.files_skipped[:20]:
            print(f"  {path}")
        if len(result.files_skipped) > 20:
            print(f"  ... and {len(result.files_skipped) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
