"""End-to-end ingestion: discover -> load -> chunk -> embed -> upsert."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.ingest.chunker import Chunk, chunk_documents
from app.ingest.loaders import discover_files, iter_documents
from app.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: list[str] = field(default_factory=list)
    chunks_written: int = 0
    duplicates_dropped: int = 0
    elapsed_ms: float = 0.0


def ingest_path(
    path: str | Path,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
    recursive: bool = True,
    source_tag: str | None = None,
    batch_size: int = 256,
) -> IngestResult:
    """Ingest a file or directory into ``store``.

    Chunk IDs are content hashes, so re-ingesting an unchanged corpus is a
    no-op upsert rather than a duplication.
    """
    started = time.perf_counter()
    result = IngestResult()

    files = discover_files(Path(path), recursive=recursive)
    result.files_seen = len(files)
    if not files:
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    extra_metadata = {"source_tag": source_tag} if source_tag else None
    pending: list[Chunk] = []
    seen_ids: set[str] = set()

    for file_path, documents in iter_documents(files):
        if not documents:
            result.files_skipped.append(str(file_path))
            continue

        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_metadata=extra_metadata,
        )
        if not chunks:
            result.files_skipped.append(str(file_path))
            continue

        result.files_ingested += 1
        for chunk in chunks:
            # Cross-file dedup: identical text from two files is stored once.
            if chunk.id in seen_ids:
                result.duplicates_dropped += 1
                continue
            seen_ids.add(chunk.id)
            pending.append(chunk)

        while len(pending) >= batch_size:
            batch, pending = pending[:batch_size], pending[batch_size:]
            store.upsert(batch)
            result.chunks_written += len(batch)
            logger.info("Wrote %d chunks (%d total)", len(batch), result.chunks_written)

    if pending:
        store.upsert(pending)
        result.chunks_written += len(pending)

    result.elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Ingest complete: %d/%d files, %d chunks, %d dupes, %.0f ms",
        result.files_ingested,
        result.files_seen,
        result.chunks_written,
        result.duplicates_dropped,
        result.elapsed_ms,
    )
    return result
