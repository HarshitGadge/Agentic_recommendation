"""Loader, chunker, and pipeline behaviour."""

from __future__ import annotations

import json

import pytest

from app.ingest.chunker import chunk_documents, chunk_id, normalize
from app.ingest.loaders import (
    UnsupportedFileError,
    discover_files,
    load_file,
    pick_text_columns,
)
from app.ingest.pipeline import ingest_path


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def test_discover_finds_every_supported_format(corpus):
    files = discover_files(corpus)
    suffixes = {f.suffix for f in files}
    assert suffixes == {".md", ".txt", ".csv", ".jsonl"}


def test_discover_ignores_unsupported_and_hidden(corpus):
    (corpus / "image.png").write_bytes(b"\x89PNG")
    (corpus / ".hidden.md").write_text("secret", encoding="utf-8")
    names = {f.name for f in discover_files(corpus)}
    assert "image.png" not in names
    assert ".hidden.md" not in names


def test_discover_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "nope")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def test_markdown_loads_as_single_document(corpus):
    docs = load_file(corpus / "photosynthesis.md")
    assert len(docs) == 1
    assert "chlorophyll" in docs[0].page_content
    assert docs[0].metadata["format"] == "md"


def test_csv_produces_one_document_per_row(corpus):
    docs = load_file(corpus / "notes.csv")
    assert len(docs) == 2
    assert docs[0].metadata["row"] == 1
    # The numeric id column must not be treated as prose.
    assert "id" not in docs[0].metadata["text_columns"].split(",")


def test_csv_keeps_short_columns_as_metadata(corpus):
    docs = load_file(corpus / "notes.csv")
    assert docs[0].metadata["col_id"] == "1"


def test_jsonl_flattens_records(corpus):
    docs = load_file(corpus / "records.jsonl")
    assert len(docs) == 2
    assert "Enzymes" in docs[0].page_content or "catalysts" in docs[0].page_content


def test_json_array_of_objects(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {"name": "alpha", "text": "A reasonably long description of alpha " * 3},
                {"name": "beta", "text": "A reasonably long description of beta " * 3},
            ]
        ),
        encoding="utf-8",
    )
    docs = load_file(path)
    assert len(docs) == 2


def test_json_nested_wrapper_is_unwrapped(tmp_path):
    path = tmp_path / "wrapped.json"
    path.write_text(
        json.dumps({"meta": {"v": 1}, "results": [{"body": "Long enough body text here " * 4}]}),
        encoding="utf-8",
    )
    docs = load_file(path)
    assert len(docs) == 1


def test_malformed_json_raises_unsupported(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(UnsupportedFileError):
        load_file(path)


def test_pick_text_columns_skips_numeric_and_id(tmp_path):
    rows = [
        {"id": "1", "price": "19.99", "review": "This product exceeded my expectations entirely."},
        {"id": "2", "price": "24.50", "review": "Battery life was disappointing over long trips."},
    ]
    assert pick_text_columns(rows) == ["review"]


def test_pick_text_columns_falls_back_to_longest(tmp_path):
    rows = [{"a": "short", "b": "a bit longer"}, {"a": "tiny", "b": "also a bit longer"}]
    assert pick_text_columns(rows) == ["b"]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def test_normalize_collapses_whitespace():
    assert normalize("a   b\n\n\n\nc") == "a b\n\nc"


def test_chunk_ids_are_content_addressed():
    assert chunk_id("s.txt", "hello") == chunk_id("s.txt", "hello")
    assert chunk_id("s.txt", "hello") != chunk_id("other.txt", "hello")


def test_chunking_deduplicates_identical_text(corpus):
    docs = load_file(corpus / "photosynthesis.md")
    duplicated = docs + docs
    chunks = chunk_documents(duplicated, chunk_size=200, chunk_overlap=40)
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_metadata_is_chroma_safe(corpus):
    docs = load_file(corpus / "notes.csv")
    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
    for chunk in chunks:
        for value in chunk.metadata.values():
            assert isinstance(value, (str, int, float, bool))


def test_chunk_overlap_must_be_smaller_than_size(corpus):
    docs = load_file(corpus / "respiration.txt")
    chunks = chunk_documents(docs, chunk_size=150, chunk_overlap=30)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def test_ingest_writes_chunks(store, corpus):
    result = ingest_path(path=corpus, store=store, chunk_size=400, chunk_overlap=60)
    assert result.files_ingested == 4
    assert result.chunks_written > 0
    assert store.count() == result.chunks_written


def test_ingest_is_idempotent(store, corpus):
    first = ingest_path(path=corpus, store=store, chunk_size=400, chunk_overlap=60)
    count_after_first = store.count()
    ingest_path(path=corpus, store=store, chunk_size=400, chunk_overlap=60)
    # Content-addressed IDs mean the second pass upserts, never duplicates.
    assert store.count() == count_after_first == first.chunks_written


def test_ingest_applies_source_tag(store, corpus):
    ingest_path(
        path=corpus, store=store, chunk_size=400, chunk_overlap=60, source_tag="kaggle-bio"
    )
    hits = store.search("photosynthesis chlorophyll", k=1)
    assert hits[0].metadata["source_tag"] == "kaggle-bio"


def test_ingest_empty_directory_is_not_an_error(store, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = ingest_path(path=empty, store=store, chunk_size=400, chunk_overlap=60)
    assert result.files_seen == 0
    assert result.chunks_written == 0
