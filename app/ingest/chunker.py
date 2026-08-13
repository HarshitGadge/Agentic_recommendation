"""Chunking with stable, content-addressed IDs.

Chunk IDs are a hash of the chunk text plus its source, which makes ingestion
idempotent: re-running over the same corpus upserts the same IDs instead of
duplicating them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize(text: str) -> str:
    """Collapse runs of whitespace so trivially-different copies hash alike."""
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def chunk_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\x00{text}".encode()).hexdigest()
    return digest[:32]


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Paragraph-first splitter, degrading to sentences then words."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        length_function=len,
        keep_separator=True,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split documents into deduplicated, content-addressed chunks."""
    if not documents:
        return []

    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    seen: set[str] = set()

    for doc in documents:
        text = normalize(doc.page_content)
        if not text:
            continue
        source = str(doc.metadata.get("source", "unknown"))
        for position, raw_piece in enumerate(splitter.split_text(text)):
            piece = raw_piece.strip()
            if len(piece) < 20:  # fragments this small carry no retrievable signal
                continue
            cid = chunk_id(source, piece)
            if cid in seen:
                continue
            seen.add(cid)

            metadata = {k: v for k, v in doc.metadata.items() if v is not None}
            metadata["chunk_index"] = position
            metadata["char_count"] = len(piece)
            if extra_metadata:
                metadata.update(extra_metadata)
            chunks.append(Chunk(id=cid, text=piece, metadata=_coerce_metadata(metadata)))

    return chunks


def _coerce_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str/int/float/bool metadata values."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
