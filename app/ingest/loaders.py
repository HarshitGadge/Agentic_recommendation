"""Format-specific loaders.

Every loader yields LangChain ``Document`` objects so the rest of the pipeline
is format-agnostic. Kaggle datasets are usually CSV/JSON, so those loaders do
the extra work of picking sensible text columns.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {
    ".txt", ".md", ".markdown", ".pdf", ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
}

# Kaggle CSVs regularly carry single cells larger than the stdlib default.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Columns that are almost never worth embedding as prose.
_ID_LIKE = {"id", "index", "unnamed: 0", "row", "rowid", "key", "uuid"}


class UnsupportedFileError(ValueError):
    """Raised when a path has no loader registered."""


def discover_files(root: Path, recursive: bool = True) -> list[Path]:
    """Return every loadable file under ``root`` (or ``root`` itself if a file)."""
    root = root.expanduser()
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    if not root.is_dir():
        raise FileNotFoundError(f"No such file or directory: {root}")

    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in sorted(root.glob(pattern))
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith(".")
    ]
    return files


def load_file(path: Path) -> list[Document]:
    """Dispatch to the loader matching ``path``'s extension."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return _load_text(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".csv", ".tsv"}:
        return _load_delimited(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(path)
    raise UnsupportedFileError(f"No loader for {suffix!r} ({path})")


# --------------------------------------------------------------------------
# Text / Markdown
# --------------------------------------------------------------------------
def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    metadata = {"source": str(path), "format": path.suffix.lstrip(".")}
    return [Document(page_content=text, metadata=metadata)]


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # corrupt or encrypted PDF
        raise UnsupportedFileError(f"Could not read PDF {path}: {exc}") from exc

    docs: list[Document] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # a single unreadable page shouldn't sink the file
            logger.warning("Skipping unreadable page %d of %s", page_no, path)
            continue
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "format": "pdf", "page": page_no},
                )
            )
    return docs


# --------------------------------------------------------------------------
# CSV / TSV
# --------------------------------------------------------------------------
def _load_delimited(path: Path, text_columns: list[str] | None = None) -> list[Document]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        return []

    chosen = text_columns or pick_text_columns(rows)
    if not chosen:
        logger.warning("No text-like columns found in %s; skipping", path)
        return []

    docs: list[Document] = []
    for row_no, row in enumerate(rows, start=1):
        content = _row_to_text(row, chosen)
        if not content:
            continue
        metadata: dict[str, Any] = {
            "source": str(path),
            "format": path.suffix.lstrip("."),
            "row": row_no,
            "text_columns": ",".join(chosen),
        }
        # Keep short non-text columns as filterable metadata.
        for key, value in row.items():
            if key and key not in chosen and value and len(str(value)) <= 64:
                metadata[f"col_{key}"] = str(value)
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def pick_text_columns(
    rows: list[dict[str, Any]], sample: int = 200, min_avg_len: int = 25
) -> list[str]:
    """Heuristically choose which columns hold prose worth embedding.

    Scores each column by mean character length over a sample and keeps those
    above ``min_avg_len``. Falls back to the single longest column so a table of
    short strings still produces something searchable.
    """
    if not rows:
        return []

    columns = [c for c in (rows[0].keys() or []) if c]
    scores: dict[str, float] = {}
    for col in columns:
        if col.strip().lower() in _ID_LIKE:
            continue
        values = [str(r.get(col) or "") for r in rows[:sample]]
        non_empty = [v for v in values if v]
        if not non_empty:
            continue
        avg_len = sum(len(v) for v in non_empty) / len(non_empty)
        # Numeric columns are filters, not prose.
        if _mostly_numeric(non_empty):
            continue
        scores[col] = avg_len

    if not scores:
        return []

    chosen = [c for c, s in scores.items() if s >= min_avg_len]
    if not chosen:
        chosen = [max(scores, key=scores.get)]
    # Preserve original column order for stable, readable output.
    return [c for c in columns if c in chosen]


def _mostly_numeric(values: list[str], threshold: float = 0.9) -> bool:
    numeric = 0
    for v in values:
        try:
            float(v.replace(",", ""))
            numeric += 1
        except ValueError:
            pass
    return numeric / len(values) >= threshold


def _row_to_text(row: dict[str, Any], columns: list[str]) -> str:
    parts = []
    for col in columns:
        value = str(row.get(col) or "").strip()
        if value:
            parts.append(f"{col}: {value}" if len(columns) > 1 else value)
    return "\n".join(parts).strip()


# --------------------------------------------------------------------------
# JSON / JSONL
# --------------------------------------------------------------------------
def _load_json(path: Path) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UnsupportedFileError(f"Invalid JSON in {path}: {exc}") from exc

    records = _as_records(payload)
    if records:
        return _records_to_docs(records, path, "json")
    # Not record-shaped -- embed the pretty-printed document whole.
    return [
        Document(
            page_content=json.dumps(payload, indent=2, ensure_ascii=False),
            metadata={"source": str(path), "format": "json"},
        )
    ]


def _load_jsonl(path: Path) -> list[Document]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON on %s:%d", path, line_no)
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return _records_to_docs(records, path, "jsonl")


def _as_records(payload: Any) -> list[dict[str, Any]]:
    """Pull a list-of-objects out of the common JSON dataset shapes."""
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
    return []


def _records_to_docs(records: list[dict[str, Any]], path: Path, fmt: str) -> list[Document]:
    if not records:
        return []
    flat = [_flatten(r) for r in records]
    chosen = pick_text_columns(flat)
    if not chosen:
        logger.warning("No text-like fields found in %s; skipping", path)
        return []

    docs: list[Document] = []
    for row_no, row in enumerate(flat, start=1):
        content = _row_to_text(row, chosen)
        if not content:
            continue
        metadata: dict[str, Any] = {
            "source": str(path),
            "format": fmt,
            "row": row_no,
            "text_columns": ",".join(chosen),
        }
        for key, value in row.items():
            if key not in chosen and value is not None and len(str(value)) <= 64:
                metadata[f"col_{key}"] = str(value)
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def _flatten(obj: dict[str, Any], prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Flatten nested JSON one dot-path at a time, bottoming out at depth 3."""
    flat: dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and depth < 3:
            flat.update(_flatten(value, prefix=f"{name}.", depth=depth + 1))
        elif isinstance(value, list):
            flat[name] = ", ".join(str(v) for v in value if not isinstance(v, (dict, list)))
        else:
            flat[name] = value
    return flat


def iter_documents(paths: list[Path]) -> Iterator[tuple[Path, list[Document]]]:
    """Yield ``(path, documents)`` per file, logging and skipping failures."""
    for path in paths:
        try:
            yield path, load_file(path)
        except (UnsupportedFileError, OSError) as exc:
            logger.warning("Skipping %s: %s", path, exc)
            yield path, []
