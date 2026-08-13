"""Logging setup.

Third-party HTTP clients log every request at INFO. During model download that
buries our own output under dozens of Hugging Face redirect lines, so they are
pinned to WARNING unless the app itself is running at DEBUG.
"""

from __future__ import annotations

import logging

NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "filelock",
    "chromadb",
    "chromadb.telemetry",
    "onnxruntime",
    "sentence_transformers",
)


def configure_logging(level: str = "INFO", fmt: str | None = None) -> None:
    resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format=fmt or "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )
    # At DEBUG the caller is explicitly asking for everything, so leave them be.
    if resolved > logging.DEBUG:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
