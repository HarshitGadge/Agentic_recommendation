"""Download BEIR SciFact (5,183 abstracts, 300 test queries with relevance labels).

Source: the BEIR benchmark's copy on the Hugging Face Hub (BeIR/scifact, BeIR/scifact-qrels),
converted to BEIR's standard layout: corpus.jsonl, queries.jsonl, qrels/test.tsv.

    python -m eval.download_scifact
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets"
FILES = {
    "corpus": f"{HF}/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "queries": f"{HF}/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
    "qrels": f"{HF}/BeIR/scifact-qrels/resolve/main/test.tsv",
}
OUT = Path(__file__).resolve().parent / "data" / "scifact"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed HTTPS URLs
        return response.read()


def main() -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required:  pip install pyarrow", file=sys.stderr)
        return 1

    (OUT / "qrels").mkdir(parents=True, exist_ok=True)
    for name in ("corpus", "queries"):
        table = pq.read_table(io.BytesIO(fetch(FILES[name])))
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in table.to_pylist():
                fh.write(json.dumps(row) + "\n")
        print(f"{name}: {table.num_rows:,} rows")
    (OUT / "qrels" / "test.tsv").write_bytes(fetch(FILES["qrels"]))
    print(f"Saved to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
