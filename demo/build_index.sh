#!/bin/sh
# Build the read-only demo index: one source tag per project, so a query can be
# scoped to a single project with {"source_filter": "<project>"}.
#
# Runs at image build time when RAG_DEMO_INDEX=true (see Dockerfile), so the
# deployed container starts with a ready index and never needs a writable
# ingest endpoint. Run it locally the same way:
#
#     sh demo/build_index.sh
set -eu

CORPUS="$(dirname "$0")/corpus"

for project in "$CORPUS"/*/; do
    tag="$(basename "$project")"
    echo "==> indexing $tag"
    python -m scripts.ingest "$project" --source-tag "$tag"
done
