# Agentic RAG Service

[![tests](https://github.com/HarshitGadge/Agentic_recommendation/actions/workflows/tests.yml/badge.svg)](https://github.com/HarshitGadge/Agentic_recommendation/actions/workflows/tests.yml)

A retrieval-augmented generation service you can point at a folder of documents
and query over HTTP. LangChain does the chunking, ChromaDB stores and searches
the vectors, an optional BM25 keyword index adds hybrid search, an adaptive
retrieval agent decides how hard to look, and answers come back grounded in
numbered citations. Runs in Docker; runs without an API key.

It ships with two measurements rather than claims:

- **Retrieval quality:** `eval/retrieval_eval.py` runs the service's own chunker,
  embedder, indexes and agent over the BEIR SciFact benchmark (5,183 abstracts,
  300 questions with human relevance labels) and reports nDCG@10, recall and MRR
  per configuration, with bootstrap confidence intervals. The service's defaults
  were changed because of what it found: **nDCG@10 went from 0.663 to 0.713
  (+4.96 points, 95% CI +3.1 to +6.9)**.
- **Latency:** `bench/benchmark.py` builds two indexes over your corpus — one on
  the PyTorch fp32 baseline, one on the graph-optimised ONNX Runtime path —
  replays the same queries against both, and reports the p50/p95/p99 difference.

---

## Contents

- [See it working](#see-it-working)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Using a Kaggle dataset](#using-a-kaggle-dataset)
- [API](#api)
- [Retrieval quality](#retrieval-quality)
- [The latency benchmark](#the-latency-benchmark)
- [Configuration](#configuration)
- [Docker](#docker)
- [Testing](#testing)
- [Design notes](#design-notes)

---

## See it working

Two commands, no API key:

```bash
docker compose up -d      # or: make install && make ingest && make run
make demo-docker          # or, for a local server: make demo
```

`make demo` walks the whole system and prints what each step proves — ingestion
and its idempotency, a grounded answer with citations, a two-part question being
decomposed and its results fused, a per-stage latency breakdown, and the query
cache. Every number it prints is measured live from the running service.

The cache step is the one worth watching:

```
    run 1  embed   5.34 ms   total   13.67 ms   cold (model runs)
    run 2  embed   0.00 ms   total    2.33 ms   warm (cache hit)
    run 3  embed   0.00 ms   total    1.97 ms   warm (cache hit)
```

For a browser instead, open **[http://localhost:8000/docs](http://localhost:8000/docs)**
and hit **Try it out** on any endpoint.

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
   POST /ingest ───►│  Ingestion                                   │
                    │  loaders → LangChain splitter → SHA-256 IDs  │
                    └───────────────────┬──────────────────────────┘
                                        │ embed (batch)
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │  ChromaDB  (persistent, cosine, tuned HNSW)  │
                    │  + BM25 keyword index (in memory, optional)  │
                    └───────────────────▲──────────────────────────┘
                                        │ vector (+ keyword) search
   POST /query ────►┌──────────────────────────────────────────────┐
                    │  Retrieval agent                             │
                    │   1. plan     decompose multi-part questions │
                    │   2. retrieve one search per sub-query       │
                    │               (vector, plus BM25 if hybrid)  │
                    │   3. fuse     reciprocal rank fusion (RRF)   │
                    │   4. diversify  MMR (opt-in, off by default) │
                    │   5. assess   retry broadened if coverage    │
                    │               was weak  (bounded: 2 rounds)  │
                    └───────────────────┬──────────────────────────┘
                                        │ numbered context
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │  Generation                                  │
                    │   Claude (grounded + cited)  ── or ──        │
                    │   extractive fallback (no API key needed)    │
                    └──────────────────────────────────────────────┘
```

**Stack:** FastAPI · ChromaDB · LangChain (`langchain-text-splitters`,
`langchain-core`) · Hugging Face embeddings via fastembed / ONNX Runtime ·
BM25 (built in, no extra dependency) · Anthropic SDK · Docker

Every response carries per-stage timings (`plan / embed / search / rerank /
generate`), and `/metrics` aggregates them into percentiles over a rolling
window. That instrumentation is what makes the optimisation work measurable
instead of anecdotal.

---

## Quickstart

```bash
git clone https://github.com/HarshitGadge/Agentic_recommendation.git agentic-rag
cd agentic-rag

make install          # venv + dependencies
```

Drop some documents in `data/docs/` (`.txt .md .pdf .csv .tsv .json .jsonl`),
index them, and start the API:

```bash
mkdir -p data/docs
cp ~/Downloads/whatever.csv data/docs/

make ingest           # or: python -m scripts.ingest ./data/docs
make run
```

Now open **[http://localhost:8000/docs](http://localhost:8000/docs)** — Swagger UI,
where you can run every endpoint from the browser without touching curl.

Ask it something:

```bash
curl -s localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "What does the dataset say about pricing?"}' | jq
```

**Answers work without an API key.** With `ANTHROPIC_API_KEY` unset the service
returns an extractive answer — the most relevant sentences from the retrieved
passages, with citations. Set the key to get synthesised answers instead:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Everything else — ingestion, retrieval, the agent loop, the benchmark, the
tests — runs identically either way.

---

## Using a Kaggle dataset

Kaggle datasets are usually CSV or JSON, and the loaders handle that shape
directly. Download and unzip anywhere, then point the ingester at the folder:

```bash
kaggle datasets download -d <owner>/<dataset> -p data/docs --unzip
python -m scripts.ingest ./data/docs --source-tag <owner>/<dataset>
```

**How tabular files become searchable text.** For each CSV/JSON file the loader
scores every column by mean string length over a sample and keeps the ones that
read like prose. ID-like and mostly-numeric columns are excluded — they're
filters, not content. Each row becomes one document (`column: value` per chosen
column), and short non-text columns are preserved as metadata you can filter on
later.

Override the heuristic when you know better:

```bash
RAG_TEXT_COLUMNS=review_body,product_title python -m scripts.ingest ./data/docs
```

`--source-tag` labels every chunk from that run, so you can keep several
datasets in one collection and scope a query to one of them:

```json
{"question": "...", "source_filter": "<owner>/<dataset>"}
```

Re-ingesting the same files is safe: chunk IDs are a SHA-256 of the chunk text
plus its source, so a second pass upserts the same rows instead of duplicating
them.

---

## API

**Open [http://localhost:8000/docs](http://localhost:8000/docs)** for interactive
Swagger UI where you can run every endpoint from the browser. The bare host
redirects there. All endpoints return an `x-request-id` header.

| Method | Path             | Purpose |
|--------|------------------|---------|
| `GET`  | `/`              | Redirects to `/docs` |
| `GET`  | `/health`        | Liveness + component checks |
| `GET`  | `/stats`         | Chunk count, sources, model and backend in use |
| `GET`  | `/metrics`       | Per-stage latency percentiles, cache hit rate |
| `POST` | `/ingest`        | Index a server-side file or directory |
| `POST` | `/ingest/upload` | Index uploaded files (multipart) |
| `POST` | `/query`         | Retrieve and answer |

### `POST /query`

```jsonc
{
  "question": "How does the Calvin cycle fix carbon?",
  "top_k": 5,              // optional, defaults to RAG_TOP_K
  "decompose": true,       // split multi-part questions into sub-queries
  "generate": false,       // false = retrieval only, no answer synthesis
  "source_filter": null    // restrict to one --source-tag
}
```

```jsonc
{
  "question": "How does the Calvin cycle fix carbon?",
  "sub_queries": ["How does the Calvin cycle fix carbon?"],
  "answer": "The Calvin cycle uses ATP and NADPH from the light-dependent reactions to fix carbon dioxide into sugar in the stroma [1][3].",
  "answer_mode": "llm",
  "citations": [
    {
      "chunk_id": "9f2a...",
      "source": "data/docs/photosynthesis.md",
      "score": 0.83,
      "snippet": "The Calvin cycle then uses these to fix carbon dioxide...",
      "metadata": {"page": 4, "source_tag": "biology"}
    }
  ],
  "timings": {
    "plan_ms": 0.04, "embed_ms": 3.9, "search_ms": 2.1,
    "rerank_ms": 1.2, "generate_ms": 812.5, "total_ms": 819.8
  },
  "request_id": "a1b2c3d4e5f6"
}
```

The bracketed numbers in `answer` index into `citations`, so every claim maps
back to a specific chunk and source file. `generate: false` gives you the
retrieval layer alone — useful for evaluating recall without paying for
generation.

---

## Retrieval quality

`generate: false` makes the retrieval layer measurable on its own, and
`eval/retrieval_eval.py` measures it on **BEIR SciFact**: 5,183 scientific
abstracts and 300 test claims, each labelled with the abstracts that support or
refute it. The corpus goes through the same chunker, embedder, ChromaDB index,
BM25 index and retrieval agent as a real deployment (17,332 chunks). A document
counts once, at the rank of its best chunk.

```bash
make eval-data      # download SciFact (~4.6 MB) into eval/data/
make eval           # prints the table below; writes eval/results/scifact.json
```

Add `--compare-model-path <dir>` to score a second local model (for example the
fp32 ONNX export in `onnx/model.onnx` plus its tokenizer files) side by side on
quality and query-embedding latency.

### Results

`BAAI/bge-small-en-v1.5` via fastembed. Top 10 chunks per query from 50 candidates;
the single ranked lists are also scored 100 deep for recall@100:

| configuration | nDCG@10 | Recall@10 | MRR@10 | Recall@100 |
|---|---|---|---|---|
| vector search | **0.727** | 0.844 | 0.696 | 0.940 |
| BM25 keyword search | 0.610 | 0.734 | 0.577 | 0.868 |
| hybrid, BM25 weight 1.0 (plain RRF) | 0.700 | 0.835 | 0.662 | 0.959 |
| hybrid, BM25 weight 0.5 | 0.710 | 0.835 | 0.675 | **0.962** |
| service, previous defaults (MMR λ=0.5) | 0.663 | 0.734 | 0.659 | – |
| **service, new defaults (MMR off)** | **0.713** | **0.806** | **0.690** | – |
| service, hybrid on (weight 0.5) | 0.697 | 0.801 | 0.670 | – |

Paired bootstrap over the 300 queries (2,000 resamples), change in nDCG@10:

| comparison | Δ nDCG@10 | 95% CI |
|---|---|---|
| new vs previous service defaults | **+0.050** | +0.031 to +0.069 |
| hybrid (weight 0.5) vs vector | −0.017 | −0.039 to +0.004 |
| hybrid on vs off, in the service | −0.016 | −0.039 to +0.005 |

The vector score is in line with the figure BAAI publishes for this model on the
SciFact test split (nDCG@10 0.713, in the model card's MTEB results), a useful
check that the chunking and document-level scoring aren't distorting anything.

### What the evaluation changed

- **MMR is now off by default** (`RAG_MMR_LAMBDA=1.0`). At λ=0.5 it cost 5 points
  of nDCG@10 and 7 points of recall@10. Pushing out near-duplicate chunks also
  pushed out relevant documents. A sweep over λ = 0.9, 0.8 and 0.7 lost 0.3, 1.0
  and 2.0 points. It stays available for corpora with many near-identical passages.
- **Hybrid search is opt-in** (`RAG_HYBRID_SEARCH=true`). Fusing BM25 raised
  recall@100 from 0.940 to 0.962, so a reranker or a larger context window would
  see more relevant documents. But it didn't improve the top 10, where this
  embedding model is already much stronger than keyword matching. Down-weighting
  BM25 to 0.5 in the fusion narrowed the gap (0.700 → 0.710, vs 0.727). It's
  worth enabling for corpora full of exact identifiers (part numbers, error codes,
  gene names) that embeddings blur; SciFact doesn't test that case.
- **A fusion bug was fixed along the way.** Rank fusion used to overwrite each
  hit's cosine score with its RRF score (about 0.03). That is always below the
  agent's weak-coverage threshold (0.35), so every multi-part question triggered a
  retry, and the retry's results replaced the decomposed ones. Scores now stay
  cosine similarities, and the fusion score is kept separately. A regression test
  covers it.

Run-to-run variation is about ±0.003 nDCG@10, because ChromaDB builds its
approximate HNSW index with multiple threads.

---

## The latency benchmark

The optimisation is running the same Hugging Face embedding model through
**ONNX Runtime** (via fastembed) instead of **PyTorch**. For
`bge-small-en-v1.5`, fastembed loads a graph-optimised export: attention, GELU and
layer-norm are fused into single kernels, and the weights are stored in fp16.

> **Correction:** earlier versions of this README called that model
> "int8-quantised", after its Hugging Face repository name
> (`Qdrant/bge-small-en-v1.5-onnx-Q`). Inspecting the file shows otherwise: its
> `ort_config.json` has an empty quantization block and all 149 weight tensors are
> FLOAT16. The speed-up comes from the optimised graph and ONNX Runtime, not from
> int8 arithmetic. The backend is now called `onnx_optimized`; `onnx_quantized` is
> still accepted as a deprecated alias.
>
> Retrieval quality is unaffected: on SciFact the served model scores nDCG@10
> 0.727 and the original fp32 ONNX export 0.728, and in spot checks their vectors
> agree to a cosine similarity above 0.99999.

On top of that: tuned HNSW parameters and an LRU cache over query embeddings.

`bench/benchmark.py` measures it rather than asserting it. It builds a separate
index per backend over your corpus, replays identical queries against each
(discarding warmup rounds), and reports percentiles per stage:

```bash
make install-baseline          # adds torch/sentence-transformers
python -m bench.benchmark --corpus ./data/docs --runs 50
```

### Measured result

Apple M3, 8 cores, 8 GB, macOS 26.3.1, Python 3.12.7 · `BAAI/bge-small-en-v1.5`
· 2,500 chunks · 60 timed queries per backend:

| stage | backend | p50 | p95 | p99 |
|-------|---------|-----|-----|-----|
| **embed** | sentence_transformers (fp32) | 7.23 ms | 13.20 ms | 34.99 ms |
| **embed** | onnx_optimized (fastembed) | **4.21 ms** | **5.05 ms** | **5.23 ms** |
| **search** | sentence_transformers | 1.04 ms | 1.42 ms | 1.81 ms |
| **search** | onnx_optimized | 1.43 ms | 2.00 ms | 2.33 ms |
| **total** | sentence_transformers | 8.31 ms | 14.34 ms | 36.17 ms |
| **total** | onnx_optimized | **5.72 ms** | **6.68 ms** | **7.13 ms** |

```
  embed    p50   7.23 ms -> 4.21 ms   +41.8%
  embed    p95  13.20 ms -> 5.05 ms   +61.7%
  search   p50   1.04 ms -> 1.43 ms   -38.3%
  total    p50   8.31 ms -> 5.72 ms   +31.2%
```

**Embedding latency drops ~42% at p50 and ~62% at p95.** The p95/p99 gap is
wider than p50 because the fp32 path has a much heavier tail — 35 ms at p99
versus 5 ms — so the optimised path is not just faster on average, it's far
more predictable, which is what actually matters for a latency SLO.

**Search gets slightly slower on the optimised index.** It reproduced at both 7
and 2,500 chunks. An earlier version of this README blamed int8 vectors being
less well separated, but the vectors are not int8 and are nearly identical to
the fp32 ones, so that explanation doesn't hold. The more likely cause is
variation between two separately built HNSW graphs. It costs about 0.4 ms and
saves about 3 ms on embedding, so the net is strongly positive. The honest
headline is **31% off embed + search combined**, not 42%.

The query cache is measured separately so its benefit is never folded into the
model comparison: a cache hit returns in ~0.0003 ms versus 4.21 ms for a miss.
That is a dictionary lookup versus a model call, which is why it's reported on
its own rather than blended into the headline.

Results are written to `bench/results/latest.json`.

**These numbers are hardware-specific.** Run it on your own machine and quote
what it produces — the point of shipping the harness is that the figure is
reproducible and defensible when someone asks how you measured it. The baseline
is skipped gracefully if `sentence-transformers` isn't installed; only the
optimised path is profiled then. Below ~1,000 chunks the benchmark warns that
the search-stage comparison is dominated by fixed overhead and carries no
signal; the embedding comparison is valid at any corpus size.

Useful flags:

| Flag | Effect |
|------|--------|
| `--runs N` | Timed queries per backend (default 30) |
| `--ef-search N` | HNSW `ef_search` for both runs — sweep it for the recall/latency curve |
| `--model NAME` | Any fastembed-supported HF model |
| `--skip-baseline` | Profile only the optimised path |
| `--queries FILE` | Newline-separated queries; defaults to corpus-derived |

---

## Configuration

Copy `.env.example` to `.env`. Every setting is an environment variable prefixed
`RAG_` (except `ANTHROPIC_API_KEY`).

| Variable | Default | Notes |
|----------|---------|-------|
| `ANTHROPIC_API_KEY` | — | Unset ⇒ extractive answers |
| `RAG_LLM_MODEL` | `claude-opus-5` | |
| `RAG_LLM_EFFORT` | `low` | `low`…`max`; keeps answer latency down |
| `RAG_LLM_THINKING` | `disabled` | `disabled` is only valid at effort ≤ `high` |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | 384-dim, strong quality/size ratio |
| `RAG_EMBED_BACKEND` | `onnx_optimized` | or `sentence_transformers` (`onnx_quantized` = deprecated alias) |
| `RAG_EMBED_MODEL_PATH` | *(empty)* | Local model directory for offline use; empty = download from the Hub |
| `RAG_EMBED_CACHE_SIZE` | `1024` | LRU over query embeddings; `0` disables |
| `RAG_HNSW_M` | `16` | Graph degree — build time vs recall |
| `RAG_HNSW_EF_CONSTRUCTION` | `200` | Index build quality |
| `RAG_HNSW_EF_SEARCH` | `64` | **The runtime recall/latency knob** |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `120` | Characters |
| `RAG_TOP_K` | `5` | Passages returned |
| `RAG_CANDIDATE_K` | `20` | Candidates fetched per search before narrowing to `top_k` |
| `RAG_MMR_LAMBDA` | `1.0` | `1.0` = pure relevance (MMR off), `0.0` = pure diversity; see [Retrieval quality](#retrieval-quality) |
| `RAG_HYBRID_SEARCH` | `false` | Add BM25 keyword search, fused with vector search |
| `RAG_KEYWORD_WEIGHT` | `0.5` | BM25's weight in the rank fusion (vector list = 1.0) |
| `RAG_RRF_K` | `60` | Rank-fusion constant; higher flattens the rank curve |
| `RAG_MAX_SUBQUERIES` | `3` | Decomposition cap |
| `RAG_TEXT_COLUMNS` | *(auto)* | Force specific CSV/JSON columns |

---

## Docker

```bash
docker compose up -d --build
```

Then open **[http://localhost:8000/docs](http://localhost:8000/docs)**.

The image bakes the embedding model in at build time, so the container starts
without network access and the first request isn't slowed by a download —
startup to ready is under a second. The vector store lives in a named volume
and survives restarts; `./data/docs` is mounted read-only at `/corpus`:

```bash
curl -s localhost:8000/ingest \
  -H 'content-type: application/json' \
  -d '{"path": "/corpus", "source_tag": "my-dataset"}'
```

Runs as a non-root user (uid 10001) with a `HEALTHCHECK` on `/health`. Image is
~1.2 GB, dominated by ONNX Runtime, ChromaDB, and the baked-in model.

> **Don't benchmark inside the container on Apple Silicon.** Docker Desktop runs
> a linux/aarch64 VM, and ONNX Runtime logs `Unknown CPU vendor` there — it
> can't detect CPU features, so it falls back to unoptimised kernels. Measured
> query latency is roughly 4× native as a result (≈57 ms vs ≈4 ms to embed).
> That's a virtualisation artefact, not a property of the service. Run
> `bench/benchmark.py` natively; use the container for deployment.

Tear down with `docker compose down` (add `-v` to drop the vector store too).

---

## Testing

```bash
make test     # pytest
make lint     # ruff
```

The suite runs fully offline — it uses a deterministic hashing embedder rather
than downloading a model, so it's fast, hermetic, and tests retrieval logic
rather than a Hugging Face download. Coverage spans every loader, chunk
determinism and dedup, MMR (including a check that the vectorised version matches
the textbook loop on 200 random cases), weighted rank fusion, the BM25 index and
its sync with ChromaDB, hybrid retrieval, the planner, the agent's retry path
(with a regression test for the fusion-score bug), the embedding cache, and the
full HTTP surface.

---

## Design notes

**Why chunk IDs are content hashes.** `sha256(source + text)` makes ingestion
idempotent. Re-running over a corpus that gained three files upserts everything
and duplicates nothing, so there's no separate "have I indexed this?"
bookkeeping to get wrong.

**Why rank fusion, and why MMR is opt-in.** Reciprocal rank fusion merges
sub-query results, and vector plus keyword results, by rank rather than raw score,
so a list with systematically higher scores can't dominate the merge. MMR was
added on the intuition that plain top-k fills the context window with one idea
stated five ways. The SciFact evaluation didn't support that intuition: the
diversity it bought cost 5 points of nDCG@10, so it's now off by default and
available for corpora with many near-duplicate passages. Measuring it is what
turned a plausible design choice into a known trade-off.

**Why the BM25 index is in memory.** It's rebuilt from ChromaDB at startup and
updated on every upsert, so the two indexes can't drift apart, and it needs no
extra service. That fits tens of thousands of chunks. Past a few hundred thousand,
a real search engine (OpenSearch, or Postgres full-text search) is the better home
for the keyword side.

**Why the agent loop is capped at two rounds.** It inspects its own retrieval
quality and retries with a broadened query when the best hit scores below
threshold — that's the adaptive part. But an unbounded agent loop is a latency
bug waiting to happen, so worst case is two rounds and the tail stays
predictable.

**Why the planner is rule-based.** Query decomposition sits on the hot path of
every request. An LLM round-trip there would cost more latency than the extra
recall is worth, and it keeps retrieval fully functional with no API key.

**Why thinking is disabled on the generation call by default.** Retrieval has
already done the reasoning; the model's job is to synthesise from passages in
front of it. Thinking tokens also count against `max_tokens` on current models,
so leaving it on both slows the response and eats the budget the answer needs.
Set `RAG_LLM_THINKING=adaptive` for genuinely hard synthesis.

**Why an extractive fallback exists.** A portfolio repository that requires a
paid API key before it does anything is a repository nobody runs. The fallback
keeps retrieval quality, the benchmark, and the API surface fully evaluable
without one.

---

## License

MIT
