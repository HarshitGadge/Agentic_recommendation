# RAGBench

Evaluate any retriever on [BEIR](https://github.com/beir-cellar/beir)-format data and
get numbers you can defend: nDCG@10, Recall@10/100 and MRR@10 per query, a bootstrap
95% confidence interval on every mean, and a paired bootstrap when you compare two runs.

- **Bring your own retriever.** Anything with a `search(query, k)` method plugs in,
  from a module or a single `.py` file. No base class, no framework.
- **No runtime dependencies.** The core is the Python standard library.
- **Checked against `trec_eval`.** The metric tests compare every score with
  `pytrec_eval` on randomised runs with graded judgements.
- **Reproducible results.** Each result file records the exact command, parameters,
  seed, Python and RAGBench versions, and every per-query score.

Every number in this README comes from a result file in [`results/`](results), produced
by [`scripts/benchmark.sh`](scripts/benchmark.sh) in the Benchmark workflow. The table
below is generated from those files, and CI fails if it drifts.

## Results

Built-in retrievers on three BEIR test sets: SciFact, NFCorpus
and ArguAna. Dense uses `BAAI/bge-small-en-v1.5` through fastembed with no query
instruction; BM25 uses k1=0.9, b=0.4. Latency is per query on a GitHub-hosted runner,
and Dense batches its queries, so its p50 is the batch time divided by the query count.

<!-- results:start -->
| dataset | retriever | ndcg@10 | recall@10 | recall@100 | mrr@10 | queries | p50 ms |
|---|---|---|---|---|---|---|---|
| arguana | bm25 | 0.4052 (0.387 to 0.424) | 0.6814 (0.657 to 0.707) | 0.9317 (0.918 to 0.945) | 0.3203 (0.302 to 0.340) | 1406 | 10.10 |
| arguana | bm25-stem | 0.4012 (0.382 to 0.420) | 0.6643 (0.639 to 0.689) | 0.9324 (0.919 to 0.946) | 0.3198 (0.301 to 0.339) | 1406 | 13.53 |
| arguana | dense-bge-small | 0.5973 (0.578 to 0.616) | 0.8535 (0.835 to 0.871) | 0.9844 (0.978 to 0.991) | 0.5166 (0.495 to 0.538) | 1406 | 182.74 |
| nfcorpus | bm25 | 0.3061 (0.272 to 0.340) | 0.1466 (0.122 to 0.173) | 0.2360 (0.208 to 0.266) | 0.5106 (0.462 to 0.561) | 323 | 0.14 |
| nfcorpus | bm25-stem | 0.3208 (0.287 to 0.355) | 0.1523 (0.127 to 0.179) | 0.2461 (0.218 to 0.276) | 0.5231 (0.474 to 0.572) | 323 | 0.22 |
| nfcorpus | dense-bge-small | 0.3385 (0.306 to 0.373) | 0.1609 (0.137 to 0.187) | 0.3057 (0.277 to 0.336) | 0.5305 (0.484 to 0.577) | 323 | 3.21 |
| scifact | bm25 | 0.6635 (0.620 to 0.710) | 0.7834 (0.737 to 0.829) | 0.8859 (0.851 to 0.922) | 0.6316 (0.585 to 0.681) | 300 | 0.91 |
| scifact | bm25-stem | 0.6804 (0.635 to 0.724) | 0.8030 (0.758 to 0.846) | 0.9220 (0.891 to 0.951) | 0.6475 (0.600 to 0.694) | 300 | 1.40 |
| scifact | dense-bge-small | 0.7201 (0.677 to 0.760) | 0.8452 (0.806 to 0.883) | 0.9533 (0.927 to 0.977) | 0.6846 (0.637 to 0.728) | 300 | 9.41 |


Means over test queries with 95% bootstrap intervals; ragbench 0.1.0, commit 16ed3d5, workflow run [45932384237](https://github.com/HarshitGadge/ragbench/actions/runs/45932384237).


Paired bootstrap, dense minus BM25 (stemmed), same queries:

- arguana, ndcg@10: +0.1961 (95% CI +0.1776 to +0.2148, 1406 queries, interval excludes 0)
- arguana, recall@100: +0.0519 (95% CI +0.0391 to +0.0647, 1406 queries, interval excludes 0)
- nfcorpus, ndcg@10: +0.0177 (95% CI -0.0040 to +0.0390, 323 queries, interval includes 0)
- nfcorpus, recall@100: +0.0596 (95% CI +0.0405 to +0.0795, 323 queries, interval excludes 0)
- scifact, ndcg@10: +0.0397 (95% CI +0.0051 to +0.0730, 300 queries, interval excludes 0)
- scifact, recall@100: +0.0313 (95% CI -0.0013 to +0.0660, 300 queries, interval includes 0)
<!-- results:end -->

Reproduce:

```bash
pip install -e '.[dense,stem,hf]'
bash scripts/benchmark.sh
python scripts/update_readme.py
```

## Install

```bash
pip install "git+https://github.com/HarshitGadge/ragbench@v1.0.0"

# with the built-in dense retriever, stemming and the Hugging Face download fallback
pip install "ragbench[dense,stem,hf] @ git+https://github.com/HarshitGadge/ragbench@v1.0.0"
```

The core package has no runtime dependencies. Extras: `dense` (fastembed, numpy),
`stem` (snowballstemmer), `hf` (pyarrow).

## Quick start

```bash
ragbench info                       # datasets worth starting with
ragbench download scifact           # into ./data/scifact
ragbench run --data data/scifact --retriever examples/overlap.py:WordOverlap
```

`run` prints a Markdown table and writes `results/<dataset>__<name>.json`.

## Built-in retrievers

Two baselines ship with RAGBench, and they double as examples of the interface.

| retriever | spec | needs |
|---|---|---|
| BM25 (Lucene scoring, k1=0.9, b=0.4) | `ragbench.retrievers:BM25` | nothing |
| Dense, cosine over normalised embeddings | `ragbench.retrievers:Dense` | `pip install 'ragbench[dense]'` |

```bash
ragbench run --data data/scifact --retriever ragbench.retrievers:BM25 --param stem=true
ragbench run --data data/scifact --retriever ragbench.retrievers:Dense \
  --param model=BAAI/bge-small-en-v1.5
```

BM25 takes `k1`, `b`, `stopwords` (Lucene's English list, on by default) and `stem`
(Snowball, `pip install 'ragbench[stem]'`). Its tokeniser is simpler than an
Elasticsearch analyser, so its scores will not match Elasticsearch-based BEIR
baselines exactly. Dense takes `model` (any [fastembed](https://github.com/qdrant/fastembed)
text model), `query_prefix` for models that expect one, and `batch_size`; from Python
you can pass `encoder=` with any object that has `embed_documents` and `embed_query`.

## Plug in a retriever

```python
# my_retriever.py
class MyRetriever:
    def __init__(self, top_n: int = 100):
        self.top_n = top_n

    def index(self, corpus):  # optional: {doc_id: Document(id, title, body)}
        ...

    def search(self, query: str, k: int):
        return [("doc-17", 12.3), ("doc-4", 11.0)]  # ids, Hit objects or (id, score)
```

```bash
ragbench run --data data/scifact --retriever my_retriever.py:MyRetriever --param top_n=50
```

`--param key=value` values are parsed as JSON when they can be (`0.5`, `true`, `[1,2]`).
A retriever can also define `search_batch(queries, k)`, which RAGBench uses instead of
`search` when present. Returning several chunks of the same document is fine: a
document counts once, at the rank of its best chunk.

From Python:

```python
from ragbench import load_beir, evaluate

dataset = load_beir("data/scifact")
result = evaluate(MyRetriever(), dataset, name="mine")
print(result.summary()["ndcg@10"])  # {'mean': ..., 'ci': [low, high], 'level': 0.95}
result.save("results/scifact__mine.json")
```

## Compare two runs

```bash
ragbench compare results/scifact__baseline.json results/scifact__mine.json --metric ndcg@10
```

The difference is resampled over queries (2,000 rounds, seeded), so the interval says
whether the change would survive a different sample of questions. Both runs must cover
the same queries.

## Make a results table

```bash
ragbench report results/*.json
```

## How the metrics are computed

| metric | definition |
|---|---|
| nDCG@10 | DCG with linear gains (the judgement grade) and log2 discounts, divided by the ideal DCG; the `ndcg_cut.10` that `trec_eval` and BEIR report |
| Recall@k | relevant documents in the top k, over all relevant documents |
| MRR@10 | 1 / rank of the first relevant document in the top 10, else 0 |

Judgements of 0 count as not relevant, and queries without a relevant document are
skipped, as in BEIR.

## Development

```bash
pip install -e '.[dev,stem]' numpy pytrec-eval-terrier
ruff check . && ruff format --check . && pytest
```

## Origin

The loader, metrics and bootstrap started as the retrieval evaluation in
[agentic-rag-service](https://github.com/HarshitGadge/agentic-rag-service) and were
generalised here so they are not tied to one service.

## License

MIT
