# Live demo: portfolio Q&A

The public deployment of this service answers questions about my own projects.
Its index holds the READMEs and design notes of six repositories, one source
tag each:

| `source_filter`             | Project |
|-----------------------------|---------|
| `agentic-rag-service`       | this service |
| `capability-runtime`        | LLM-discovered UI flows replayed deterministically |
| `payments-data-platform`    | Debezium CDC → Spark → Iceberg lakehouse, FX warehouse in dbt |
| `kafka-databricks-nasa`     | NASA GCN Kafka stream → Databricks medallion lakehouse |
| `medicare-claims-analytics` | dbt + DuckDB claims pipeline, readmissions, high-cost model |
| `ragbench`                  | RAG retrieval evaluation harness: BEIR, nDCG@10, bootstrap CIs |

Questions worth trying:

```bash
URL=https://agentic-rag-service-fxsc.onrender.com

curl -s $URL/query -H 'content-type: application/json' \
  -d '{"question": "How does the payments pipeline stop a replayed Kafka event from moving the table backwards?"}' | jq .answer

curl -s $URL/query -H 'content-type: application/json' \
  -d '{"question": "Why is MMR off by default?", "source_filter": "agentic-rag-service"}' | jq

curl -s $URL/query -H 'content-type: application/json' \
  -d '{"question": "How are hospital readmissions defined and how are transfers handled?"}' | jq .citations

curl -s $URL/query -H 'content-type: application/json' \
  -d '{"question": "How much better is dense retrieval than BM25 on ArguAna?", "source_filter": "ragbench"}' | jq .answer
```

Or open [the interactive API docs](https://agentic-rag-service-fxsc.onrender.com/docs) and use **Try it out**.

## How the deployment differs from a local run

- **Read-only.** `RAG_ALLOW_INGEST=false`, so `/ingest` and `/ingest/upload`
  return 403. Both write to the index, and `/ingest` reads server paths, so
  neither belongs on an open URL.
- **Index baked into the image.** With `RAG_DEMO_INDEX=true` the Dockerfile runs
  `demo/build_index.sh` during the build. A free instance that sleeps and wakes
  up again starts with the index already in place instead of re-embedding.
- **Extractive answers.** No `ANTHROPIC_API_KEY` is set on the public service,
  so answers are the most relevant sentences from the retrieved passages, with
  citations. Retrieval, the agent loop and timings are identical to a run with
  a key.
- **Cold starts.** On Render's free plan the service sleeps after 15 minutes
  without traffic; the first request after that takes about a minute.

## Refreshing the corpus

`demo/corpus/` is a snapshot. To refresh it, copy each project's `README.md`
and `docs/*.md` into `demo/corpus/<project>/` and push; Render rebuilds the
image and the index on every push to `main`.
