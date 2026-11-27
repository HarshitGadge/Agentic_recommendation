# Deployment

## 1. GCN credentials

Register a client at <https://gcn.nasa.gov/quickstart>. You receive a client ID
and secret; the feed itself is public, but the broker requires authentication.

## 2. Databricks secret scope

```bash
databricks secrets create-scope gcn
databricks secrets put-secret gcn client-id
databricks secrets put-secret gcn client-secret
```

The scope and key names are the defaults in `gcn_lakehouse.config.SecretConfig`.
Never inline the credentials in a notebook — they land in the revision history.

## 3. Catalog objects

Run [`setup/create_catalog.sql`](../setup/create_catalog.sql) in a SQL warehouse
or notebook. It creates the catalog, the three schemas and the checkpoint volume.

## 4. Make the package importable on executors

The pandas UDF runs in a Python worker on each executor, which needs to import
`gcn_lakehouse`. Either install it on the cluster:

```python
%pip install git+https://github.com/HarshitGadge/kafka-databricks-nasa
dbutils.library.restartPython()
```

or, when running from a cloned repo, ship it at job start:

```python
from gcn_lakehouse.deploy import ship_to_executors
ship_to_executors(spark)
```

Skipping this step produces a `ModuleNotFoundError` raised from the worker on the
first batch, after the stream has started.

## 5. Cluster

- Databricks Runtime 14.3 LTS or later (Spark 3.5+, Delta, Unity Catalog).
- The `spark-sql-kafka-0-10` connector is included in the runtime — do not
  install `pyspark` on the cluster.
- Single node is sufficient; GCN publishes a handful of Fermi notices per day.
- Unity Catalog access mode, so the volume and catalog are reachable.

## 6. Job

Three tasks in sequence, each depending on the previous:

| Task | Script |
|---|---|
| `bronze_ingest` | `pipelines/bronze_ingest.py` |
| `silver_parse` | `pipelines/silver_parse.py` |
| `gold_model` | `pipelines/gold_dimensional.py` |

All three use `trigger(availableNow=True)`: each run drains what has accumulated
since the last checkpoint and exits, so the job can be scheduled (hourly is
ample) rather than run continuously.

## Operating notes

**First run.** `startingOffsets` defaults to `earliest`, so the first ingest
pulls GCN's full retained history. Set `StreamConfig.starting_offsets="latest"`
to start from now instead.

**Reprocessing after a parser change.** Delete the Silver and Gold checkpoints
and re-run from Bronze. Do not clear the Bronze checkpoint — GCN retains only a
bounded window, and anything aged off the broker exists nowhere else.

**Monitoring.** Check that quarantine stays empty and that `rescued_fields` is
null; both are early warning that the notice format has changed. The queries are
in [DATA_MODEL.md](DATA_MODEL.md#data-quality).

**Timezone.** `configure_session` pins `spark.sql.session.timeZone` to UTC. Do
not override it: `burst_hour_utc` and `burst_day_of_year` are derived with Spark
date functions, which read that setting.
