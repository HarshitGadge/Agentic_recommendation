# Operations

## First run

The Debezium connector's `snapshot.mode=initial` reads every existing row as
`op='r'` before streaming changes. On a large payments table that snapshot holds
a read transaction open for its duration — schedule the first start accordingly.

## Replaying after a decoder change

```bash
# Silver and Gold only. Never clear the Bronze checkpoint.
spark-submit pipelines/silver_merge.py
spark-submit pipelines/gold_marts.py
```

Bronze retains the raw payloads, so a decoder fix is replayed from there. The
Kafka topic has finite retention: once an event ages off the broker it exists
only in Bronze, and clearing that checkpoint means re-snapshotting Postgres.

The MERGE is idempotent — the LSN guard ignores anything already applied — so a
replay over an overlapping window is safe.

## Replication slot hygiene

A Postgres replication slot retains WAL until the consumer confirms it. If the
connector stops, WAL accumulates and eventually fills the disk, taking the
source database down with it.

```sql
SELECT slot_name, active,
       pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn))
         AS retained
FROM pg_replication_slots;
```

The connector sets `heartbeat.interval.ms=10000` so a quiet period still advances
the confirmed LSN. Without heartbeats, a low-traffic table lets the slot stall
even though nothing is wrong.

## Monitoring

| Check | Healthy | Meaning when not |
|---|---|---|
| `silver.payments_quarantine` count | 0 | The source schema changed, or a decoder assumption broke |
| Connector task state | `RUNNING` | Check `/connectors/payments-cdc/status` for the trace |
| Slot `retained` | bounded | The consumer is behind; WAL is growing |
| `fct_payments_usd` where `is_unconverted` | ~0 | A rate feed gap, or a new currency with no rates |
| `carried_forward_count` | weekends only | A long run means the rate feed has stopped |

```sql
-- Unconverted payments by currency: the first thing to check when USD revenue
-- looks low.
SELECT currency, count(*), min(payment_date), max(payment_date)
FROM fct_payments_usd
WHERE is_unconverted
GROUP BY currency;
```

## Schema changes on the source

A new column appears in `after` and is ignored by the decoder, which projects a
fixed column list. Nothing breaks; the column is simply absent from Silver until
`records`/`transforms` is extended.

A **removed or renamed** column is different: the decoder writes `NULL` for it
and the pipeline keeps running. Quarantine will not catch this, because the
envelope still parses. The check that does catch it is a not-null assertion on
the columns that matter.

## Backfilling FX

```bash
python -m snowflake_etl.src.extract_fx  # writes newline-delimited JSON
python -m snowflake_etl.src.load_snowflake FX_STAGE fx/
cd snowflake_etl/dbt && dbt build --target snowflake
```

The Snowflake load is a MERGE on `(rate_date, base_currency, currency)`, so a
restated rate updates in place rather than appending a second row — which would
fan out every payment joined to it.
