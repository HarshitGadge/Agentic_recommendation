# Design

## Two pipelines, one domain

| | Streaming lakehouse | Batch warehouse |
|---|---|---|
| Source | Postgres CDC via Debezium | FX REST API + Postgres extract |
| Transport | Kafka | Object storage (S3) |
| Compute | Spark Structured Streaming | Snowflake + dbt |
| Storage | Iceberg | Snowflake tables |
| Grain | one payment, continuously corrected | one payment, converted to USD |
| Latency | seconds to minutes | daily |
| Answers | "what is happening now" | "what did we earn, in one currency" |

They are not redundant. The streaming side must reflect a refund within
minutes; the batch side must produce a number finance can close a month on,
which means stable historical FX and a full restatement window.

## Layer contracts

**Bronze** stores the Kafka message verbatim with its topic, partition and
offset. It never parses. A CDC topic has finite retention, so once an event ages
off the broker Bronze is the only copy — keeping it raw means a decoder fix is
replayed from Bronze instead of re-snapshotting Postgres, which on a large table
means hours of load on the production database.

**Silver** is one row per payment, current state, deletes applied. Anything that
fails to decode goes to a quarantine table with its payload rather than being
dropped, so Silver row counts stay reconcilable against Bronze.

**Gold** is recomputed, not appended. A CDC source mutates history — a payment
authorised last week can be refunded today, changing last week's revenue — so an
append-only gold layer drifts from silver within days.

## The CDC merge

Four properties, one MERGE clause each. See
[`src/payments_platform/merge.py`](../src/payments_platform/merge.py).

```sql
MERGE INTO silver.payments AS t
USING cdc_batch AS s ON t.payment_id = s.payment_id
WHEN MATCHED AND s.is_deleted             THEN DELETE
WHEN MATCHED AND s.cdc_lsn > t.cdc_lsn    THEN UPDATE SET ...
WHEN NOT MATCHED AND NOT s.is_deleted     THEN INSERT ...
```

1. A delete must delete.
2. A replay must not move the table backwards — hence the LSN predicate.
3. The source must hold one row per key. Iceberg fails a MERGE whose source
   matches a target row more than once, deliberately, since the result would be
   non-deterministic. A CDC batch routinely holds several changes to one
   payment, so it is collapsed by LSN first.
4. A delete for a row never seen inserts nothing.

### Why LSN and not the timestamp

`source.ts_ms` has millisecond resolution. Two updates to the same payment
inside one millisecond tie, and a tie broken the wrong way applies a stale image
over a fresh one. The LSN is Postgres's own write-ahead-log position and is
strictly increasing.

### Why REPLICA IDENTITY FULL

Under the Postgres default, the WAL records only the primary key for the old
row, so Debezium's `before` image on a delete has one populated column. The
source tables are set to `REPLICA IDENTITY FULL` so a delete arrives with the
whole row. The cost is WAL volume; the alternative is a delete event you cannot
do anything with.

## FX conversion

```
published rates ──▶ grid of (currency × dates in the data)
                 ──▶ forward-fill  ──▶ effective_rate per currency per date
                 ──▶ equi-join to payments on (currency, payment_date)
```

The date spine is built from dates present in the data rather than a generated
calendar, which keeps the SQL portable — Snowflake's `GENERATOR` and DuckDB's
`generate_series` have no common spelling — and generates no rows for dates
nothing happened on.

`is_carried_forward` means an older published rate was reused. It is false when
there was no rate to reuse at all: a date before the feed began, or a currency
the feed does not cover. Those rows are flagged `is_unconverted` and excluded
from revenue totals, with the excluded count reported alongside so the gap is
visible rather than silently understating revenue.

## Incremental strategy

Spark jobs use `trigger(availableNow=True)`: each run drains what has
accumulated since the last checkpoint and exits, so they can be scheduled rather
than held open.

`fct_payments_usd` is a dbt incremental model that re-processes a trailing
window (`fx_restatement_days`, default 7) rather than only new rows — a rate
published late changes the USD value of payments already loaded.

## What money is, in types

| Boundary | Type |
|---|---|
| Postgres | `NUMERIC(12,2)` |
| Kafka wire | base64 unscaled integer |
| Python | `decimal.Decimal` |
| Spark | `DECIMAL(12,2)` |
| Snowflake | `NUMBER(18,8)` for rates, `DECIMAL` for amounts |
| JSON response | **string** |

`float` appears nowhere in that chain. The one place a string carrier is used
deliberately is the Spark decode UDF: Arrow's decimal128 round-trip through
pandas is fragile across pyarrow versions, so the UDF returns the digits as a
string and casts to `DECIMAL` in SQL immediately after, which is exact.
