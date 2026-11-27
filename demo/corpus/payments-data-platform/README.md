# Payments Data Platform

[![tests](https://github.com/HarshitGadge/payments-data-platform/actions/workflows/tests.yml/badge.svg)](https://github.com/HarshitGadge/payments-data-platform/actions/workflows/tests.yml)

A payments data platform running two pipelines over one domain: a **streaming
CDC lakehouse** for operational analytics, and a **batch FX warehouse** for
cross-currency financial reporting, with a **FastAPI serving tier** over the
gold layer.

```
Postgres ──▶ Debezium/Kafka ──▶ Spark ──▶ Iceberg ──▶ Trino ──▶ FastAPI
                                                                    
FX REST API ──────────────▶ S3 ──▶ Snowflake ──▶ dbt ──▶ star schema
```

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Spark-Structured%20Streaming-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Apache Iceberg](https://img.shields.io/badge/Apache-Iceberg-1E90FF)](https://iceberg.apache.org/)
[![Debezium](https://img.shields.io/badge/Debezium-CDC-red)](https://debezium.io/)
[![dbt](https://img.shields.io/badge/dbt-Snowflake-FF694B?logo=dbt&logoColor=white)](https://www.getdbt.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-102%20passing-3fb950)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Demo

Everything below is real output from [`demo.py`](demo.py), which runs the actual
decoding, merge, FX and serving code — no Postgres, Kafka, Trino, Snowflake or
cloud account needed:

```bash
pip install -r requirements-dev.txt
python demo.py
```

<p align="center">
  <img src="docs/img/demo.svg" alt="demo.py output: a real Debezium message decoded to the exact source value, FX carry-forward across a weekend, a CDC merge into Iceberg, and the serving API" width="100%">
</p>

The first section decodes a message
[captured from a real Debezium connector](tests/fixtures/debezium_snapshot_real.json)
and checks it against the row it came from in Postgres. The rest runs the FX
carry-forward, a CDC merge into a real Iceberg table (including a replayed stale
LSN that must be ignored), and the serving API.

## The problems this is built around

Most of the code exists because of four specific ways a payments CDC pipeline
corrupts data. Each one has a test named after it.

**Money arrives base64-encoded.** Under Debezium's default
`decimal.handling.mode=precise`, a Postgres `NUMERIC(12,2)` is sent as a base64
string holding the unscaled integer as big-endian two's-complement bytes. A real
message captured from this repo's own stack:

```json
{"payment_id": 1001, "amount": "EHU=", "created_at": 1790395387456703}
```

`EHU=` is `42.13`. Read as a string it is nonsense; cast to float it raises. The
decoder in [`src/payments_platform/cdc.py`](src/payments_platform/cdc.py)
decodes it to `Decimal` — never `float`, because binary floating point cannot
represent `0.10` and the error compounds into a reconciliation break.

**Timestamps are microseconds, not milliseconds.** `1790395387456703` is 16
digits. Divided by 1000 out of habit, every payment lands in 1970.

**Deletes carry the row in `before`, not `after`.** A job that only upserts the
`after` image leaves deleted payments in the warehouse forever. The source tables
are `REPLICA IDENTITY FULL` so the `before` image has real columns — under the
Postgres default it holds only the primary key.

**Replays must not move the table backwards.** Kafka offsets get replayed after a
failed batch. The MERGE guards on `cdc_lsn > target.cdc_lsn`, so a stale event
re-applied after a newer one is ignored. LSN, not `ts_ms`: the timestamp has
millisecond resolution and two changes in the same millisecond tie.

On the batch side, one more:

**FX markets close; payments do not.** There is no rate published for a Saturday.
[`int_fx_rates_daily`](snowflake_etl/dbt/models/intermediate/int_fx_rates_daily.sql)
carries the last published rate forward, so a Saturday payment converts at
Friday's rate. Conversion uses the rate **as of the payment date** — revaluing
history at today's rate makes last quarter's revenue move every morning. A
currency with no rate at all converts to `NULL` and is flagged, never defaulted
to `1.0`, which would report 100 GBP as 100 USD.

## Layout

```
src/payments_platform/    pure logic, no cluster needed to test
  cdc.py                  Debezium envelope decoding
  fx.py                   rate lookup, carry-forward, conversion
  transforms.py           Spark bindings (Arrow-vectorised UDF)
  merge.py                the CDC MERGE and its four guarantees
  schemas.py  config.py  session.py  deploy.py
pipelines/                bronze_cdc → silver_merge → gold_marts
api/                      FastAPI serving tier + repository layer
snowflake_etl/
  src/                    FX extract, Snowflake COPY/MERGE
  dbt/                    staging → intermediate → marts, 33 dbt checks
config/                   Postgres init, Debezium connector, Trino catalog
tests/                    102 tests, incl. real captured Debezium messages
docs/                     design notes
```

## Running the tests

```bash
pip install -r requirements-dev.txt

# Iceberg runtime jar, for the merge tests
curl -sSL -o /tmp/iceberg.jar \
  https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.5.2/iceberg-spark-runtime-3.5_2.12-1.5.2.jar

ICEBERG_JAR=/tmp/iceberg.jar pytest
```

Tests degrade rather than fail: without `ICEBERG_JAR` the merge tests skip,
without pyspark the Spark tests skip, and the pure decoding and FX tests need
nothing but Python.

The dbt models run on DuckDB, so the Snowflake SQL is genuinely executed:

```bash
cd snowflake_etl/dbt && DBT_PROFILES_DIR=. dbt build --target duckdb
```

## Running the stack

```bash
cp .env.example .env    # fill in POSTGRES_PASSWORD
docker compose up -d

curl -s -XPOST -H 'Content-Type: application/json' \
  --data @config/connect/payments-connector.json \
  http://localhost:8083/connectors
```

Compose refuses to start if a credential is unset, so the stack never comes up
on a default password. See [docs/DESIGN.md](docs/DESIGN.md) for layer contracts
and [docs/OPERATIONS.md](docs/OPERATIONS.md) for replay and backfill.

## API

```
GET /health
GET /v1/merchants/{id}/revenue?start_date=&end_date=
GET /v1/payment-methods/{date}?country_code=
```

Amounts serialise as JSON **strings**. A JSON number is an IEEE 754 double in
every mainstream client, so `1234567.89` comes back as `1234567.8899999999`.
Totals are reported per currency and never summed across them.

## What is verified, and what is not

Verified by running it: the CDC decoding (against messages captured from a real
Debezium 2.7 connector reading Postgres 16), the Iceberg MERGE semantics
(against a real Iceberg table), the gold aggregations, the API behaviour
(against DuckDB holding real rows), and the dbt models (executed on DuckDB).

Not verified end-to-end: the Snowflake load path, which needs a Snowflake
account. Its SQL is rendered and asserted in tests, but no statement has run
against Snowflake itself.

## License

MIT — see [LICENSE](LICENSE).
