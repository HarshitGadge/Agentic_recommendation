# Classic-text notice parsing

GCN "classic" notices are newline-delimited `KEY: value` records formatted for
human readers. A complete Fermi GBM Final Position notice looks like this:

```
TITLE:           GCN/FERMI NOTICE
NOTICE_DATE:     Wed 13 Aug 25 04:12:33 UT
NOTICE_TYPE:     Fermi-GBM Final Position
RECORD_NUM:      52
TRIGGER_NUM:     776751152
GRB_RA:          214.517d {+14h 18m 04s} (J2000),
                 214.856d {+14h 19m 25s} (current),
                 213.869d {+14h 15m 28s} (1950)
GRB_DEC:         -11.300d {-11d 18' 00"} (J2000),
                 -11.437d {-11d 26' 13"} (current),
                 -11.036d {-11d 02' 09"} (1950)
GRB_ERROR:       1.75 [deg radius, statistical only]
GRB_DATE:        20900 TJD;   225 DOY;   25/08/13
GRB_TIME:        14712.00 SOD {04:05:12.00} UT
SUN_DIST:         73.44 [deg]   Sun_angle= -4.7 [hr] (West of Sun)
GAL_COORDS:      337.06,  47.88 [deg] galactic lon,lat of the burst
COMMENTS:        Fermi-GBM Final Position.
COMMENTS:        This is a GRB.
```

## Why not `str_to_map`

Splitting the payload on newlines and colons is the obvious approach and it is
wrong in four ways:

**Values contain colons.** `GRB_TIME` embeds `{04:05:12.00}` and `LOC_URL` is an
`http://` address. Splitting on every colon truncates both. The parser splits on
the **first** colon only.

**Values wrap.** `GRB_RA` and `GRB_DEC` run across three physical lines, and the
continuations carry no key. They are recognised by indentation — a key line
matches `^[A-Z][A-Z0-9_]*:` anchored at column zero, anything else is a
continuation and is folded onto the preceding value.

**Keys repeat.** A notice may carry any number of `COMMENTS:` lines. A map keyed
by field name keeps one and loses the rest; the parser joins them in order.

**Non-breaking spaces appear in the padding.** `U+00A0` survives an ordinary
`strip()`, leaving values that compare unequal to their visible text. They are
normalised to spaces before anything else.

## Typed extraction

`fields.py` turns prose values into numbers and timestamps:

| Field | Raw | Parsed |
|---|---|---|
| `GRB_RA` | `214.517d {...} (J2000), 214.856d ... (current)` | `214.517` (J2000, first epoch) |
| `GRB_ERROR` | `1.75 [deg radius, statistical only]` | `1.75` |
| `SUN_DIST` | `73.44 [deg]   Sun_angle= -4.7 [hr]` | `73.44` (not `-4.7`) |
| `GAL_COORDS` | `337.06,  47.88 [deg] galactic lon,lat` | `(337.06, 47.88)` |
| `SUN_POSTN` | `143.21d {...}  +14.32d {...}` | `(143.21, 14.32)` |
| `GRB_DATE` + `GRB_TIME` | `20900 TJD` + `14712.00 SOD` | `2025-08-13T04:05:12Z` |

TJD (Truncated Julian Date) 0 is 1968-05-24. It is used in preference to the
`25/08/13` calendar date beside it because it carries no century ambiguity, and
seconds-of-day is used in preference to the `{04:05:12.00}` gloss because it
keeps sub-second precision.

Every extractor returns `None` rather than raising, so one malformed value nulls
one column instead of failing a micro-batch.

## Running on Spark

The parser is plain `str` handling with no Spark import, which is what makes it
unit-testable. It reaches the cluster through an Arrow-vectorised pandas UDF
(`spark_udf.py`): the JVM hands over a whole column per call, so serialisation
cost is paid once per batch rather than once per notice.

The UDF is unpickled inside a Python worker process on each executor, which does
not inherit the driver's `sys.path`. The package must therefore be installed on
the cluster, or shipped with `gcn_lakehouse.deploy.ship_to_executors`. Without
that, the job fails with `ModuleNotFoundError` on the first batch — after the
stream has already started.

## Known limits

- A field label containing lowercase letters or leading whitespace would be read
  as a continuation of the previous field.
- `is_parsed` is `False` only when no `TRIGGER_NUM` could be read. A notice that
  parses structurally but carries garbage values is written to Silver with nulls.
- Only the first epoch of a coordinate field is retained; the current and 1950
  positions are discarded after parsing.
