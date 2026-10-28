# PySpark Performance Tuning
[![CI](https://github.com/faishalnuruz/pyspark-performance-tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/faishalnuruz/pyspark-performance-tuning/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A naive PySpark job and an optimised one over the same skewed, nested telemetry — with the
measurement that proves the difference and the check that proves the answers still match.

Most "performance tuning" write-ups show a before/after screenshot of a progress bar. This one
prints numbers a reviewer can regenerate: how long each stage took, how badly the work was
distributed across tasks, and how many files a partitioned filter has to touch.

```text
end to end        naive    37.1s   optimised    33.5s
  read+flatten    naive    25.6s   optimised     8.4s
  aggregate       naive     7.8s   optimised     3.2s
schema inference  inferSchema     3.3s   explicit     0.8s
task skew         hash by key 9.84   hash by key+salt 1.74
results identical True (8 events compared)
layout            54 files in 6 partition directories; one country needs 9 of them
filter effect     484,750 rows for one country vs 3,023,500 unfiltered
```

## The data, and why it is shaped this way

3,023,500 per-second telemetry points across 715 activities, written as nested JSON: each activity
holds an array of samples inside an `event` struct and an `athlete` struct. One ultra-distance race
holds **66.1% of all rows**, which is the whole point — a single join key that dwarfs every other
key is what makes a hash shuffle collapse onto one task.

```bash
python -m spark_tuning.generate                  # ~340 MB, seeded and reproducible
python -m spark_tuning.generate --scale 0.05     # a fast smoke dataset
```

## What changes between the two jobs

| | Naive | Optimised |
| --- | --- | --- |
| Read | `inferSchema` (reads the file to guess types, then again to parse) | Explicit `StructType` written down in `schema.py` |
| Join | Shuffle join against the event dimension | `broadcast()` the dimension — it is 8 rows |
| Aggregation | `hash by event_id` — the hot key lands in one task | Salted two-stage: group by `(event_id, salt)`, then re-aggregate |
| Partitions | Left at the default | Pinned, with adaptive query execution enabled |
| Write | Not written | Parquet partitioned by year, month, country |

## Results

Measured on the full dataset with 8 shuffle partitions and 16 salt buckets:

- **Schema inference alone costs 4.1x:** `inferSchema` 3.3s versus an explicit schema at 0.8s, on
  the same files.
- **Reading and flattening: 3.0x faster** (25.6s → 8.4s). Most of the win is the inference pass
  disappearing; the rest is the flattening work happening once instead of twice.
- **Aggregation: 2.4x faster** (7.8s → 3.2s) with the dimension broadcast and the hot key salted.
- **The pipeline stages together: 2.9x faster** (33.4s → 11.6s). The end-to-end totals above are
  only 10% apart because both runs also spend time in the benchmark's own distribution
  measurement, which is reported separately rather than hidden.
- **Task skew drops from 9.84 to 1.74**: the worst task went from 2,126,000 rows to 610,160 while
  the median task rose from 216,000 to 351,052. That is the difference between one task doing 10x
  the median work and one task doing under 2x.
- **The results are identical** across all 8 events — counts exactly, averages within 0.01 — which
  is asserted, not assumed.
- **The partitioned write makes later queries cheap**: 54 files in 6 partition directories, and a
  query filtered to one country opens 9 of them and reads 484,750 rows instead of 3,023,500.

## How the numbers are produced

`python -m spark_tuning.benchmark` runs both pipelines, and for each one records:

- **wall-clock seconds per stage**, so a change can be attributed to reading, aggregating, or
  writing;
- **the row count per task** (`spark_partition_id()` grouped and counted), which yields
  `skew_ratio = max task rows / median task rows`. This is measurable and reviewable, unlike a
  screenshot of a progress bar;
- **the correctness comparison** between the two outputs, because an optimisation that changes the
  answer is a bug with a nicer runtime;
- **the partition layout on disk** plus real row counts through Spark, which is how pruning is
  evidenced without relying on planner internals.

Results are written to `docs/generated/benchmark.json` so a claim in this README can be traced to a
run.

## Tests

```bash
.venv/bin/python -m pytest -q     # 22 passed (~2 min on a quiet machine, one JVM for the whole suite)
.venv/bin/ruff check .
```

They cover the nested flattening and struct lifting, the generator's deliberate skew, the salted
aggregation agreeing with the unsalted one (including the weighted-average recombination), the
skew measurement itself (a synthetic hot key must produce a high ratio, and salting must lower it),
the partitioned layout, and the comparison harness — including a test that feeds it a deliberately
tampered result to prove the check can fail.

## Deliberate limitations

- **Local mode, one machine.** `local[*]` with a single JVM demonstrates skew and shuffle behaviour,
  but not network-bound shuffle or executor death. The numbers are honest for the machine they ran
  on and are not cluster benchmarks.
- **Skew is measured per task, not per shuffle byte.** Spark's shuffle read/write sizes live behind
  the listener API and are not captured here.
- **Averages are compared within 0.01.** The salted path recombines sums, so last-digit differences
  are expected; counts are compared exactly.
- **The dimension is tiny by construction** (8 rows), which is what makes broadcasting correct. The
  job does not demonstrate the case where the "small" table is too big to broadcast.
- **The dataset is synthetic.** Its skew is deliberate and its shape is documented, but no real
  telemetry pipeline sits behind it.

## Next

1. Capture shuffle read/write bytes and spill-to-disk via a `SparkListener` so the cost of the
   naive path is visible in I/O as well as time.
2. Add the same comparison with `repartitionByRange` and a salted join to cover the "large table
   joins large table" case that broadcasting cannot fix.
3. Run it on a real cluster and record whether the ratio improves or the bottleneck moves.
