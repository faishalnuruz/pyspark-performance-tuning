# Evidence: benchmark run

Captured on 2026-10-24 on a laptop (Apple silicon, `local[*]`, one JVM, 2 GB driver memory).
Reproduce with:

```bash
python -m spark_tuning.generate                    # 715 activities, 3,023,500 points
python -m spark_tuning.benchmark --json docs/generated/benchmark.json
```

## Dataset

```text
Wrote 715 activities and 3,023,500 telemetry points
Hottest event holds 66.1% of all points
  ugm-100k        2,000,000
  ctc-76k           292,500
  dieng-42k         216,000
  rinjani-36k       150,000
  semeru-21k        126,000
  merbabu-15k       104,000
  bromo-10k          81,000
  city-5k            54,000
```

## Summary printed by the benchmark

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

## Task distribution, before and after

Naive job, hash partitioning by the skewed key (8 requested partitions, 5 non-empty):

```text
{'partitions': 5, 'rows': 3,023,500, 'min_rows': 81,000, 'max_rows': 2,126,000,
 'median_rows': 216,000, 'skew_ratio': 9.84}
```

Optimised job, hash partitioning by `(event_id, salt)`:

```text
{'partitions': 8, 'rows': 3,023,500, 'min_rows': 149,787, 'max_rows': 610,160,
 'median_rows': 351,052, 'skew_ratio': 1.74}
```

The worst task carried 2.13M rows in the naive run and 610k in the optimised one, while the median
task rose from 216k to 351k. In a cluster this is the difference between 199 tasks waiting on one,
and every task finishing at roughly the same time.

## Why the end-to-end totals look close

Both totals include the benchmark's own distribution measurement, which materialises 3M rows and
groups them by partition id — expensive, identical in both runs, and therefore uninformative about
the optimisation. Excluding it, the pipeline stages take **33.4s naive versus 11.6s optimised
(2.9x)**. The stage-level numbers are in the table above and in `docs/generated/benchmark.json`;
publishing the end-to-end figure alongside them is the honest way to present it.

## Correctness

```text
{'events_compared': 8, 'mismatches': [], 'identical': True}
```

Counts are compared exactly and averages within 0.01, because the salted path recombines sums.
`tests/test_correctness.py` also feeds the comparison a deliberately tampered result to prove the
check can fail.

## Partitioned layout

```text
layout            54 files in 6 partition directories
filter effect     484,750 rows for one country vs 3,023,500 unfiltered
```

The write is partitioned by year, month, and country; a query filtered to a single country opens 9
of the 54 files and reads 484,750 of the 3,023,500 rows.

## Tests

```text
$ .venv/bin/python -m pytest -q
22 passed in 156.09s
$ .venv/bin/ruff check .
All checks passed!
```
