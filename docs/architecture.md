# Architecture and decisions

## Flow

```text
nested telemetry JSON (one file per event, deliberately skewed)
        |
        +--> NAIVE job                                    OPTIMISED job
             inferSchema read                              explicit StructType read
             shuffle join on event_id                      broadcast join (dimension is 8 rows)
             groupBy(event_id)                             groupBy(event_id, salt) then re-aggregate
             default partitioning                          pinned partitions + AQE
             (not written)                                 write Parquet partitioned by year/month/country
        |                                                   |
        +----------------> compare ------------------------ +
                            counts exact, averages within 0.01
        |
        v
  measurement: seconds per stage, rows per task, skew ratio, files per partition
        |
        v
  docs/generated/benchmark.json + printed summary
```

## Decisions worth defending

**The naive job is a real job, not a straw man.** It infers the schema, shuffle-joins the dimension,
and hashes by the skewed key — exactly what a first draft does. It is written the way a competent
engineer would write it before profiling.

**Skew is measured, not asserted.** `skew_ratio = max task rows / median task rows` comes from
grouping `spark_partition_id()`. A progress-bar screenshot proves nothing a reviewer can check.

**Correctness is part of the benchmark.** Salting changes the physical plan, so the two outputs are
joined and compared: counts exactly, averages within a stated tolerance. A test tampers with one
result to prove the comparison can fail — otherwise it is decoration.

**The salted path recombines weighted averages explicitly.** Stage one emits sum and count per
`(key, salt)`; stage two divides. Recomputing a mean from per-bucket means would be wrong, and the
test asserts the two paths agree.

**Pruning is evidenced at the filesystem level.** `DataFrame.inputFiles()` does not reflect partition
pruning in a way that is stable to depend on, so the layout (files per partition directory) is
measured directly and the row counts are measured through Spark. Both halves are real; together they
support the claim.

**End-to-end totals are published next to stage totals.** The benchmark's own measurement stage is
expensive and identical in both runs, which makes the end-to-end difference look smaller (10%) than
the pipeline's (2.9x). Reporting only the flattering number would be dishonest; reporting only
end-to-end would hide the result, so both appear with an explanation.

**One JVM for the test suite.** Spark startup dominates a small suite, so the session is shared and
the test dataset is scaled to 2% of the real one.

## Failure modes and edge cases

| Situation | What happens |
| --- | --- |
| Telemetry array is empty or null | `explode` drops the activity; the left join keeps dimensions from erasing points |
| Event is missing from the dimension | Left join keeps the telemetry, `name` and `terrain` are null |
| Dataset generated at tiny scale | Smallest events clamp to a floor, so the hot-key share shifts; tests assert dominance rather than an exact ratio |
| Salt buckets exceed partitions | Distribution still improves, with some partitions receiving several buckets |
| Spark unavailable | The suite fails loudly at the first fixture; nothing silently skips |
| Parquet output already exists | The previous directory is removed before the partitioned write |
