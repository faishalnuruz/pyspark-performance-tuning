"""The naive job and the optimised job, measured on the same data.

The pair exists to make the difference visible and reproducible:

* naive: let Spark infer the schema, shuffle-join the dimension, hash the
  aggregation by the skewed key, and let the default shuffle partitioning decide
  the task sizes;
* optimised: declare the schema, broadcast the dimension, salt the key so the
  hot event spreads across buckets, two-stage aggregate, and write partitioned
  Parquet so later queries prune.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_tuning import schema
from spark_tuning.measure import Timings, cardinality, partition_stats


@dataclass
class PipelineRun:
    label: str
    frame: DataFrame | None
    timings: Timings
    distributions: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return self.timings.total

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "total_seconds": self.total_seconds,
            "timings": self.timings.values,
            "distributions": self.distributions,
            "notes": self.notes,
        }


def read_events(spark: SparkSession, path: Path) -> DataFrame:
    return spark.read.option("header", True).option("inferSchema", True).csv(str(path))


def naive_run(spark: SparkSession, activities: Path, events: Path, shuffle_partitions: int = 8) -> PipelineRun:
    """What a first draft looks like."""
    timings = Timings()
    notes = [
        "inferSchema reads the JSON once to guess types, then again to parse it",
        f"spark.sql.shuffle.partitions pinned to {shuffle_partitions} so the skew is visible",
    ]
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))

    with timings.stage("read_and_flatten"):
        raw = spark.read.json(str(activities))  # inferSchema is the default
        points = schema.flatten_telemetry(raw)
        points.cache()
        point_count = cardinality(points)

    dimension = read_events(spark, events)

    with timings.stage("aggregate"):
        joined = points.join(dimension, on="event_id", how="left")
        # Hash by the skewed key: one enormous key lands in one task.
        joined = joined.repartition(shuffle_partitions, "event_id")
        result = (
            joined.groupBy("event_id", "name", "terrain")
            .agg(
                F.count("*").alias("points"),
                F.avg("heart_rate").alias("avg_hr"),
                F.max("heart_rate").alias("max_hr"),
            )
        )
        row_count = cardinality(result)

    # Measured separately: this is the distribution the shuffle created.
    with timings.stage("measure_distribution"):
        distributions = {f"hash_by_key_{shuffle_partitions}": partition_stats(joined)}

    notes.append(f"{point_count:,} telemetry rows flattened into {row_count} event rows")
    return PipelineRun("naive", result, timings, distributions, notes)


def optimised_run(
    spark: SparkSession,
    activities: Path,
    events: Path,
    output: Path | None = None,
    shuffle_partitions: int = 8,
    salt_buckets: int = 16,
) -> PipelineRun:
    """The same answers, produced without waiting on one task."""
    timings = Timings()
    notes = [
        "explicit schema avoids the inference pass and the wrong-guess risk",
        f"dimension broadcast, key salted into {salt_buckets} buckets, two-stage aggregation",
        "adaptive query execution enabled",
    ]
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))

    with timings.stage("read_and_flatten"):
        raw = spark.read.schema(schema.ACTIVITY).json(str(activities))
        points = schema.flatten_telemetry(raw)
        points.cache()
        point_count = cardinality(points)

    dimension = read_events(spark, events)

    with timings.stage("aggregate"):
        result = schema.per_event_metrics_salted(points, dimension, salt_buckets=salt_buckets)
        row_count = cardinality(result)

    with timings.stage("measure_distribution"):
        joined = points.join(F.broadcast(dimension), on="event_id", how="left")
        salted = joined.withColumn("salt", (F.rand(seed=7) * salt_buckets).cast("int"))
        distributions = {
            f"hash_by_key_{shuffle_partitions}": partition_stats(
                joined.repartition(shuffle_partitions, "event_id")
            ),
            f"hash_by_key_and_salt_{shuffle_partitions}": partition_stats(
                salted.repartition(shuffle_partitions, "event_id", "salt")
            ),
        }

    if output is not None:
        with timings.stage("write_partitioned"):
            target = Path(output)
            if target.exists():
                shutil.rmtree(target)
            (
                points.write.mode("overwrite")
                .partitionBy("year", "month", "country")
                .parquet(str(target))
            )
            files = len(list(target.rglob("*.parquet")))
            notes.append(f"wrote {files} Parquet files partitioned by year, month, and country")
            notes.append(f"aggregated {point_count:,} telemetry rows into {row_count} event rows")

    return PipelineRun("optimised", result, timings, distributions, notes)
