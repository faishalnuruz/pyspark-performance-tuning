"""Run both pipelines on the same data and prove they agree.

    python -m spark_tuning.benchmark --activities data/activities --events data/events.csv

Writes a JSON result, a markdown evidence file, and prints the comparison.
The correctness check is not an afterthought: an optimisation that changes the
answer is a bug with a nicer runtime.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_tuning import jobs, schema
from spark_tuning.measure import Timings, partition_layout, rows_for_filter

TOLERANCE = 0.01  # averages can differ in the last digits; counts cannot differ at all


def build_spark(app_name: str = "spark-tuning-benchmark", shuffle_partitions: int = 8) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.autoBroadcastJoinThreshold", "10m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def compare(naive: DataFrame, optimised: DataFrame) -> dict:
    """Join the two outputs and check every measure they have in common."""
    left = naive.select(
        "event_id",
        F.col("name").alias("naive_name"),
        F.col("points").alias("naive_points"),
        F.col("max_hr").alias("naive_max_hr"),
        F.col("avg_hr").alias("naive_avg_hr"),
    )
    right = optimised.select(
        "event_id",
        F.col("event_name").alias("optimised_name"),
        F.col("points").alias("optimised_points"),
        F.col("max_hr").alias("optimised_max_hr"),
        F.col("avg_hr").alias("optimised_avg_hr"),
    )
    compared = left.join(right, on="event_id", how="full_outer").collect()

    mismatches = []
    for row in compared:
        if row["naive_points"] != row["optimised_points"]:
            mismatches.append({"event_id": row["event_id"], "field": "points"})
        if row["naive_max_hr"] != row["optimised_max_hr"]:
            mismatches.append({"event_id": row["event_id"], "field": "max_hr"})
        if abs((row["naive_avg_hr"] or 0) - (row["optimised_avg_hr"] or 0)) > TOLERANCE:
            mismatches.append({
                "event_id": row["event_id"],
                "field": "avg_hr",
                "naive": row["naive_avg_hr"],
                "optimised": row["optimised_avg_hr"],
            })

    return {
        "events_compared": len(compared),
        "mismatches": mismatches,
        "identical": not mismatches,
    }


def run(
    activities: Path,
    events: Path,
    parquet_output: Path,
    shuffle_partitions: int = 8,
    salt_buckets: int = 16,
) -> dict:
    spark = build_spark(shuffle_partitions=shuffle_partitions)
    try:
        naive = jobs.naive_run(spark, activities, events, shuffle_partitions=shuffle_partitions)
        optimised = jobs.optimised_run(
            spark,
            activities,
            events,
            output=parquet_output,
            shuffle_partitions=shuffle_partitions,
            salt_buckets=salt_buckets,
        )
        # Price the schema-inference pass on its own, on the same files.
        schema_read = Timings()
        with schema_read.stage("read_infer_schema"):
            spark.read.json(str(activities)).count()
        with schema_read.stage("read_explicit_schema"):
            spark.read.schema(schema.ACTIVITY).json(str(activities)).count()

        correctness = compare(naive.frame, optimised.frame)

        # Pruning evidence: the layout, plus the row counts the filter actually reads.
        layout = partition_layout(str(parquet_output))
        pruning = {
            "layout": layout,
            "one_country": {
                "files": partition_layout(str(parquet_output), "country=ID")["matching_files"],
                "rows": rows_for_filter(spark, str(parquet_output), "country = 'ID'"),
            },
            "all_rows": rows_for_filter(spark, str(parquet_output)),
        }

        naive_dist = naive.distributions.get(f"hash_by_key_{shuffle_partitions}", {})
        salted_dist = optimised.distributions.get(
            f"hash_by_key_and_salt_{shuffle_partitions}", {}
        )
        plain_dist = optimised.distributions.get(f"hash_by_key_{shuffle_partitions}", {})

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "input": {"activities": str(activities), "events": str(events)},
            "config": {
                "shuffle_partitions": shuffle_partitions,
                "salt_buckets": salt_buckets,
                "tolerance": TOLERANCE,
            },
            "naive": naive.as_dict(),
            "optimised": optimised.as_dict(),
            "schema_read": schema_read.values,
            "correctness": correctness,
            "skew": {
                "naive_hash_by_key": naive_dist,
                "optimised_hash_by_key": plain_dist,
                "optimised_hash_by_key_and_salt": salted_dist,
            },
            "pruning": pruning,
        }
    finally:
        spark.stop()


def summarise(result: dict) -> str:
    naive, optimised = result["naive"], result["optimised"]
    skew = result["skew"]
    read = result["schema_read"]
    pruning = result["pruning"]
    lines = [
        (
            f"end to end        naive {naive['total_seconds']:>7.1f}s   "
            f"optimised {optimised['total_seconds']:>7.1f}s"
        ),
        (
            f"  read+flatten    naive {naive['timings'].get('read_and_flatten', 0):>7.1f}s   "
            f"optimised {optimised['timings'].get('read_and_flatten', 0):>7.1f}s"
        ),
        (
            f"  aggregate       naive {naive['timings'].get('aggregate', 0):>7.1f}s   "
            f"optimised {optimised['timings'].get('aggregate', 0):>7.1f}s"
        ),
        (
            f"schema inference  inferSchema {read.get('read_infer_schema', 0):>7.1f}s   "
            f"explicit {read.get('read_explicit_schema', 0):>7.1f}s"
        ),
        (
            f"task skew         hash by key {skew['naive_hash_by_key'].get('skew_ratio')}   "
            f"hash by key+salt {skew['optimised_hash_by_key_and_salt'].get('skew_ratio')}"
        ),
        (
            f"results identical {result['correctness']['identical']} "
            f"({result['correctness']['events_compared']} events compared)"
        ),
        (
            f"layout            {pruning['layout']['files']} files in "
            f"{pruning['layout']['partition_directories']} partition directories; "
            f"one country needs {pruning['one_country']['files']} of them"
        ),
        (
            f"filter effect     {pruning['one_country']['rows']:,} rows for one country vs "
            f"{pruning['all_rows']:,} unfiltered"
        ),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activities", type=Path, default=Path("data/activities"))
    parser.add_argument("--events", type=Path, default=Path("data/events.csv"))
    parser.add_argument("--parquet", type=Path, default=Path("data/parquet/telemetry"))
    parser.add_argument("--json", type=Path, default=Path("docs/generated/benchmark.json"))
    parser.add_argument("--shuffle-partitions", type=int, default=8)
    parser.add_argument("--salt-buckets", type=int, default=16)
    args = parser.parse_args()

    result = run(
        args.activities,
        args.events,
        args.parquet,
        shuffle_partitions=args.shuffle_partitions,
        salt_buckets=args.salt_buckets,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")

    print(summarise(result))
    print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
