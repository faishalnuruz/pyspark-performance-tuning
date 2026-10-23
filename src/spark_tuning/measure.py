"""Measurement helpers.

A performance claim needs a number and a distribution, not a screenshot of a
progress bar. Everything here is computed from the data itself so the report can
be regenerated and checked.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


@dataclass
class Timings:
    """Named wall-clock measurements in seconds."""

    values: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = round(time.perf_counter() - started, 3)

    @property
    def total(self) -> float:
        return round(sum(self.values.values()), 3)


def partition_stats(frame: DataFrame) -> dict:
    """Count rows per partition, and how badly the worst one dwarfs the rest.

    ``skew_ratio`` is the number that matters: a ratio near 1 means the work is
    spread evenly, and a ratio of 50 means one task is doing 50 times the median
    task's share of the data while the rest of the cluster waits.
    """
    counts = [
        row["count"]
        for row in frame.select(F.spark_partition_id().alias("pid")).groupBy("pid").count().collect()
    ]
    counts = sorted(counts)
    if not counts:
        return {"partitions": 0, "rows": 0, "max_rows": 0, "median_rows": 0, "skew_ratio": 0.0}

    median = statistics.median(counts)
    maximum = counts[-1]
    return {
        "partitions": len(counts),
        "rows": sum(counts),
        "min_rows": counts[0],
        "max_rows": maximum,
        "median_rows": int(median),
        "skew_ratio": round(maximum / median, 2) if median else 0.0,
    }


def cardinality(frame: DataFrame) -> int:
    return frame.count()


def partition_layout(path: str, subset_fragment: str | None = None) -> dict:
    """Describe the partitioned layout on disk.

    Pruning is a property of the layout plus the query filter, so it is measured
    where it is unambiguous: how many partition directories and files exist, and
    how many of them a filter on a partition column has to touch. Row counts are
    measured separately, through Spark, to confirm the filter really selects less
    data.
    """
    root = Path(path)
    files = sorted(root.rglob("*.parquet"))
    directories = sorted({file.parent for file in files})

    selected = [file for file in files if subset_fragment is None or subset_fragment in file.parent.as_posix()]
    return {
        "files": len(files),
        "partition_directories": len(directories),
        "matching_files": len(selected),
        "sample_directories": [directory.relative_to(root).as_posix() for directory in directories[:6]],
    }


def rows_for_filter(spark: SparkSession, path: str, where: str | None = None) -> int:
    frame = spark.read.parquet(path)
    if where:
        frame = frame.where(where)
    return frame.count()
