"""Explicit schemas and flattening helpers for the telemetry dataset.

``inferSchema`` is the single most expensive mistake in a JSON ingestion job:
it reads the whole file to guess types, and it guesses wrong the first time a
value goes missing. The schema here is written down instead.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

TELEMETRY_POINT = StructType([
    StructField("ts", LongType(), False),
    StructField("lat", DoubleType(), True),
    StructField("lon", DoubleType(), True),
    StructField("elevation_m", DoubleType(), True),
    StructField("heart_rate", IntegerType(), True),
    StructField("cadence", IntegerType(), True),
])

EVENT = StructType([
    StructField("event_id", StringType(), False),
    StructField("name", StringType(), False),
    StructField("distance_km", DoubleType(), True),
    StructField("year", IntegerType(), False),
    StructField("month", IntegerType(), False),
])

ATHLETE = StructType([
    StructField("athlete_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("country", StringType(), True),
    StructField("birth_year", IntegerType(), True),
])

SUMMARY = StructType([
    StructField("duration_s", LongType(), True),
    StructField("avg_hr", IntegerType(), True),
    StructField("max_hr", IntegerType(), True),
])

ACTIVITY = StructType([
    StructField("activity_id", StringType(), False),
    StructField("event", EVENT, False),
    StructField("athlete", ATHLETE, False),
    StructField("summary", SUMMARY, True),
    StructField("telemetry", ArrayType(TELEMETRY_POINT), True),
])

# Dimension table: small enough that broadcasting it is always the right call.
EVENT_DIMENSION = StructType([
    StructField("event_id", StringType(), False),
    StructField("name", StringType(), False),
    StructField("distance_km", DoubleType(), True),
    StructField("terrain", StringType(), True),
])


def flatten_telemetry(activities: DataFrame) -> DataFrame:
    """Explode the per-second telemetry array into one row per sample.

    ``col("event.*")`` and ``col("athlete.*")`` lift the nested structs to
    top-level columns without a second pass over the data.
    """
    return (
        activities.select(
            "activity_id",
            F.col("event.event_id").alias("event_id"),
            F.col("event.year").alias("year"),
            F.col("event.month").alias("month"),
            F.col("athlete.athlete_id").alias("athlete_id"),
            F.col("athlete.country").alias("country"),
            "telemetry",
        )
        .withColumn("point", F.explode("telemetry"))
        .select(
            "activity_id",
            "event_id",
            "year",
            "month",
            "athlete_id",
            "country",
            F.col("point.ts").alias("ts"),
            F.col("point.lat").alias("lat"),
            F.col("point.lon").alias("lon"),
            F.col("point.elevation_m").alias("elevation_m"),
            F.col("point.heart_rate").alias("heart_rate"),
            F.col("point.cadence").alias("cadence"),
        )
    )


def per_event_metrics(points: DataFrame, event_dimension: DataFrame, broadcast: bool) -> DataFrame:
    """Aggregate per-second samples into one row per event.

    This is the join that suffers when one event dwarfs the others, so the
    dimension can be broadcast and the grouping can be salted.
    """
    dimension = F.broadcast(event_dimension) if broadcast else event_dimension
    joined = points.join(dimension, on="event_id", how="left")
    return joined.groupBy(
        F.col("event_id"),
        F.col("name").alias("event_name"),
        F.col("terrain"),
        F.col("year"),
        F.col("month"),
    ).agg(
        F.count("*").alias("points"),
        F.avg("heart_rate").alias("avg_hr"),
        F.max("heart_rate").alias("max_hr"),
        F.avg("elevation_m").alias("avg_elevation_m"),
        F.max("elevation_m").alias("max_elevation_m"),
    )


def per_event_metrics_salted(
    points: DataFrame, event_dimension: DataFrame, salt_buckets: int = 16
) -> DataFrame:
    """Two-stage aggregation that spreads a hot key across partitions.

    Stage one groups by ``(key, salt)``, which turns one enormous partition into
    ``salt_buckets`` smaller ones. Stage two re-aggregates the partial results,
    which are tiny by comparison. Averages are recombined as weighted sums, so
    the output is identical to the unsalted aggregation.
    """
    dimension = F.broadcast(event_dimension)
    joined = points.join(dimension, on="event_id", how="left")

    salted = joined.withColumn("salt", (F.rand(seed=42) * salt_buckets).cast("int"))

    partial = salted.groupBy(
        F.col("event_id").alias("event_id"),
        F.col("name").alias("event_name"),
        F.col("terrain"),
        F.col("year"),
        F.col("month"),
        "salt",
    ).agg(
        F.count("*").alias("points"),
        F.sum("heart_rate").alias("hr_sum"),
        F.max("heart_rate").alias("max_hr"),
        F.sum("elevation_m").alias("elevation_sum"),
        F.max("elevation_m").alias("max_elevation_m"),
    )

    return partial.groupBy("event_id", "event_name", "terrain", "year", "month").agg(
        F.sum("points").alias("points"),
        F.round(F.sum("hr_sum") / F.sum("points"), 6).alias("avg_hr"),
        F.max("max_hr").alias("max_hr"),
        F.round(F.sum("elevation_sum") / F.sum("points"), 6).alias("avg_elevation_m"),
        F.max("max_elevation_m").alias("max_elevation_m"),
    )
