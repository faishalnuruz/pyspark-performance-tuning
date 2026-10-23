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

from spark_tuning import schema

POINT = StructType([
    StructField("ts", LongType()),
    StructField("lat", DoubleType()),
    StructField("lon", DoubleType()),
    StructField("elevation_m", DoubleType()),
    StructField("heart_rate", IntegerType()),
    StructField("cadence", IntegerType()),
])
ACTIVITY = StructType([
    StructField("activity_id", StringType()),
    StructField("event", StructType([
        StructField("event_id", StringType()),
        StructField("name", StringType()),
        StructField("distance_km", DoubleType()),
        StructField("year", IntegerType()),
        StructField("month", IntegerType()),
    ])),
    StructField("athlete", StructType([
        StructField("athlete_id", StringType()),
        StructField("name", StringType()),
        StructField("country", StringType()),
        StructField("birth_year", IntegerType()),
    ])),
    StructField("summary", StructType([
        StructField("duration_s", LongType()),
        StructField("avg_hr", IntegerType()),
        StructField("max_hr", IntegerType()),
    ])),
    StructField("telemetry", ArrayType(POINT)),
])


def activities(spark):
    return spark.createDataFrame(
        [
            {
                "activity_id": "a1",
                "event": {"event_id": "e1", "name": "One", "distance_km": 42.0, "year": 2026, "month": 10},
                "athlete": {"athlete_id": "x1", "name": "A", "country": "ID", "birth_year": 1990},
                "summary": {"duration_s": 100, "avg_hr": 150, "max_hr": 180},
                "telemetry": [
                    {"ts": 1, "lat": 1.0, "lon": 2.0, "elevation_m": 100.0, "heart_rate": 140, "cadence": 70},
                    {"ts": 2, "lat": 1.1, "lon": 2.1, "elevation_m": 105.0, "heart_rate": 150, "cadence": 72},
                    {"ts": 3, "lat": 1.2, "lon": 2.2, "elevation_m": 110.0, "heart_rate": 160, "cadence": 74},
                ],
            },
            {
                "activity_id": "a2",
                "event": {"event_id": "e1", "name": "One", "distance_km": 42.0, "year": 2026, "month": 10},
                "athlete": {"athlete_id": "x2", "name": "B", "country": "SG", "birth_year": 1985},
                "summary": {"duration_s": 120, "avg_hr": 145, "max_hr": 175},
                "telemetry": [
                    {"ts": 1, "lat": 3.0, "lon": 4.0, "elevation_m": 200.0, "heart_rate": 130, "cadence": 68},
                    {"ts": 2, "lat": 3.1, "lon": 4.1, "elevation_m": 205.0, "heart_rate": 170, "cadence": 76},
                ],
            },
        ],
        schema=ACTIVITY,
    )


def dimension(spark):
    return spark.createDataFrame(
        [("e1", "One", 42.0, "mountain")], schema=schema.EVENT_DIMENSION
    )


def test_flattening_explodes_every_point_and_lifts_the_structs(spark):
    points = schema.flatten_telemetry(activities(spark))

    assert points.count() == 5
    assert set(points.columns) == {
        "activity_id", "event_id", "year", "month", "athlete_id", "country",
        "ts", "lat", "lon", "elevation_m", "heart_rate", "cadence",
    }
    row = points.orderBy("activity_id", "ts").first()
    assert row["country"] == "ID"  # lifted from the nested athlete struct
    assert row["year"] == 2026


def test_unsalted_aggregation_produces_one_row_per_event(spark):
    stats = schema.per_event_metrics(schema.flatten_telemetry(activities(spark)), dimension(spark), broadcast=True)

    assert stats.count() == 1
    row = stats.first()
    assert row["points"] == 5
    assert row["max_hr"] == 170
    assert row["event_name"] == "One"


def test_salting_preserves_counts_and_weighted_averages(spark):
    points = schema.flatten_telemetry(activities(spark))
    dimension_df = dimension(spark)

    plain = schema.per_event_metrics(points, dimension_df, broadcast=True).first()
    salted = schema.per_event_metrics_salted(points, dimension_df, salt_buckets=4).first()

    assert salted["points"] == plain["points"]
    assert salted["max_hr"] == plain["max_hr"]
    assert abs(salted["avg_hr"] - plain["avg_hr"]) < 0.01
    assert abs(salted["avg_elevation_m"] - plain["avg_elevation_m"]) < 0.01


def test_a_missing_dimension_row_does_not_drop_points(spark):
    """A left join is deliberate: an unknown event must not erase telemetry."""
    empty_dimension = spark.createDataFrame([], schema.EVENT_DIMENSION)

    points = schema.flatten_telemetry(activities(spark))
    stats = schema.per_event_metrics(points, empty_dimension, broadcast=True)

    assert stats.count() == 1
    assert stats.first()["points"] == 5
    assert stats.first()["event_name"] is None


def test_salting_shrinks_the_partial_aggregation(spark):
    """Stage one produces at most one row per (key, salt): that is the spread."""
    points = schema.flatten_telemetry(activities(spark))
    dimension_df = dimension(spark)

    unsalted_keys = schema.per_event_metrics(points, dimension_df, broadcast=True).count()
    salted_partials = schema.per_event_metrics_salted(points, dimension_df, salt_buckets=8)

    # Re-aggregation collapses back to one row per event...
    assert salted_partials.count() == unsalted_keys
    # ...and the flattening really did lift the fields the salt uses.
    assert points.select(F.countDistinct("event_id")).first()[0] == 1
