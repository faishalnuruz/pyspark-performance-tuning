from pyspark.sql import functions as F

from spark_tuning.measure import Timings, cardinality, partition_stats


def skewed_frame(spark):
    """One key with 900 rows and nine keys with 10 rows each."""
    rows = [("hot", index) for index in range(900)]
    rows += [(f"cold-{bucket}", index) for bucket in range(9) for index in range(10)]
    return spark.createDataFrame(rows, ["key", "value"])


def test_partition_stats_measure_the_spread(spark):
    frame = skewed_frame(spark).repartition(4, "key")

    stats = partition_stats(frame)

    assert stats["partitions"] == 4
    assert stats["rows"] == 990
    assert stats["max_rows"] > stats["median_rows"]
    assert stats["skew_ratio"] > 2


def test_salting_flattens_the_distribution(spark):
    frame = skewed_frame(spark)
    salted = frame.withColumn("salt", (F.rand(seed=1) * 16).cast("int"))

    hot = partition_stats(frame.repartition(4, "key"))
    spread = partition_stats(salted.repartition(4, "key", "salt"))

    assert spread["skew_ratio"] < hot["skew_ratio"]
    assert spread["rows"] == 990


def test_evenly_distributed_data_reports_a_low_ratio(spark):
    frame = spark.range(0, 1000).withColumn("key", (F.col("id") % 20).cast("string"))

    stats = partition_stats(frame.repartition(5, "key"))

    assert stats["skew_ratio"] <= 2


def test_timings_record_named_stages():
    timings = Timings()
    with timings.stage("first"):
        pass
    with timings.stage("second"):
        pass

    assert set(timings.values) == {"first", "second"}
    assert timings.total == round(sum(timings.values.values()), 3)


def test_cardinality_matches_counts(spark):
    assert cardinality(spark.range(0, 7).toDF("id")) == 7
