"""The point of the whole project: the optimised job must agree with the naive one."""

from spark_tuning import benchmark, jobs


def test_both_pipelines_reach_the_same_answers(spark, dataset):
    naive = jobs.naive_run(spark, dataset["activities"], dataset["events"], shuffle_partitions=4)
    optimised = jobs.optimised_run(
        spark, dataset["activities"], dataset["events"], shuffle_partitions=4, salt_buckets=8
    )

    comparison = benchmark.compare(naive.frame, optimised.frame)

    assert comparison["mismatches"] == []
    assert comparison["identical"] is True
    assert comparison["events_compared"] > 0


def test_a_deliberate_mismatch_is_detected(spark, dataset):
    """A comparison that cannot fail is not a test."""
    optimised = jobs.optimised_run(
        spark, dataset["activities"], dataset["events"], shuffle_partitions=4, salt_buckets=4
    )
    tampered = optimised.frame.withColumn("points", optimised.frame["points"] + 1)
    naive = jobs.naive_run(spark, dataset["activities"], dataset["events"], shuffle_partitions=4)

    comparison = benchmark.compare(naive.frame, tampered)

    assert comparison["identical"] is False
    assert comparison["mismatches"]
    assert comparison["mismatches"][0]["field"] == "points"


def test_the_optimised_run_reports_its_distributions(spark, dataset):
    run = jobs.optimised_run(
        spark, dataset["activities"], dataset["events"], shuffle_partitions=4, salt_buckets=8
    )

    distributions = run.distributions
    hot = distributions["hash_by_key_4"]
    spread = distributions["hash_by_key_and_salt_4"]

    assert hot["rows"] == spread["rows"]
    assert spread["skew_ratio"] < hot["skew_ratio"]
    assert run.total_seconds > 0


def test_partitioned_write_lets_a_filter_skip_files(spark, dataset, tmp_path):
    output = tmp_path / "parquet"
    jobs.optimised_run(
        spark, dataset["activities"], dataset["events"], output=output, shuffle_partitions=4
    )

    from spark_tuning.measure import partition_layout, rows_for_filter

    layout = partition_layout(str(output))
    one_country = partition_layout(str(output), "country=ID")

    assert layout["files"] > 0
    assert layout["partition_directories"] > 1
    assert 0 < one_country["matching_files"] < layout["files"]
    assert rows_for_filter(spark, str(output), "country = 'ID'") < rows_for_filter(spark, str(output))
