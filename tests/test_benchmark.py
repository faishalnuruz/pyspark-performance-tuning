import json

from spark_tuning import benchmark


def test_benchmark_run_is_complete_and_self_consistent(dataset, tmp_path):
    result = benchmark.run(
        dataset["activities"],
        dataset["events"],
        tmp_path / "parquet",
        shuffle_partitions=4,
        salt_buckets=8,
    )

    assert result["correctness"]["identical"] is True
    assert result["skew"]["naive_hash_by_key"]["skew_ratio"] > 1
    assert (
        result["skew"]["optimised_hash_by_key_and_salt"]["skew_ratio"]
        < result["skew"]["naive_hash_by_key"]["skew_ratio"]
    )
    pruning = result["pruning"]
    assert pruning["one_country"]["files"] < pruning["layout"]["files"]
    assert pruning["one_country"]["rows"] < pruning["all_rows"]
    assert pruning["layout"]["partition_directories"] > 1
    assert set(result["schema_read"]) == {"read_infer_schema", "read_explicit_schema"}
    assert result["naive"]["timings"]["aggregate"] >= 0


def test_summary_mentions_every_headline(dataset, tmp_path):
    result = benchmark.run(
        dataset["activities"], dataset["events"], tmp_path / "parquet", shuffle_partitions=4
    )

    summary = benchmark.summarise(result)

    for expected in ["end to end", "schema inference", "task skew", "results identical", "layout", "filter effect"]:
        assert expected in summary


def test_json_result_round_trips(dataset, tmp_path):
    result = benchmark.run(
        dataset["activities"], dataset["events"], tmp_path / "parquet", shuffle_partitions=4
    )

    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(result, indent=2))

    reloaded = json.loads(path.read_text())

    assert reloaded["correctness"]["mismatches"] == []
    assert reloaded["config"]["shuffle_partitions"] == 4
