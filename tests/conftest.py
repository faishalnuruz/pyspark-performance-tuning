import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spark_tuning.generate import generate


@pytest.fixture(scope="session")
def spark():
    """One JVM for the whole suite: startup dominates otherwise."""
    session = (
        SparkSession.builder.appName("spark-tuning-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    """A small dataset with the same shape and skew as the full one."""
    root = tmp_path_factory.mktemp("telemetry")
    activities = root / "activities"
    events = root / "events.csv"
    stats = generate(activities, events, seed=3, scale=0.02)
    return {"activities": activities, "events": events, "stats": stats, "root": root}
