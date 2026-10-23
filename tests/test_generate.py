import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spark_tuning.generate import EVENTS, generate


def test_generated_records_are_nested(tmp_path):
    activities = tmp_path / "activities"
    generate(activities, tmp_path / "events.csv", seed=1, scale=0.02)

    record = json.loads((activities / "ugm-100k.json").read_text().splitlines()[0])

    assert set(record) == {"activity_id", "event", "athlete", "summary", "telemetry"}
    assert isinstance(record["telemetry"], list)
    assert set(record["telemetry"][0]) == {"ts", "lat", "lon", "elevation_m", "heart_rate", "cadence"}
    assert record["event"]["event_id"] == "ugm-100k"


def test_the_dataset_is_deliberately_skewed(tmp_path):
    """One hot key is the whole point: without it there is nothing to tune."""
    stats = generate(tmp_path / "activities", tmp_path / "events.csv", seed=1)

    assert stats.hottest_event_share > 0.5
    assert max(stats.points_by_event.values()) > min(stats.points_by_event.values()) * 20


def test_one_file_per_event_and_a_dimension_file(tmp_path):
    activities = tmp_path / "activities"
    events = tmp_path / "events.csv"
    generate(activities, events, seed=1, scale=0.02)

    assert len(list(activities.glob("*.json"))) == len(EVENTS)
    header, *rows = events.read_text().strip().splitlines()
    assert header == "event_id,name,distance_km,terrain"
    assert len(rows) == len(EVENTS)


def test_scale_shrinks_the_dataset_without_changing_its_shape(tmp_path):
    small = generate(tmp_path / "small", tmp_path / "e.csv", seed=1, scale=0.01)
    large = generate(tmp_path / "large", tmp_path / "e2.csv", seed=1, scale=0.05)

    assert small.points < large.points
    assert set(small.points_by_event) == set(large.points_by_event)
    # The smallest events clamp to a floor, so the share shifts a little at tiny
    # scales; what must hold is that the ultra event stays dominant either way.
    assert small.hottest_event_share > 0.25
    assert large.hottest_event_share > 0.5
    assert max(small.points_by_event, key=small.points_by_event.get) == "ugm-100k"
    assert max(large.points_by_event, key=large.points_by_event.get) == "ugm-100k"


def test_generation_is_reproducible(tmp_path):
    first = generate(tmp_path / "a", tmp_path / "e1.csv", seed=5, scale=0.01)
    second = generate(tmp_path / "b", tmp_path / "e2.csv", seed=5, scale=0.01)

    assert first.points_by_event == second.points_by_event
    assert (tmp_path / "a" / "ugm-100k.json").read_text() == (tmp_path / "b" / "ugm-100k.json").read_text()
