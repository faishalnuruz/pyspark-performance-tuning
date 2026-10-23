"""Generate nested telemetry JSON with the skew that breaks naive jobs.

Two properties matter for a performance story:

* the records are nested — an array of per-second samples inside each activity —
  so flattening is real work, not a formality;
* one event is an ultra-distance race with hundreds of athletes, so a single
  join key holds most of the data and a hash shuffle concentrates it in one
  task. That is the skew the optimised job has to survive.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

EVENTS = (
    # event_id, name, distance_km, terrain, athletes, points per athlete
    ("ugm-100k", "UGM Trail 100K", 100.0, "mountain", 200, 10_000),
    ("dieng-42k", "Dieng Caldera 42K", 42.0, "volcanic", 60, 3_600),
    ("ctc-76k", "Chapada Trail Challenge 76K", 76.0, "rocky", 45, 6_500),
    ("rinjani-36k", "Rinjani Rim 36K", 36.0, "mountain", 50, 3_000),
    ("semeru-21k", "Semeru Hills 21K", 21.0, "forest", 70, 1_800),
    ("merbabu-15k", "Merbabu Ridge 15K", 15.0, "mountain", 80, 1_300),
    ("bromo-10k", "Bromo Caldera 10K", 10.0, "sandy", 90, 900),
    ("city-5k", "City Night Run 5K", 5.0, "road", 120, 450),
)

COUNTRIES = ("ID", "MY", "SG", "AU", "JP", "GB")
YEAR, MONTH = 2026, 10


@dataclass
class GenerationStats:
    activities: int
    points: int
    points_by_event: dict[str, int]

    @property
    def hottest_event_share(self) -> float:
        return max(self.points_by_event.values()) / self.points


def _telemetry(rng: random.Random, points: int, base_lat: float, base_lon: float) -> list[dict]:
    elevation = rng.uniform(400, 2_600)
    heart_rate = rng.randint(120, 150)
    samples = []
    for index in range(points):
        elevation += rng.uniform(-1.6, 1.8)
        heart_rate = max(95, min(190, heart_rate + rng.randint(-2, 2)))
        samples.append({
            "ts": 1_760_000_000 + index,
            "lat": round(base_lat + index * 0.000_02 + rng.uniform(-1e-5, 1e-5), 7),
            "lon": round(base_lon + index * 0.000_03 + rng.uniform(-1e-5, 1e-5), 7),
            "elevation_m": round(elevation, 1),
            "heart_rate": heart_rate,
            "cadence": rng.randint(62, 88),
        })
    return samples


def _activity(rng: random.Random, event: tuple, athlete_index: int) -> dict:
    event_id, name, distance_km, _, _, points = event
    athlete_id = f"{event_id}-a{athlete_index:04d}"
    duration = int(points * rng.uniform(0.75, 1.4))
    return {
        "activity_id": f"{athlete_id}-act",
        "event": {
            "event_id": event_id,
            "name": name,
            "distance_km": distance_km,
            "year": YEAR,
            "month": MONTH,
        },
        "athlete": {
            "athlete_id": athlete_id,
            "name": f"athlete {athlete_index:04d}",
            "country": rng.choice(COUNTRIES),
            "birth_year": rng.randint(1968, 2004),
        },
        "summary": {
            "duration_s": duration,
            "avg_hr": rng.randint(130, 165),
            "max_hr": rng.randint(165, 191),
        },
        "telemetry": _telemetry(rng, points, -7.6 + rng.uniform(-0.4, 0.4), 110.4 + rng.uniform(-0.4, 0.4)),
    }


def generate(activities_dir: Path, events_path: Path, seed: int = 21, scale: float = 1.0) -> GenerationStats:
    """Write one JSON file per event, plus the small event dimension as CSV."""
    rng = random.Random(seed)
    activities_dir.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    points_by_event: dict[str, int] = {}
    activities = 0
    points = 0

    for event in EVENTS:
        event_id, name, distance_km, terrain, athletes, points_each = event
        athletes = max(2, int(athletes * scale))
        points_each = max(20, int(points_each * scale))

        with (activities_dir / f"{event_id}.json").open("w") as handle:
            for index in range(athletes):
                record = _activity(rng, (event_id, name, distance_km, terrain, athletes, points_each), index)
                handle.write(json.dumps(record) + "\n")
                activities += 1
                points += points_each

        points_by_event[event_id] = athletes * points_each

    with events_path.open("w") as handle:
        handle.write("event_id,name,distance_km,terrain\n")
        for event_id, name, distance_km, terrain, _, _ in EVENTS:
            handle.write(f"{event_id},{name},{distance_km},{terrain}\n")

    return GenerationStats(activities=activities, points=points, points_by_event=points_by_event)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activities-dir", type=Path, default=Path("data/activities"))
    parser.add_argument("--events", type=Path, default=Path("data/events.csv"))
    parser.add_argument("--scale", type=float, default=1.0, help="shrink the dataset for quick runs")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    stats = generate(args.activities_dir, args.events, seed=args.seed, scale=args.scale)
    print(f"Wrote {stats.activities:,} activities and {stats.points:,} telemetry points")
    print(f"Hottest event holds {stats.hottest_event_share:.1%} of all points")
    for event_id, count in sorted(stats.points_by_event.items(), key=lambda item: -item[1]):
        print(f"  {event_id:14} {count:>10,}")


if __name__ == "__main__":
    main()
