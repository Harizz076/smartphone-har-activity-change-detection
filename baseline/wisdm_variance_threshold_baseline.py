#!/usr/bin/env python3
"""Six-class, variance-only threshold baseline for WISDM v1.1 raw data.

WISDM supplies no prescribed train/test split. This script uses every third
participant (IDs 3, 6, ..., 36) as a subject-disjoint test partition, and all
other participants for fitting. Consecutive same-user/same-activity records
are formed into non-overlapping 200-sample (10-second at 20 Hz) windows, as in
the WISDM transformation description.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev


ACTIVITIES = ("Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing")
TEST_USERS = frozenset(range(3, 37, 3))
WINDOW_SIZE = 200
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def mean_axis_variance(samples: list[tuple[float, float, float]]) -> float:
    """Mean of the three unbiased axis variances in a 10-second window."""
    axis_means = [sum(sample[axis] for sample in samples) / len(samples) for axis in range(3)]
    return mean(
        sum((sample[axis] - axis_means[axis]) ** 2 for sample in samples) / (len(samples) - 1)
        for axis in range(3)
    )


def read_windows(path: Path) -> tuple[list[tuple[int, str, float]], Counter[str]]:
    """Parse raw records, clean malformed lines, and create activity-pure windows."""
    windows: list[tuple[int, str, float]] = []
    discarded: Counter[str] = Counter()
    active_key: tuple[int, str] | None = None
    samples: list[tuple[float, float, float]] = []

    def flush() -> None:
        nonlocal samples
        if active_key is None:
            return
        while len(samples) >= WINDOW_SIZE:
            windows.append((active_key[0], active_key[1], mean_axis_variance(samples[:WINDOW_SIZE])))
            samples = samples[WINDOW_SIZE:]
        if samples:
            discarded["incomplete_activity_run_samples"] += len(samples)
        samples = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            # A few source lines contain two semicolon-delimited records; splitting
            # them recovers both instead of treating the full line as corrupt.
            for record in line.strip().split(";"):
                if not record:
                    continue
                fields = record.split(",")
                if len(fields) != 6:
                    discarded["malformed_records"] += 1
                    flush()
                    active_key = None
                    continue
                try:
                    user = int(fields[0])
                    activity = fields[1]
                    acceleration = tuple(float(value) for value in fields[3:6])
                except ValueError:
                    discarded["non_numeric_records"] += 1
                    flush()
                    active_key = None
                    continue
                if activity not in ACTIVITIES or not all(math.isfinite(value) for value in acceleration):
                    discarded["invalid_records"] += 1
                    flush()
                    active_key = None
                    continue
                key = (user, activity)
                if key != active_key:
                    flush()
                    active_key = key
                samples.append(acceleration)
        flush()
    return windows, discarded


def fit_thresholds(train: list[tuple[int, str, float]]) -> tuple[float, float, list[str], list[float]]:
    values = [value for _, _, value in train]
    location, scale = mean(values), pstdev(values)
    class_means = {
        activity: mean((value - location) / scale for _, label, value in train if label == activity)
        for activity in ACTIVITIES
    }
    order = sorted(ACTIVITIES, key=class_means.get)
    thresholds = [(class_means[left] + class_means[right]) / 2 for left, right in zip(order, order[1:])]
    return location, scale, order, thresholds


def predict(value: float, location: float, scale: float, order: list[str], thresholds: list[float]) -> str:
    normalized = (value - location) / scale
    return order[sum(normalized >= boundary for boundary in thresholds)]


def evaluate(rows: list[tuple[int, str, float]], location: float, scale: float,
             order: list[str], thresholds: list[float]) -> tuple[float, list[dict[str, object]], list[list[int]]]:
    predictions = [predict(value, location, scale, order, thresholds) for _, _, value in rows]
    labels = [label for _, label, _ in rows]
    matrix = [[sum(actual == row and guess == column for actual, guess in zip(labels, predictions))
               for column in ACTIVITIES] for row in ACTIVITIES]
    breakdown = []
    for index, activity in enumerate(ACTIVITIES):
        count = sum(label == activity for label in labels)
        breakdown.append({"activity": activity, "windows": count, "correct": matrix[index][index],
                          "accuracy": matrix[index][index] / count})
    return sum(actual == guess for actual, guess in zip(labels, predictions)) / len(rows), breakdown, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path,
                        default=PROJECT_ROOT / "Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    rows, discarded = read_windows(args.raw_file)
    train = [row for row in rows if row[0] not in TEST_USERS]
    test = [row for row in rows if row[0] in TEST_USERS]
    location, scale, order, thresholds = fit_thresholds(train)
    accuracy, breakdown, matrix = evaluate(test, location, scale, order, thresholds)

    print("WISDM six-class variance-only threshold baseline")
    print(f"Windows: {len(rows)} total; {len(train)} train; {len(test)} subject-disjoint test")
    print("Test users: " + ", ".join(map(str, sorted(TEST_USERS))))
    print("Discarded during cleaning: " + ", ".join(f"{key}={value}" for key, value in sorted(discarded.items())))
    print(f"Training normalization: z = (variance - {location:.8f}) / {scale:.8f}")
    print("Variance-region order: " + " -> ".join(order))
    print("Normalized thresholds: " + ", ".join(f"{value:.8f}" for value in thresholds))
    print("Raw variance thresholds (m/s^2)^2: " +
          ", ".join(f"{location + scale * value:.8f}" for value in thresholds))
    print(f"Held-out six-class accuracy={accuracy:.2%}")
    print("Confusion matrix (rows/columns: " + ", ".join(ACTIVITIES) + ")")
    for row in matrix:
        print("  " + " ".join(f"{value:4d}" for value in row))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "wisdm_six_class_variance_thresholds.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["activity", "windows", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(breakdown)
    print(f"Per-activity held-out results written to {output}")


if __name__ == "__main__":
    main()
