#!/usr/bin/env python3
"""Continuous scalar one-class GLRT over WISDM raw accelerometer streams."""
from __future__ import annotations
import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from baseline.wisdm_variance_threshold_baseline import TEST_USERS
STILL = frozenset({"Sitting", "Standing"})

@dataclass(frozen=True)
class Frame:
    user: int
    end_timestamp: int
    log_mean_variance: float
    pure_label: str | None

def variance(values: list[float]) -> float:
    average = sum(values) / len(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)

def scalar_feature(samples: list[tuple[float, float, float]]) -> float:
    """Original windowed feature: log mean tri-axial variance."""
    if len(samples) < 3: raise ValueError("Feature window needs at least three samples")
    value = mean(variance([sample[axis] for sample in samples]) for axis in range(3))
    return math.log(max(value, 1e-12))

def read_streams(path: Path, max_gap: int) -> list[list[tuple[int, str, int, tuple[float, float, float]]]]:
    """Labels are kept for fit/evaluation but never create a stream boundary."""
    streams, current = [], []
    last_user = last_time = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            for record in line.strip().split(";"):
                fields = record.split(",")
                if len(fields) != 6: continue
                try:
                    user, timestamp = int(fields[0]), int(fields[2])
                    sample = tuple(float(value) for value in fields[3:6])
                except ValueError: continue
                if not all(math.isfinite(value) for value in sample): continue
                if last_user is not None and (user != last_user or timestamp < last_time or timestamp - last_time > max_gap):
                    if current: streams.append(current)
                    current = []
                current.append((user, fields[1], timestamp, sample))
                last_user, last_time = user, timestamp
    if current: streams.append(current)
    return streams

def rolling_frames(stream: list[tuple[int, str, int, tuple[float, float, float]]], window: int, hop: int) -> list[Frame]:
    frames = []
    for start in range(0, len(stream) - window + 1, hop):
        section = stream[start:start + window]
        labels = {row[1] for row in section}
        frames.append(Frame(section[-1][0], section[-1][2], scalar_feature([row[3] for row in section]), next(iter(labels)) if len(labels) == 1 else None))
    return frames

def fit_stillness(frames: list[Frame]) -> tuple[float, float]:
    reference = [frame.log_mean_variance for frame in frames if frame.pure_label in STILL]
    if len(reference) < 2: raise ValueError("Need at least two pure still training frames")
    location, scale = mean(reference), pstdev(reference)
    if scale == 0: raise ValueError("Stillness feature is constant")
    return location, scale

def sequential_scores(projections: list[float], lookback: int) -> list[float]:
    """max_m [max(0, sum(last m standardized scores)/sqrt(m))]^2."""
    scores = []
    for end in range(len(projections)):
        total = best = 0.0
        for width in range(1, min(lookback, end + 1) + 1):
            total += projections[end - width + 1]
            best = max(best, max(0.0, total / math.sqrt(width)) ** 2)
        scores.append(best)
    return scores

def upper_quantile(values: list[float], alpha: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil((1 - alpha) * len(ordered)) - 1)]

def evaluate(frames: list[Frame], scores: list[float], boundary: float) -> dict[str, float | int]:
    pairs = [(frame.pure_label not in STILL, score > boundary) for frame, score in zip(frames, scores) if frame.pure_label is not None]
    tp = sum(actual and predicted for actual, predicted in pairs); fn = sum(actual and not predicted for actual, predicted in pairs)
    fp = sum(not actual and predicted for actual, predicted in pairs); tn = sum(not actual and not predicted for actual, predicted in pairs)
    return {"evaluated": len(pairs), "mixed": len(frames) - len(pairs), "accuracy": (tp + tn) / len(pairs),
            "recall": tp / (tp + fn), "specificity": tn / (tn + fp), "false_alarm": fp / (fp + tn)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, default=PROJECT_ROOT / "Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
    parser.add_argument("--window-size", type=int, default=100); parser.add_argument("--hop-size", type=int, default=20)
    parser.add_argument("--max-lookback", type=int, default=8); parser.add_argument("--false-alarm-target", type=float, default=.01)
    parser.add_argument("--max-gap", type=int, default=500_000_000)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/wisdm_sequential_scalar_glrt.csv")
    args = parser.parse_args()
    if args.window_size < 3 or args.hop_size < 1 or args.max_lookback < 1 or not 0 < args.false_alarm_target < 1: parser.error("invalid detector parameter")
    streams = read_streams(args.raw_file, args.max_gap)
    train_groups = [rolling_frames(stream, args.window_size, args.hop_size) for stream in streams if stream[0][0] not in TEST_USERS]
    test_groups = [rolling_frames(stream, args.window_size, args.hop_size) for stream in streams if stream[0][0] in TEST_USERS]
    train = [frame for group in train_groups for frame in group]; test = [frame for group in test_groups for frame in group]
    location, scale = fit_stillness(train)
    train_scores = [score for group in train_groups for score in sequential_scores([(frame.log_mean_variance-location)/scale for frame in group], args.max_lookback)]
    boundary = upper_quantile([score for frame, score in zip(train, train_scores) if frame.pure_label in STILL], args.false_alarm_target)
    test_scores = [score for group in test_groups for score in sequential_scores([(frame.log_mean_variance-location)/scale for frame in group], args.max_lookback)]
    result = evaluate(test, test_scores, boundary)
    print("Continuous scalar one-class GLRT (WISDM raw stream)")
    print(f"Frames train={len(train)}, test={len(test)}; rolling window={args.window_size}, hop={args.hop_size}; lookback={args.max_lookback}")
    print(f"Still-only threshold={boundary:.8f}; pure held-out frames={result['evaluated']}; mixed transition frames={result['mixed']}")
    print(f"Accuracy={result['accuracy']:.2%}; motion recall={result['recall']:.2%}; still specificity={result['specificity']:.2%}; false alarm={result['false_alarm']:.2%}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user", "end_timestamp", "pure_label", "score", "alarm"]); writer.writeheader()
        writer.writerows({"user": f.user, "end_timestamp": f.end_timestamp, "pure_label": f.pure_label or "MIXED_TRANSITION", "score": s, "alarm": s > boundary} for f, s in zip(test, test_scores))
    print(f"Held-out decisions written to {args.output}")

if __name__ == "__main__": main()
