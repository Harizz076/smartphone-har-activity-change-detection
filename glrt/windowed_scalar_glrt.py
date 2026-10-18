#!/usr/bin/env python3
"""Scalar one-class GLRT motion detector for UCI HAR and WISDM."""
from __future__ import annotations
import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from baseline.variance_threshold_baseline import ACTIVITY_NAMES, read_labels, read_variance_statistic
from baseline.wisdm_variance_threshold_baseline import TEST_USERS, read_windows
UCI_STILL = frozenset({4, 5, 6})
WISDM_STILL = frozenset({"Sitting", "Standing"})

def upper_threshold(scores: list[float], alpha: float) -> float:
    ordered = sorted(scores)
    return ordered[min(len(ordered) - 1, math.ceil((1 - alpha) * len(ordered)) - 1)]

def score(log_variance: float, location: float, scale: float) -> float:
    return max(0.0, (log_variance - location) / scale) ** 2

def fit(statistics: list[float], is_still: list[bool], alpha: float) -> tuple[float, float, float]:
    reference = [math.log(value) for value, still in zip(statistics, is_still) if still and value > 0]
    if len(reference) < 2: raise ValueError("Need at least two positive stillness windows")
    location, scale = mean(reference), pstdev(reference)
    if scale == 0: raise ValueError("Stillness log-variance is constant")
    return location, scale, upper_threshold([score(value, location, scale) for value in reference], alpha)

def evaluate(statistics: list[float], is_still: list[bool], location: float, scale: float, threshold: float) -> tuple[dict[str, float | int], list[float], list[bool]]:
    scores = [score(math.log(value), location, scale) if value > 0 else math.inf for value in statistics]
    predicted = [value > threshold for value in scores]
    actual = [not value for value in is_still]
    tp = sum(a and p for a, p in zip(actual, predicted)); fn = sum(a and not p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted)); tn = sum(not a and not p for a, p in zip(actual, predicted))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "accuracy": (tp + tn) / len(statistics),
            "motion_recall": tp / (tp + fn), "still_specificity": tn / (tn + fp),
            "false_alarm_rate": fp / (fp + tn), "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2}, scores, predicted

def rows(labels: list[str], statistics: list[float], is_still: list[bool], scores: list[float], predicted: list[bool]) -> list[dict[str, object]]:
    answer = []
    for label in dict.fromkeys(labels):
        indexes = [index for index, value in enumerate(labels) if value == label]
        correct = sum(predicted[index] == (not is_still[index]) for index in indexes)
        answer.append({"activity": label, "windows": len(indexes), "mean_variance": mean(statistics[index] for index in indexes),
                       "mean_glr_score": mean(scores[index] for index in indexes), "correct": correct, "accuracy": correct / len(indexes)})
    return answer

def load_uci(root: Path):
    train_stats, test_stats = read_variance_statistic(root, "train"), read_variance_statistic(root, "test")
    train_labels, test_labels = read_labels(root / "train/y_train.txt"), read_labels(root / "test/y_test.txt")
    return (train_stats, [label in UCI_STILL for label in train_labels], [ACTIVITY_NAMES[label] for label in train_labels],
            test_stats, [label in UCI_STILL for label in test_labels], [ACTIVITY_NAMES[label] for label in test_labels])

def load_wisdm(path: Path):
    data, _ = read_windows(path); train = [row for row in data if row[0] not in TEST_USERS]; test = [row for row in data if row[0] in TEST_USERS]
    return ([value for _, _, value in train], [label in WISDM_STILL for _, label, _ in train], [label for _, label, _ in train],
            [value for _, _, value in test], [label in WISDM_STILL for _, label, _ in test], [label for _, label, _ in test])

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uci", "wisdm", "all"), default="all")
    parser.add_argument("--false-alarm-target", type=float, default=.01)
    parser.add_argument("--uci-root", type=Path, default=PROJECT_ROOT / "Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset")
    parser.add_argument("--wisdm-file", type=Path, default=PROJECT_ROOT / "Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if not 0 < args.false_alarm_target < 1: parser.error("--false-alarm-target must be strictly between zero and one")
    loaders = []
    if args.dataset in ("uci", "all"): loaders.append(("UCI HAR", load_uci(args.uci_root), "uci_har_windowed_scalar_glrt_activity_breakdown.csv"))
    if args.dataset in ("wisdm", "all"): loaders.append(("WISDM", load_wisdm(args.wisdm_file), "wisdm_windowed_scalar_glrt_activity_breakdown.csv"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, data, filename in loaders:
        train_stats, train_still, train_labels, test_stats, test_still, test_labels = data
        location, scale, threshold = fit(train_stats, train_still, args.false_alarm_target)
        train_result, _, _ = evaluate(train_stats, train_still, location, scale, threshold)
        test_result, test_scores, predictions = evaluate(test_stats, test_still, location, scale, threshold)
        print(f"\n{dataset}: fitted to {sum(train_still)} still training windows only")
        print(f"Log-variance still model: mean={location:.8f}; population std={scale:.8f}; threshold={threshold:.8f}")
        print(f"Training false-alarm={train_result['false_alarm_rate']:.2%}; held-out accuracy={test_result['accuracy']:.2%}; motion recall={test_result['motion_recall']:.2%}; still specificity={test_result['still_specificity']:.2%}")
        output = args.output_dir / filename
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["activity", "windows", "mean_variance", "mean_glr_score", "correct", "accuracy"]); writer.writeheader(); writer.writerows(rows(test_labels, test_stats, test_still, test_scores, predictions))
        print(f"Per-activity results written to {output}")

if __name__ == "__main__": main()
