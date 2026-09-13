#!/usr/bin/env python3
"""Variance-threshold accelerometer baseline for UCI HAR.

The detector produces one decision per supplied 2.56-second sensor window:

    moving  if mean(var(acc_x), var(acc_y), var(acc_z)) >= threshold
    still   otherwise

Labels 1--3 (walking, upstairs, downstairs) are treated as moving and labels
4--6 (sitting, standing, laying) as still.  The threshold is selected using
only the UCI HAR training partition by maximizing balanced accuracy, then it
is evaluated once on the official subject-disjoint test partition.

No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev


ACTIVITY_NAMES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}
MOVING_LABELS = {1, 2, 3}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_labels(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as handle:
        return [int(line) for line in handle if line.strip()]


def sample_variance(values: list[float]) -> float:
    """Unbiased variance; each UCI window has 128 samples."""
    avg = sum(values) / len(values)
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


def read_variance_statistic(dataset_root: Path, partition: str) -> list[float]:
    """Return mean per-axis total-accelerometer variance for each window."""
    signal_dir = dataset_root / partition / "Inertial Signals"
    axis_paths = [signal_dir / f"total_acc_{axis}_{partition}.txt"
                  for axis in ("x", "y", "z")]
    axis_rows: list[list[list[float]]] = []
    for path in axis_paths:
        with path.open(encoding="utf-8") as handle:
            axis_rows.append([[float(value) for value in line.split()]
                              for line in handle if line.strip()])

    if len({len(rows) for rows in axis_rows}) != 1:
        raise ValueError("Accelerometer-axis files do not contain equal numbers of windows")
    return [mean([sample_variance(axis_rows[axis][row]) for axis in range(3)])
            for row in range(len(axis_rows[0]))]


def clean_pairs(statistics: list[float], labels: list[int]) -> tuple[list[float], list[int], int]:
    """Remove invalid feature values while retaining the aligned activity labels."""
    clean = [(stat, label) for stat, label in zip(statistics, labels) if math.isfinite(stat)]
    return [stat for stat, _ in clean], [label for _, label in clean], len(statistics) - len(clean)


def fit_six_class_thresholds(train_statistics: list[float], train_labels: list[int]) -> tuple[float, float, list[int], list[float]]:
    """Fit a z-score scaler and five thresholds between ordered class means.

    With one scalar feature, the six classes must occupy six contiguous regions.
    The class order and thresholds are therefore learned from the training data:
    classes are ordered by their normalized mean variance and a boundary is put
    halfway between adjacent means. This is equivalent to a nearest-class-mean
    classifier on the normalized one-dimensional feature.
    """
    location, scale = mean(train_statistics), pstdev(train_statistics)
    if scale == 0:
        raise ValueError("Training variance statistic is constant; thresholds cannot be fitted")
    class_means = {
        label: mean((stat - location) / scale
                    for stat, actual in zip(train_statistics, train_labels) if actual == label)
        for label in ACTIVITY_NAMES
    }
    activity_order = sorted(ACTIVITY_NAMES, key=class_means.get)
    thresholds = [(class_means[left] + class_means[right]) / 2
                  for left, right in zip(activity_order, activity_order[1:])]
    return location, scale, activity_order, thresholds


def predict_six_class(statistic: float, location: float, scale: float,
                      activity_order: list[int], thresholds: list[float]) -> int:
    normalized = (statistic - location) / scale
    return activity_order[sum(normalized >= threshold for threshold in thresholds)]


def six_class_summary(statistics: list[float], labels: list[int], location: float, scale: float,
                      activity_order: list[int], thresholds: list[float]) -> tuple[float, list[list[int]], list[dict[str, object]]]:
    predictions = [predict_six_class(stat, location, scale, activity_order, thresholds) for stat in statistics]
    matrix = [[sum(actual == row and predicted == column for actual, predicted in zip(labels, predictions))
               for column in range(1, 7)] for row in range(1, 7)]
    breakdown = []
    for label in range(1, 7):
        total = sum(actual == label for actual in labels)
        correct = matrix[label - 1][label - 1]
        breakdown.append({"activity": ACTIVITY_NAMES[label], "windows": total,
                          "correct": correct, "accuracy": correct / total})
    return sum(actual == predicted for actual, predicted in zip(labels, predictions)) / len(labels), matrix, breakdown


def metrics(statistics: list[float], labels: list[int], threshold: float) -> dict[str, float | int]:
    if len(statistics) != len(labels):
        raise ValueError("Number of sensor windows and labels differs")
    truth = [label in MOVING_LABELS for label in labels]
    predicted = [stat >= threshold for stat in statistics]
    tp = sum(actual and guess for actual, guess in zip(truth, predicted))
    tn = sum(not actual and not guess for actual, guess in zip(truth, predicted))
    fp = sum(not actual and guess for actual, guess in zip(truth, predicted))
    fn = sum(actual and not guess for actual, guess in zip(truth, predicted))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        # A false alarm means declaring movement during a labelled still window.
        "false_alarm_rate": fp / (fp + tn),
    }


def choose_threshold(statistics: list[float], labels: list[int]) -> tuple[float, dict[str, float | int]]:
    """Find threshold at a training-observation boundary, maximizing balanced accuracy.

    A cumulative scan avoids rescoring all windows at every candidate threshold.
    """
    observations = sorted(zip(statistics, (label in MOVING_LABELS for label in labels)))
    total_moving = sum(is_moving for _, is_moving in observations)
    total_still = len(observations) - total_moving
    # At a threshold below the smallest statistic, every window is moving.
    tp, fp = total_moving, total_still
    best_threshold = observations[0][0] - 1e-12
    best_result = metrics(statistics, labels, best_threshold)

    index = 0
    while index < len(observations):
        value = observations[index][0]
        # Raising the threshold just above this tied group turns it to still.
        while index < len(observations) and observations[index][0] == value:
            _, is_moving = observations[index]
            if is_moving:
                tp -= 1
            else:
                fp -= 1
            index += 1
        fn = total_moving - tp
        tn = total_still - fp
        result = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(observations),
            "sensitivity": tp / total_moving,
            "specificity": tn / total_still,
            "balanced_accuracy": (tp / total_moving + tn / total_still) / 2,
            "false_alarm_rate": fp / total_still,
        }
        # A midpoint represents this decision unless this is the final group.
        threshold = (value + observations[index][0]) / 2 if index < len(observations) else value + 1e-12
        if (result["balanced_accuracy"], result["accuracy"], -threshold) > (
            best_result["balanced_accuracy"], best_result["accuracy"], -best_threshold
        ):
            best_threshold, best_result = threshold, result
    return best_threshold, best_result


def activity_breakdown(statistics: list[float], labels: list[int], threshold: float) -> list[dict[str, object]]:
    rows = []
    for label in sorted(ACTIVITY_NAMES):
        values = [value for value, actual_label in zip(statistics, labels) if actual_label == label]
        expected_moving = label in MOVING_LABELS
        correct = sum((value >= threshold) == expected_moving for value in values)
        rows.append({
            "activity": ACTIVITY_NAMES[label], "windows": len(values),
            "mean_variance_g2": mean(values), "correct": correct,
            "accuracy": correct / len(values),
        })
    return rows


def print_metrics(title: str, result: dict[str, float | int]) -> None:
    print(f"{title}: accuracy={result['accuracy']:.2%}; balanced accuracy={result['balanced_accuracy']:.2%}")
    print(f"  confusion matrix [actual moving/still, predicted moving/still]: "
          f"TP={result['tp']}, FN={result['fn']}, FP={result['fp']}, TN={result['tn']}")
    print(f"  moving recall={result['sensitivity']:.2%}; still specificity={result['specificity']:.2%}; "
          f"false-alarm rate={result['false_alarm_rate']:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=PROJECT_ROOT / "Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    train_stats = read_variance_statistic(args.dataset_root, "train")
    test_stats = read_variance_statistic(args.dataset_root, "test")
    train_labels = read_labels(args.dataset_root / "train" / "y_train.txt")
    test_labels = read_labels(args.dataset_root / "test" / "y_test.txt")
    train_stats, train_labels, removed_train = clean_pairs(train_stats, train_labels)
    test_stats, test_labels, removed_test = clean_pairs(test_stats, test_labels)
    threshold, train_result = choose_threshold(train_stats, train_labels)
    test_result = metrics(test_stats, test_labels, threshold)

    print("Variance-threshold motion/stillness baseline (UCI HAR)")
    print("Statistic: mean of the sample variances of total_acc_x/y/z in each 128-sample window.")
    print(f"Threshold selected on training data only: {threshold:.8f} g^2")
    print_metrics("Training", train_result)
    print_metrics("Held-out test", test_result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "uci_har_variance_threshold_activity_breakdown.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["activity", "windows", "mean_variance_g2", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(activity_breakdown(test_stats, test_labels, threshold))
    print(f"Per-activity held-out results written to {output_path}")

    location, scale, activity_order, thresholds = fit_six_class_thresholds(train_stats, train_labels)
    six_accuracy, six_matrix, six_breakdown = six_class_summary(
        test_stats, test_labels, location, scale, activity_order, thresholds
    )
    print("\nSix-class variance-only threshold baseline")
    print(f"Cleaning removed {removed_train} training and {removed_test} test windows with non-finite variance.")
    print(f"Training normalization: z = (variance - {location:.8f}) / {scale:.8f}")
    print("Variance-region order: " + " -> ".join(ACTIVITY_NAMES[label] for label in activity_order))
    print("Normalized thresholds: " + ", ".join(f"{value:.8f}" for value in thresholds))
    print(f"Held-out six-class accuracy={six_accuracy:.2%}")
    print("Confusion matrix (rows=actual 1..6; columns=predicted 1..6):")
    for row in six_matrix:
        print("  " + " ".join(f"{value:4d}" for value in row))
    six_output = args.output_dir / "uci_har_six_class_variance_thresholds.csv"
    with six_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["activity", "windows", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(six_breakdown)
    print(f"Six-class held-out results written to {six_output}")


if __name__ == "__main__":
    main()
