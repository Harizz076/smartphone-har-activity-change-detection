#!/usr/bin/env python3
"""Six-class, window-level Gaussian GLRT for UCI HAR and WISDM.

The implementation is dependency-free and fits only on training subjects.
For each activity c it estimates a regularized Gaussian feature model, then
classifies a held-out window by maximum class-conditional likelihood. The
reported GLR margin is twice the winner-versus-runner-up log-likelihood ratio.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Hashable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UCI_ACTIVITIES = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
)
WISDM_ACTIVITIES = ("Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing")
WISDM_TEST_USERS = frozenset(range(3, 37, 3))
WISDM_WINDOW_SIZE = 200
FEATURE_NAMES = (
    "mean_x", "mean_y", "mean_z",
    "log_variance_x", "log_variance_y", "log_variance_z",
    "correlation_xy", "correlation_xz", "correlation_yz",
    "log_jerk_x", "log_jerk_y", "log_jerk_z",
    "dominant_frequency_hz", "spectral_entropy", "log_band_power",
)
FEATURE_SETS = {
    "variance": (3, 4, 5),
    "posture_variance": (0, 1, 2, 3, 4, 5),
    "compact": (0, 1, 2, 3, 4, 5, 12, 13, 14),
    "time_domain": tuple(range(12)),
    "dynamics": tuple(range(3, 15)),
    "full": tuple(range(15)),
}
COVARIANCE_MODES = ("class_full", "class_diagonal", "shared_full", "shared_diagonal")


@dataclass(frozen=True)
class Standardizer:
    location: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, vector: Sequence[float]) -> tuple[float, ...]:
        if len(vector) != len(self.location):
            raise ValueError("Feature vector dimension differs from fitted standardizer")
        return tuple((value - center) / spread for value, center, spread in zip(vector, self.location, self.scale))


@dataclass(frozen=True)
class GaussianClassModel:
    label: Hashable
    count: int
    mean: tuple[float, ...]
    cholesky: tuple[tuple[float, ...], ...]
    log_determinant: float

    def log_likelihood(self, vector: Sequence[float]) -> float:
        if len(vector) != len(self.mean):
            raise ValueError("Feature vector dimension differs from fitted class model")
        residual = tuple(value - center for value, center in zip(vector, self.mean))
        whitened = forward_solve(self.cholesky, residual)
        mahalanobis = sum(value * value for value in whitened)
        dimensions = len(vector)
        return -0.5 * (dimensions * math.log(2 * math.pi) + self.log_determinant + mahalanobis)


@dataclass(frozen=True)
class Prediction:
    label: Hashable
    glr_margin: float
    log_likelihoods: tuple[tuple[Hashable, float], ...]


def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("Variance requires at least two samples")
    average = fmean(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)


def safe_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_energy = sum((value - left_mean) ** 2 for value in left)
    right_energy = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator > 0 else 0.0


def spectral_features(samples: Sequence[tuple[float, float, float]], sample_rate: float,
                      minimum_hz: float = 0.3, maximum_hz: float = 5.0) -> tuple[float, float, float]:
    """Return dominant frequency, normalized entropy, and log band power.

    Power is summed across the three mean-centered axes, so the result is less
    sensitive to phone orientation than selecting a single axis.
    """
    count = len(samples)
    centered = []
    for axis in range(3):
        values = [sample[axis] for sample in samples]
        average = fmean(values)
        centered.append([value - average for value in values])
    first_bin = max(1, math.ceil(minimum_hz * count / sample_rate))
    last_bin = min(count // 2, math.floor(maximum_hz * count / sample_rate))
    if first_bin > last_bin:
        raise ValueError("Window is too short for the requested spectral band")
    powers: list[tuple[int, float]] = []
    for frequency_bin in range(first_bin, last_bin + 1):
        angle_step = 2 * math.pi * frequency_bin / count
        power = 0.0
        for axis_values in centered:
            real = sum(value * math.cos(angle_step * index) for index, value in enumerate(axis_values))
            imaginary = -sum(value * math.sin(angle_step * index) for index, value in enumerate(axis_values))
            power += real * real + imaginary * imaginary
        powers.append((frequency_bin, power))
    total_power = sum(power for _, power in powers)
    if total_power <= 0:
        return 0.0, 0.0, math.log(1e-12)
    dominant_bin = max(powers, key=lambda item: item[1])[0]
    probabilities = [power / total_power for _, power in powers if power > 0]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized_entropy = entropy / math.log(len(powers)) if len(powers) > 1 else 0.0
    normalized_power = total_power / (count * count)
    return dominant_bin * sample_rate / count, normalized_entropy, math.log(max(normalized_power, 1e-12))


def extract_features(samples: Sequence[tuple[float, float, float]], sample_rate: float) -> tuple[float, ...]:
    if len(samples) < 3:
        raise ValueError("Feature extraction requires at least three samples")
    axes = [[sample[axis] for sample in samples] for axis in range(3)]
    means = [fmean(values) for values in axes]
    variances = [sample_variance(values) for values in axes]
    correlations = [
        safe_correlation(axes[0], axes[1]),
        safe_correlation(axes[0], axes[2]),
        safe_correlation(axes[1], axes[2]),
    ]
    jerk_energies = [
        fmean((current - previous) ** 2 for previous, current in zip(values, values[1:]))
        for values in axes
    ]
    dominant_frequency, spectral_entropy, band_power = spectral_features(samples, sample_rate)
    vector = (
        *means,
        *(math.log(max(value, 1e-12)) for value in variances),
        *correlations,
        *(math.log(max(value, 1e-12)) for value in jerk_energies),
        dominant_frequency, spectral_entropy, band_power,
    )
    if len(vector) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in vector):
        raise ValueError("Feature extraction produced an invalid vector")
    return tuple(vector)


def select_features(vector: Sequence[float], feature_set: str) -> tuple[float, ...]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}")
    if len(vector) != len(FEATURE_NAMES):
        raise ValueError("Feature selection expects the complete feature vector")
    return tuple(vector[index] for index in FEATURE_SETS[feature_set])


def fit_standardizer(vectors: Sequence[Sequence[float]]) -> Standardizer:
    if not vectors:
        raise ValueError("Cannot fit a standardizer without training vectors")
    dimensions = len(vectors[0])
    if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Training feature vectors have inconsistent dimensions")
    location = tuple(fmean(vector[index] for vector in vectors) for index in range(dimensions))
    raw_scale = tuple(math.sqrt(fmean((vector[index] - location[index]) ** 2 for vector in vectors))
                      for index in range(dimensions))
    scale = tuple(value if value > 1e-12 else 1.0 for value in raw_scale)
    return Standardizer(location, scale)


def cholesky(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    dimensions = len(matrix)
    if dimensions == 0 or any(len(row) != dimensions for row in matrix):
        raise ValueError("Cholesky input must be a non-empty square matrix")
    result = [[0.0] * dimensions for _ in range(dimensions)]
    for row in range(dimensions):
        for column in range(row + 1):
            remainder = matrix[row][column] - sum(result[row][index] * result[column][index]
                                                   for index in range(column))
            if row == column:
                if remainder <= 0 or not math.isfinite(remainder):
                    raise ValueError("Covariance is not positive definite")
                result[row][column] = math.sqrt(remainder)
            else:
                result[row][column] = remainder / result[column][column]
    return tuple(tuple(row) for row in result)


def forward_solve(lower: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, ...]:
    if len(lower) != len(vector):
        raise ValueError("Triangular system dimensions differ")
    solution: list[float] = []
    for row, value in enumerate(vector):
        solution.append((value - sum(lower[row][column] * solution[column] for column in range(row)))
                        / lower[row][row])
    return tuple(solution)


def fit_class_model(label: Hashable, vectors: Sequence[Sequence[float]], shrinkage: float,
                    ridge: float, diagonal: bool = False) -> GaussianClassModel:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1]")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    count = len(vectors)
    if count < 2:
        raise ValueError(f"Class {label!r} needs at least two training windows")
    dimensions = len(vectors[0])
    mean = tuple(fmean(vector[index] for vector in vectors) for index in range(dimensions))
    covariance = [[fmean((vector[row] - mean[row]) * (vector[column] - mean[column])
                         for vector in vectors)
                   for column in range(dimensions)] for row in range(dimensions)]
    if diagonal:
        covariance = [[covariance[row][column] if row == column else 0.0
                       for column in range(dimensions)] for row in range(dimensions)]
    return model_from_covariance(label, count, mean, covariance, shrinkage, ridge)


def model_from_covariance(label: Hashable, count: int, mean: Sequence[float],
                          covariance: Sequence[Sequence[float]], shrinkage: float,
                          ridge: float) -> GaussianClassModel:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1]")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    dimensions = len(mean)
    target_variance = fmean(covariance[index][index] for index in range(dimensions))
    if target_variance <= 0 or not math.isfinite(target_variance):
        target_variance = 1.0
    regularized = [[
        (1 - shrinkage) * covariance[row][column]
        + (shrinkage * target_variance if row == column else 0.0)
        + (ridge if row == column else 0.0)
        for column in range(dimensions)] for row in range(dimensions)]
    lower = cholesky(regularized)
    log_determinant = 2 * sum(math.log(lower[index][index]) for index in range(dimensions))
    return GaussianClassModel(label, count, tuple(mean), lower, log_determinant)


def fit_models(vectors: Sequence[Sequence[float]], labels: Sequence[Hashable], class_order: Sequence[Hashable],
               shrinkage: float, ridge: float,
               covariance_mode: str = "class_full") -> tuple[Standardizer, tuple[GaussianClassModel, ...]]:
    if len(vectors) != len(labels) or not vectors:
        raise ValueError("Training features and labels must be non-empty and aligned")
    if len(set(class_order)) != len(class_order):
        raise ValueError("class_order contains duplicates")
    if covariance_mode not in COVARIANCE_MODES:
        raise ValueError(f"Unknown covariance mode {covariance_mode!r}")
    unknown = set(labels) - set(class_order)
    if unknown:
        raise ValueError(f"Training labels are absent from class_order: {unknown}")
    standardizer = fit_standardizer(vectors)
    standardized = [standardizer.transform(vector) for vector in vectors]
    groups = {label: [vector for vector, actual in zip(standardized, labels) if actual == label]
              for label in class_order}
    diagonal = covariance_mode.endswith("diagonal")
    if covariance_mode.startswith("class"):
        models = tuple(fit_class_model(label, groups[label], shrinkage, ridge, diagonal)
                       for label in class_order)
    else:
        class_means = {label: tuple(fmean(vector[index] for vector in groups[label])
                                    for index in range(len(standardized[0])))
                       for label in class_order}
        dimensions = len(standardized[0])
        pooled = [[fmean((vector[row] - class_means[label][row])
                         * (vector[column] - class_means[label][column])
                         for vector, label in zip(standardized, labels))
                   for column in range(dimensions)] for row in range(dimensions)]
        if diagonal:
            pooled = [[pooled[row][column] if row == column else 0.0
                       for column in range(dimensions)] for row in range(dimensions)]
        models = tuple(model_from_covariance(label, len(groups[label]), class_means[label],
                                             pooled, shrinkage, ridge)
                       for label in class_order)
    return standardizer, models


def predict(vector: Sequence[float], standardizer: Standardizer,
            models: Sequence[GaussianClassModel]) -> Prediction:
    if len(models) < 2:
        raise ValueError("Multi-hypothesis GLRT requires at least two class models")
    standardized = standardizer.transform(vector)
    scores = tuple((model.label, model.log_likelihood(standardized)) for model in models)
    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    margin = 2 * (ranked[0][1] - ranked[1][1])
    return Prediction(ranked[0][0], margin, scores)


def classification_metrics(actual: Sequence[Hashable], predicted: Sequence[Hashable],
                           class_order: Sequence[Hashable]) -> dict[str, object]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted labels must be non-empty and aligned")
    matrix = [[sum(truth == row and guess == column for truth, guess in zip(actual, predicted))
               for column in class_order] for row in class_order]
    per_class: dict[str, dict[str, float | int]] = {}
    recalls, f1_scores = [], []
    for index, label in enumerate(class_order):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        recall = tp / support if support else 0.0
        precision = tp / predicted_count if predicted_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall); f1_scores.append(f1)
        per_class[str(label)] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return {
        "windows": len(actual),
        "accuracy": sum(truth == guess for truth, guess in zip(actual, predicted)) / len(actual),
        "balanced_accuracy": fmean(recalls),
        "macro_f1": fmean(f1_scores),
        "class_order": [str(label) for label in class_order],
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def read_uci_partition(root: Path, partition: str) -> tuple[list[tuple[tuple[float, float, float], ...]], list[str]]:
    signal_dir = root / partition / "Inertial Signals"
    axes: list[list[list[float]]] = []
    for axis in ("x", "y", "z"):
        with (signal_dir / f"total_acc_{axis}_{partition}.txt").open(encoding="utf-8") as handle:
            axes.append([[float(value) for value in line.split()] for line in handle if line.strip()])
    numeric_labels = [int(value) for value in (root / partition / f"y_{partition}.txt").read_text(encoding="utf-8").split()]
    if not (len(axes[0]) == len(axes[1]) == len(axes[2]) == len(numeric_labels)):
        raise ValueError("UCI axes and labels are not aligned")
    windows = [tuple(zip(axes[0][index], axes[1][index], axes[2][index])) for index in range(len(numeric_labels))]
    labels = [UCI_ACTIVITIES[label - 1] for label in numeric_labels]
    return windows, labels


def read_wisdm_windows(path: Path) -> tuple[list[tuple[int, str, tuple[tuple[float, float, float], ...]]], Counter[str]]:
    windows: list[tuple[int, str, tuple[tuple[float, float, float], ...]]] = []
    discarded: Counter[str] = Counter()
    active_key: tuple[int, str] | None = None
    samples: list[tuple[float, float, float]] = []

    def flush() -> None:
        nonlocal samples
        if active_key is None:
            return
        while len(samples) >= WISDM_WINDOW_SIZE:
            windows.append((active_key[0], active_key[1], tuple(samples[:WISDM_WINDOW_SIZE])))
            samples = samples[WISDM_WINDOW_SIZE:]
        if samples:
            discarded["incomplete_activity_run_samples"] += len(samples)
        samples = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            for record in line.strip().split(";"):
                if not record:
                    continue
                fields = record.split(",")
                if len(fields) != 6:
                    discarded["malformed_records"] += 1; flush(); active_key = None; continue
                try:
                    user = int(fields[0]); activity = fields[1]
                    acceleration = tuple(float(value) for value in fields[3:6])
                except ValueError:
                    discarded["non_numeric_records"] += 1; flush(); active_key = None; continue
                if activity not in WISDM_ACTIVITIES or not all(math.isfinite(value) for value in acceleration):
                    discarded["invalid_records"] += 1; flush(); active_key = None; continue
                key = (user, activity)
                if key != active_key:
                    flush(); active_key = key
                samples.append(acceleration)
        flush()
    return windows, discarded


def evaluate_dataset(name: str, train_windows: Sequence[Sequence[tuple[float, float, float]]],
                     train_labels: Sequence[Hashable], test_windows: Sequence[Sequence[tuple[float, float, float]]],
                     test_labels: Sequence[Hashable], class_order: Sequence[Hashable], sample_rate: float,
                     shrinkage: float, ridge: float, feature_set: str,
                     covariance_mode: str, output_dir: Path) -> dict[str, object]:
    print(f"\n{name}: extracting {len(train_windows)} training and {len(test_windows)} held-out feature vectors")
    train_features = [select_features(extract_features(window, sample_rate), feature_set) for window in train_windows]
    test_features = [select_features(extract_features(window, sample_rate), feature_set) for window in test_windows]
    standardizer, models = fit_models(train_features, train_labels, class_order, shrinkage, ridge, covariance_mode)
    predictions = [predict(vector, standardizer, models) for vector in test_features]
    predicted_labels = [prediction.label for prediction in predictions]
    metrics = classification_metrics(test_labels, predicted_labels, class_order)
    metrics.update({
        "dataset": name,
        "train_windows": len(train_windows),
        "feature_set": feature_set,
        "feature_names": [FEATURE_NAMES[index] for index in FEATURE_SETS[feature_set]],
        "covariance_mode": covariance_mode,
        "covariance_shrinkage": shrinkage,
        "covariance_ridge": ridge,
        "class_training_counts": {str(model.label): model.count for model in models},
    })
    stem = "uci_har" if name == "UCI HAR" else "wisdm"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{stem}_six_class_glrt_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    likelihood_fields = [f"log_likelihood_{label}" for label in class_order]
    with (output_dir / f"{stem}_six_class_glrt_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["window", "actual", "predicted", "correct", "twice_log_glr_margin", *likelihood_fields]
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader()
        for index, (actual, prediction) in enumerate(zip(test_labels, predictions)):
            score_map = dict(prediction.log_likelihoods)
            row = {"window": index, "actual": actual, "predicted": prediction.label,
                   "correct": actual == prediction.label, "twice_log_glr_margin": prediction.glr_margin}
            row.update({field: score_map[label] for field, label in zip(likelihood_fields, class_order)})
            writer.writerow(row)
    print(f"Accuracy={metrics['accuracy']:.2%}; balanced accuracy={metrics['balanced_accuracy']:.2%}; macro F1={metrics['macro_f1']:.2%}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("  " + " ".join(f"{str(label)[:8]:>8}" for label in class_order))
    for label, row in zip(class_order, metrics["confusion_matrix"]):
        print(f"  {str(label)[:8]:>8} " + " ".join(f"{value:8d}" for value in row))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uci", "wisdm", "all"), default="all")
    parser.add_argument("--uci-root", type=Path, default=PROJECT_ROOT / "Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset")
    parser.add_argument("--wisdm-file", type=Path, default=PROJECT_ROOT / "Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="full")
    parser.add_argument("--covariance-mode", choices=COVARIANCE_MODES, default="class_full")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if not 0 <= args.shrinkage <= 1 or args.ridge <= 0:
        parser.error("--shrinkage must be in [0,1] and --ridge must be positive")
    if args.dataset in ("uci", "all"):
        train_windows, train_labels = read_uci_partition(args.uci_root, "train")
        test_windows, test_labels = read_uci_partition(args.uci_root, "test")
        evaluate_dataset("UCI HAR", train_windows, train_labels, test_windows, test_labels,
                         UCI_ACTIVITIES, 50.0, args.shrinkage, args.ridge,
                         args.feature_set, args.covariance_mode, args.output_dir)
    if args.dataset in ("wisdm", "all"):
        windows, discarded = read_wisdm_windows(args.wisdm_file)
        train = [row for row in windows if row[0] not in WISDM_TEST_USERS]
        test = [row for row in windows if row[0] in WISDM_TEST_USERS]
        print("WISDM discarded during cleaning: " + ", ".join(f"{key}={value}" for key, value in sorted(discarded.items())))
        evaluate_dataset("WISDM", [row[2] for row in train], [row[1] for row in train],
                         [row[2] for row in test], [row[1] for row in test],
                         WISDM_ACTIVITIES, 20.0, args.shrinkage, args.ridge,
                         args.feature_set, args.covariance_mode, args.output_dir)


if __name__ == "__main__":
    main()
