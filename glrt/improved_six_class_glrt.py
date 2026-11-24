#!/usr/bin/env python3
"""Run the literature-reviewed six-class likelihood-ratio configurations."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Hashable, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glrt.gravity_features import GRAVITY_FEATURE_NAMES, extract_gravity_features
from glrt.rda_glrt import fit_rda_models
from glrt.six_class_glrt import (
    FEATURE_NAMES,
    PROJECT_ROOT,
    UCI_ACTIVITIES,
    WISDM_ACTIVITIES,
    WISDM_TEST_USERS,
    classification_metrics,
    extract_features,
    predict,
    read_uci_partition,
    read_wisdm_windows,
)


def evaluate(
    name: str,
    train_windows: Sequence[Sequence[tuple[float, float, float]]],
    train_labels: Sequence[Hashable],
    test_windows: Sequence[Sequence[tuple[float, float, float]]],
    test_labels: Sequence[Hashable],
    class_order: Sequence[Hashable],
    sample_rate: float,
    gravity_aligned: bool,
    covariance_mixing: float,
    eigenvalue_shrinkage: float,
    output: Path,
) -> dict[str, object]:
    extractor = extract_gravity_features if gravity_aligned else extract_features
    train_features = [extractor(window, sample_rate) for window in train_windows]
    test_features = [extractor(window, sample_rate) for window in test_windows]
    standardizer, models = fit_rda_models(
        train_features, train_labels, class_order,
        covariance_mixing, eigenvalue_shrinkage, 1e-6,
    )
    predictions = [predict(vector, standardizer, models) for vector in test_features]
    predicted = [result.label for result in predictions]
    metrics = classification_metrics(test_labels, predicted, class_order)
    metrics.update({
        "dataset": name,
        "status": "exploratory_post_development_evaluation",
        "gravity_aligned_features": gravity_aligned,
        "feature_names": list(FEATURE_NAMES)
        + (list(GRAVITY_FEATURE_NAMES) if gravity_aligned else []),
        "covariance_mixing_lambda": covariance_mixing,
        "eigenvalue_shrinkage_gamma": eigenvalue_shrinkage,
        "ridge": 1e-6,
    })
    output.mkdir(parents=True, exist_ok=True)
    stem = "uci_har" if name == "UCI HAR" else "wisdm"
    (output / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (output / f"{stem}_predictions.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = ["window", "actual", "predicted", "correct",
                  "twice_log_glr_margin"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, (actual, result) in enumerate(zip(test_labels, predictions)):
            writer.writerow({
                "window": index,
                "actual": actual,
                "predicted": result.label,
                "correct": actual == result.label,
                "twice_log_glr_margin": result.glr_margin,
            })
    print(
        f"{name}: accuracy={metrics['accuracy']:.2%}; "
        f"balanced accuracy={metrics['balanced_accuracy']:.2%}; "
        f"macro F1={metrics['macro_f1']:.2%}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uci", "wisdm", "all"),
                        default="all")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results/improved")
    args = parser.parse_args()

    if args.dataset in ("uci", "all"):
        root = PROJECT_ROOT / "Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset"
        train_windows, train_labels = read_uci_partition(root, "train")
        test_windows, test_labels = read_uci_partition(root, "test")
        # Gravity augmentation did not transfer on the held-out UCI subjects;
        # retain the training-CV-selected original representation.
        evaluate("UCI HAR", train_windows, train_labels, test_windows,
                 test_labels, UCI_ACTIVITIES, 50.0, False, 0.0, 0.03,
                 args.output)

    if args.dataset in ("wisdm", "all"):
        path = PROJECT_ROOT / "Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
        windows, _ = read_wisdm_windows(path)
        train = [row for row in windows if row[0] not in WISDM_TEST_USERS]
        test = [row for row in windows if row[0] in WISDM_TEST_USERS]
        evaluate("WISDM", [row[2] for row in train], [row[1] for row in train],
                 [row[2] for row in test], [row[1] for row in test],
                 WISDM_ACTIVITIES, 20.0, True, 0.25, 0.01, args.output)


if __name__ == "__main__":
    main()
