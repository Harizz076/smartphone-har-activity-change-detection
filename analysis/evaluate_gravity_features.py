#!/usr/bin/env python3
"""Evaluate gravity-aligned features without touching held-out labels in selection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate_rda_glrt import (  # noqa: E402
    Configuration,
    GAMMAS,
    LAMBDAS,
    RIDGE,
    cross_validate,
    fit_test,
    rank_key,
)
from analysis.optimize_six_class_glrt import (  # noqa: E402
    Dataset,
    group_folds,
)
from glrt.gravity_features import (  # noqa: E402
    GRAVITY_FEATURE_NAMES,
    extract_gravity_features,
)
from glrt.six_class_glrt import (  # noqa: E402
    FEATURE_NAMES,
    UCI_ACTIVITIES,
    WISDM_ACTIVITIES,
    WISDM_TEST_USERS,
    extract_features,
    read_uci_partition,
    read_wisdm_windows,
)


OUTPUT = ROOT / "analysis/gravity_features"
REFERENCE_GAMMA = {"UCI HAR": 0.03, "WISDM": 0.75}


def datasets_uci() -> tuple[Dataset, Dataset]:
    root = ROOT / "Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset"
    train_windows, train_labels = read_uci_partition(root, "train")
    test_windows, test_labels = read_uci_partition(root, "test")
    train_groups = tuple(int(value) for value in
                         (root / "train/subject_train.txt").read_text().split())
    test_groups = tuple(int(value) for value in
                        (root / "test/subject_test.txt").read_text().split())
    original = Dataset(
        "UCI HAR", UCI_ACTIVITIES,
        tuple(extract_features(window, 50.0) for window in train_windows),
        tuple(train_labels), train_groups,
        tuple(extract_features(window, 50.0) for window in test_windows),
        tuple(test_labels), test_groups,
    )
    augmented = Dataset(
        "UCI HAR", UCI_ACTIVITIES,
        tuple(extract_gravity_features(window, 50.0) for window in train_windows),
        tuple(train_labels), train_groups,
        tuple(extract_gravity_features(window, 50.0) for window in test_windows),
        tuple(test_labels), test_groups,
    )
    return original, augmented


def datasets_wisdm() -> tuple[Dataset, Dataset]:
    path = ROOT / "Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    windows, _ = read_wisdm_windows(path)
    train = [row for row in windows if row[0] not in WISDM_TEST_USERS]
    test = [row for row in windows if row[0] in WISDM_TEST_USERS]
    original = Dataset(
        "WISDM", WISDM_ACTIVITIES,
        tuple(extract_features(row[2], 20.0) for row in train),
        tuple(row[1] for row in train), tuple(row[0] for row in train),
        tuple(extract_features(row[2], 20.0) for row in test),
        tuple(row[1] for row in test), tuple(row[0] for row in test),
    )
    augmented = Dataset(
        "WISDM", WISDM_ACTIVITIES,
        tuple(extract_gravity_features(row[2], 20.0) for row in train),
        tuple(row[1] for row in train), tuple(row[0] for row in train),
        tuple(extract_gravity_features(row[2], 20.0) for row in test),
        tuple(row[1] for row in test), tuple(row[0] for row in test),
    )
    return original, augmented


def evaluate(original: Dataset, augmented: Dataset, output: Path) -> dict[str, object]:
    assignment, _ = group_folds(
        original.train_groups, original.train_labels, original.class_order
    )
    reference_config = Configuration(0.0, REFERENCE_GAMMA[original.name])
    reference_cv = cross_validate(original, assignment, reference_config)
    reference_test = fit_test(original, reference_config)

    augmented_results = [
        cross_validate(augmented, assignment, Configuration(covariance_mixing, gamma))
        for covariance_mixing in LAMBDAS
        for gamma in GAMMAS
    ]
    best_augmented = max(augmented_results, key=rank_key)
    selected_family = ("gravity_aligned_augmented"
                       if rank_key(best_augmented) > rank_key(reference_cv)
                       else "original")
    selected_cv = best_augmented if selected_family == "gravity_aligned_augmented" else reference_cv
    selected_dataset = augmented if selected_family == "gravity_aligned_augmented" else original
    selected_test = fit_test(selected_dataset, selected_cv["config"])

    summary = {
        "dataset": original.name,
        "status": "exploratory_post_development_evaluation",
        "selection_protocol": "subject-disjoint training folds only",
        "original_feature_count": len(FEATURE_NAMES),
        "augmented_feature_count": len(FEATURE_NAMES) + len(GRAVITY_FEATURE_NAMES),
        "added_features": list(GRAVITY_FEATURE_NAMES),
        "reference": {
            "gamma": reference_config.eigenvalue_shrinkage,
            "cv_metrics": reference_cv["metrics"],
            "test_metrics": reference_test["metrics"],
        },
        "best_augmented": {
            "lambda": best_augmented["config"].covariance_mixing,
            "gamma": best_augmented["config"].eigenvalue_shrinkage,
            "cv_metrics": best_augmented["metrics"],
            "test_metrics": fit_test(augmented, best_augmented["config"])["metrics"],
        },
        "selected_family": selected_family,
        "selected_test_metrics": selected_test["metrics"],
    }
    directory = output / ("uci_har" if original.name == "UCI HAR" else "wisdm")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    print(
        f"{original.name}: augmented CV balanced="
        f"{best_augmented['metrics']['balanced_accuracy']:.2%} vs "
        f"{reference_cv['metrics']['balanced_accuracy']:.2%}; "
        f"selected={selected_family}; test accuracy="
        f"{selected_test['metrics']['accuracy']:.2%}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uci", "wisdm", "all"),
                        default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summaries = []
    if args.dataset in ("uci", "all"):
        summaries.append(evaluate(*datasets_uci(), args.output))
    if args.dataset in ("wisdm", "all"):
        summaries.append(evaluate(*datasets_wisdm(), args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
