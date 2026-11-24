#!/usr/bin/env python3
"""Subject-disjoint evaluation of Friedman's two-parameter RDA GLRT."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.optimize_six_class_glrt import (  # noqa: E402
    Dataset,
    group_folds,
    read_uci,
    read_wisdm,
)
from glrt.rda_glrt import fit_rda_models  # noqa: E402
from glrt.six_class_glrt import classification_metrics, predict  # noqa: E402


OUTPUT = ROOT / "analysis/rda_glrt"
LAMBDAS = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
GAMMAS = (0.0, 0.01, 0.03, 0.10, 0.25, 0.50, 0.75, 1.0)
FOLDS = 3
RIDGE = 1e-6
REFERENCE_GAMMA = {"UCI HAR": 0.03, "WISDM": 0.75}


@dataclass(frozen=True)
class Configuration:
    covariance_mixing: float
    eigenvalue_shrinkage: float


def cross_validate(dataset: Dataset, assignment: dict[int, int],
                   config: Configuration) -> dict[str, object]:
    predictions: list[str | None] = [None] * len(dataset.train_features)
    for fold in range(FOLDS):
        train_indices = [index for index, group in enumerate(dataset.train_groups)
                         if assignment[group] != fold]
        valid_indices = [index for index, group in enumerate(dataset.train_groups)
                         if assignment[group] == fold]
        standardizer, models = fit_rda_models(
            [dataset.train_features[index] for index in train_indices],
            [dataset.train_labels[index] for index in train_indices],
            dataset.class_order,
            config.covariance_mixing,
            config.eigenvalue_shrinkage,
            RIDGE,
        )
        for index in valid_indices:
            predictions[index] = str(predict(
                dataset.train_features[index], standardizer, models
            ).label)
    if any(value is None for value in predictions):
        raise AssertionError("Cross-validation did not predict every window")
    metrics = classification_metrics(dataset.train_labels, predictions,
                                     dataset.class_order)
    return {"config": config, "metrics": metrics}


def rank_key(result: dict[str, object]) -> tuple[float, float, float]:
    metrics = result["metrics"]
    return (float(metrics["balanced_accuracy"]),
            float(metrics["macro_f1"]), float(metrics["accuracy"]))


def fit_test(dataset: Dataset, config: Configuration) -> dict[str, object]:
    standardizer, models = fit_rda_models(
        dataset.train_features,
        dataset.train_labels,
        dataset.class_order,
        config.covariance_mixing,
        config.eigenvalue_shrinkage,
        RIDGE,
    )
    predictions = [str(predict(vector, standardizer, models).label)
                   for vector in dataset.test_features]
    metrics = classification_metrics(dataset.test_labels, predictions,
                                     dataset.class_order)
    return {"metrics": metrics, "predictions": predictions}


def subject_accuracies(dataset: Dataset, predictions: list[str]) -> dict[int, float]:
    values = {}
    for subject in sorted(set(dataset.test_groups)):
        indices = [index for index, group in enumerate(dataset.test_groups)
                   if group == subject]
        values[subject] = fmean(
            dataset.test_labels[index] == predictions[index] for index in indices
        )
    return values


def bootstrap_mean_delta(deltas: dict[int, float], iterations: int = 10000,
                         seed: int = 1989) -> list[float]:
    subjects = np.array(sorted(deltas), dtype=int)
    values = np.array([deltas[int(subject)] for subject in subjects])
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for index in range(iterations):
        estimates[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5))]


def evaluate(dataset: Dataset, output: Path) -> dict[str, object]:
    assignment, fold_sizes = group_folds(
        dataset.train_groups, dataset.train_labels, dataset.class_order, FOLDS
    )
    results = []
    for covariance_mixing in LAMBDAS:
        for eigenvalue_shrinkage in GAMMAS:
            config = Configuration(covariance_mixing, eigenvalue_shrinkage)
            results.append(cross_validate(dataset, assignment, config))
    ranked = sorted(results, key=rank_key, reverse=True)
    selected = ranked[0]
    selected_test = fit_test(dataset, selected["config"])

    reference_config = Configuration(0.0, REFERENCE_GAMMA[dataset.name])
    reference_cv = next(result for result in results
                        if result["config"] == reference_config)
    reference_test = fit_test(dataset, reference_config)
    selected_subjects = subject_accuracies(dataset, selected_test["predictions"])
    reference_subjects = subject_accuracies(dataset, reference_test["predictions"])
    deltas = {subject: selected_subjects[subject] - reference_subjects[subject]
              for subject in selected_subjects}

    directory = output / ("uci_har" if dataset.name == "UCI HAR" else "wisdm")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "configuration_search.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["rank", "covariance_mixing_lambda",
                         "eigenvalue_shrinkage_gamma", "cv_accuracy",
                         "cv_balanced_accuracy", "cv_macro_f1"])
        for rank, result in enumerate(ranked, 1):
            config, metrics = result["config"], result["metrics"]
            writer.writerow([rank, config.covariance_mixing,
                             config.eigenvalue_shrinkage, metrics["accuracy"],
                             metrics["balanced_accuracy"], metrics["macro_f1"]])
    with (directory / "subject_accuracy_comparison.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["subject", "reference_accuracy", "rda_accuracy",
                         "accuracy_delta"])
        for subject in sorted(deltas):
            writer.writerow([subject, reference_subjects[subject],
                             selected_subjects[subject], deltas[subject]])

    summary = {
        "dataset": dataset.name,
        "status": "exploratory_post_development_evaluation",
        "selection": "three subject-disjoint folds; maximize balanced accuracy, then macro F1, then accuracy",
        "grid": {"covariance_mixing_lambda": list(LAMBDAS),
                 "eigenvalue_shrinkage_gamma": list(GAMMAS)},
        "fold_window_counts": fold_sizes,
        "selected": {
            "covariance_mixing_lambda": selected["config"].covariance_mixing,
            "eigenvalue_shrinkage_gamma": selected["config"].eigenvalue_shrinkage,
            "cv_metrics": selected["metrics"],
            "test_metrics": selected_test["metrics"],
        },
        "reference": {
            "covariance_mixing_lambda": 0.0,
            "eigenvalue_shrinkage_gamma": reference_config.eigenvalue_shrinkage,
            "cv_metrics": reference_cv["metrics"],
            "test_metrics": reference_test["metrics"],
        },
        "test_comparison": {
            "pooled_accuracy_delta": (
                selected_test["metrics"]["accuracy"]
                - reference_test["metrics"]["accuracy"]
            ),
            "mean_subject_accuracy_delta": fmean(deltas.values()),
            "mean_subject_accuracy_delta_bootstrap_95_ci": bootstrap_mean_delta(deltas),
            "subjects_improved": sum(value > 0 for value in deltas.values()),
            "subjects_tied": sum(value == 0 for value in deltas.values()),
            "subjects_worsened": sum(value < 0 for value in deltas.values()),
        },
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"{dataset.name}: selected lambda={selected['config'].covariance_mixing:g}, "
        f"gamma={selected['config'].eigenvalue_shrinkage:g}; "
        f"CV balanced={selected['metrics']['balanced_accuracy']:.2%}; "
        f"test accuracy={selected_test['metrics']['accuracy']:.2%} "
        f"(reference {reference_test['metrics']['accuracy']:.2%})"
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
        summaries.append(evaluate(read_uci(), args.output))
    if args.dataset in ("wisdm", "all"):
        summaries.append(evaluate(read_wisdm(), args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
