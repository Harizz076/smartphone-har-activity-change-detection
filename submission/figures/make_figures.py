#!/usr/bin/env python3
"""Generate every analytical figure used in the final report and slide deck."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glrt.six_class_glrt import (  # noqa: E402
    UCI_ACTIVITIES,
    WISDM_TEST_USERS,
    read_uci_partition,
    read_wisdm_windows,
)


OUT = Path(__file__).resolve().parent / "generated"
BLUE = "#195AA6"
RED = "#BE1E2D"
GRAY = "#C7C9CC"
DARK_GRAY = "#59636E"
MAGENTA = "#D87CE6"
INK = "#20252B"
GRID = "#E6E8EB"
WHITE = "#FFFFFF"


STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.edgecolor": "#A8ADB3",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", dpi=300)
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def raw_signal_examples() -> None:
    data_root = ROOT / "Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset"
    windows, labels = read_uci_partition(data_root, "test")
    examples = {}
    for window, label in zip(windows, labels):
        examples.setdefault(label, window)
    display_names = {
        "WALKING": "Walking",
        "WALKING_UPSTAIRS": "Walking upstairs",
        "WALKING_DOWNSTAIRS": "Walking downstairs",
        "SITTING": "Sitting",
        "STANDING": "Standing",
        "LAYING": "Laying",
    }
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.65), sharex=True,
                                 layout="constrained")
        for panel, (axis, label) in enumerate(zip(axes.flat, UCI_ACTIVITIES)):
            values = np.asarray(examples[label])
            time = np.arange(len(values)) / 50.0
            for column, (color, name) in enumerate(((BLUE, "x"), (RED, "y"),
                                                     (DARK_GRAY, "z"))):
                axis.plot(time, values[:, column], color=color, linewidth=0.85,
                          label=name)
            axis.axhline(0.0, color=GRAY, linewidth=0.55, zorder=0)
            axis.set_title(display_names[label], loc="left", pad=3)
            axis.grid(True, color=GRID, linewidth=0.45)
            axis.text(0.98, 0.92, chr(ord("a") + panel), transform=axis.transAxes,
                      ha="right", va="top", color=DARK_GRAY, fontweight="bold")
            if panel % 3 == 0:
                axis.set_ylabel("Acceleration (g)")
            if panel >= 3:
                axis.set_xlabel("Time (s)")
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside upper center", ncol=3,
                   frameon=False)
        save_figure(fig, "raw_signal_examples")


def performance_overview() -> None:
    uci_gaussian = load_json("analysis/six_class_optimization/uci_har/selection_summary.json")["test_metrics"]
    wisdm_gaussian = load_json("analysis/six_class_optimization/wisdm/selection_summary.json")["test_metrics"]
    uci_student = load_json("analysis/robust_likelihoods/uci_har/selection_summary.json")["test_metrics"]
    wisdm_student = load_json("analysis/robust_likelihoods/wisdm/selection_summary.json")["test_metrics"]
    uci_final = load_json("glrt/results/improved/uci_har_metrics.json")
    wisdm_final = load_json("glrt/results/improved/wisdm_metrics.json")
    methods = ["Variance\nthreshold", "Gaussian\nGLR", "Student-t\nGLR", "Final\nconfiguration"]
    values = {
        "UCI HAR": [0.3787, uci_gaussian["accuracy"], uci_student["accuracy"],
                    uci_final["accuracy"]],
        "WISDM": [0.5811, wisdm_gaussian["accuracy"], wisdm_student["accuracy"],
                  wisdm_final["accuracy"]],
    }
    colors = [GRAY, DARK_GRAY, RED, BLUE]
    hatches = ["///", "", "\\\\", "..."]
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85), sharey=True,
                                 layout="constrained")
        for axis, (dataset, scores) in zip(axes, values.items()):
            bars = axis.bar(np.arange(len(methods)), np.asarray(scores) * 100,
                            color=colors, width=0.72, edgecolor=INK, linewidth=0.55)
            for bar, hatch, score in zip(bars, hatches, scores):
                bar.set_hatch(hatch)
                axis.text(bar.get_x() + bar.get_width() / 2, score * 100 + 1.4,
                          f"{score * 100:.1f}", ha="center", va="bottom",
                          fontsize=7.5, fontweight="bold")
            axis.set_title(dataset)
            axis.set_xticks(np.arange(len(methods)), methods)
            axis.set_ylim(0, 100)
            axis.set_yticks(np.arange(0, 101, 20))
            axis.grid(axis="y", color=GRID, linewidth=0.55)
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Held-out accuracy (%)")
        fig.suptitle("A single variance feature does not support six-class recognition",
                     fontsize=11, fontweight="bold")
        save_figure(fig, "performance_overview")


def confusion_matrices() -> None:
    datasets = [
        ("UCI HAR", load_json("glrt/results/improved/uci_har_metrics.json")),
        ("WISDM", load_json("glrt/results/improved/wisdm_metrics.json")),
    ]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "signature_blue", [WHITE, "#DCE6F3", BLUE]
    )
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55), layout="constrained")
        for axis, (name, metrics) in zip(axes, datasets):
            matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            image = axis.imshow(normalized * 100, vmin=0, vmax=100, cmap=cmap,
                                interpolation="nearest", aspect="equal")
            labels = [str(value).replace("WALKING_", "W_").replace("ING", "")
                      for value in metrics["class_order"]]
            if name == "UCI HAR":
                labels = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]
            else:
                labels = ["Walk", "Jog", "Up", "Down", "Sit", "Stand"]
            axis.set_xticks(range(6), labels, rotation=35, ha="right")
            axis.set_yticks(range(6), labels)
            axis.set_xlabel("Predicted activity")
            axis.set_ylabel("True activity")
            axis.set_title(name)
            for row in range(6):
                for column in range(6):
                    value = normalized[row, column] * 100
                    color = WHITE if value >= 55 else (RED if row != column and value >= 10 else INK)
                    axis.text(column, row, f"{value:.0f}", ha="center", va="center",
                              color=color, fontsize=7.2,
                              fontweight="bold" if row == column else "normal")
        colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("Row-normalized rate (%)")
        fig.suptitle("Most remaining errors occur between physically similar activities",
                     fontsize=11, fontweight="bold")
        save_figure(fig, "confusion_matrices")


def _read_prediction_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def subject_variation() -> None:
    raw_path = ROOT / "Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    windows, _ = read_wisdm_windows(raw_path)
    test_groups = [row[0] for row in windows if row[0] in WISDM_TEST_USERS]
    original_rows = _read_prediction_rows(
        "analysis/six_class_optimization/wisdm/optimized_test_predictions.csv"
    )
    final_rows = _read_prediction_rows("glrt/results/improved/wisdm_predictions.csv")
    if not (len(test_groups) == len(original_rows) == len(final_rows)):
        raise ValueError("WISDM predictions and subject identifiers are not aligned")
    original = defaultdict(list)
    improved = defaultdict(list)
    for subject, old, new in zip(test_groups, original_rows, final_rows):
        original[subject].append(old["correct"] == "True")
        improved[subject].append(new["correct"] == "True")
    subjects = sorted(original)
    old_values = np.asarray([np.mean(original[subject]) for subject in subjects]) * 100
    new_values = np.asarray([np.mean(improved[subject]) for subject in subjects]) * 100
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.16, 3.05), layout="constrained")
        x = np.arange(len(subjects))
        for index in range(len(subjects)):
            ax.plot([x[index], x[index]], [old_values[index], new_values[index]],
                    color=GRAY, linewidth=1.2, zorder=1)
        ax.scatter(x, old_values, color=RED, marker="s", s=28, label="Original Gaussian GLR",
                   edgecolor=WHITE, linewidth=0.5, zorder=2)
        ax.scatter(x, new_values, color=BLUE, marker="o", s=34, label="Gravity-aligned RDA",
                   edgecolor=WHITE, linewidth=0.5, zorder=3)
        ax.axhline(np.mean(old_values), color=RED, linestyle="--", linewidth=0.8, alpha=0.8)
        ax.axhline(np.mean(new_values), color=BLUE, linestyle="-", linewidth=0.8, alpha=0.8)
        ax.set_xticks(x, [str(subject) for subject in subjects])
        ax.set_xlabel("Held-out WISDM subject")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(20, 102)
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.legend(frameon=False, ncol=2, loc="lower right")
        ax.set_title("WISDM gains are uneven across subjects", loc="left")
        fig.suptitle("Subject variation remains larger than the average model gain",
                     fontsize=11, fontweight="bold")
        save_figure(fig, "subject_variation")


def confidence_coverage() -> None:
    datasets = [
        ("UCI HAR", BLUE, _read_prediction_rows("glrt/results/improved/uci_har_predictions.csv")),
        ("WISDM", RED, _read_prediction_rows("glrt/results/improved/wisdm_predictions.csv")),
    ]
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.9, 3.0), layout="constrained")
        for name, color, rows in datasets:
            ordered = sorted(rows, key=lambda row: float(row["twice_log_glr_margin"]),
                             reverse=True)
            coverage, accuracy = [], []
            for percent in range(10, 101, 10):
                count = math.ceil(len(ordered) * percent / 100)
                retained = ordered[:count]
                coverage.append(percent)
                accuracy.append(100 * np.mean([row["correct"] == "True" for row in retained]))
            marker = "o" if name == "UCI HAR" else "s"
            ax.plot(coverage, accuracy, color=color, marker=marker, linewidth=1.7,
                    markersize=4.2, label=name)
        ax.set_xlabel("Retained windows with largest GLR margins (%)")
        ax.set_ylabel("Accuracy among retained windows (%)")
        ax.set_xticks(range(10, 101, 10))
        ax.set_ylim(70, 101)
        ax.grid(True, color=GRID, linewidth=0.55)
        ax.legend(frameon=False)
        ax.set_title("GLR margin supports rejection better on UCI HAR", loc="left")
        save_figure(fig, "confidence_coverage")


def rda_heatmaps() -> None:
    datasets = [
        ("UCI HAR", ROOT / "analysis/rda_glrt/uci_har/configuration_search.csv"),
        ("WISDM", ROOT / "analysis/rda_glrt/wisdm/configuration_search.csv"),
    ]
    lambdas = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
    gammas = [0.0, 0.01, 0.03, 0.10, 0.25, 0.50, 0.75, 1.0]
    matrices = []
    for _, path in datasets:
        matrix = np.full((len(gammas), len(lambdas)), np.nan)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                i = gammas.index(float(row["eigenvalue_shrinkage_gamma"]))
                j = lambdas.index(float(row["covariance_mixing_lambda"]))
                matrix[i, j] = 100 * float(row["cv_balanced_accuracy"])
        matrices.append(matrix)
    low = min(np.nanmin(matrix) for matrix in matrices)
    high = max(np.nanmax(matrix) for matrix in matrices)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "signature", ["#F2F3F4", "#AFC5DF", BLUE]
    )
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), layout="constrained")
        for axis, (dataset, _), matrix in zip(axes, datasets, matrices):
            image = axis.imshow(matrix, vmin=low, vmax=high, cmap=cmap,
                                interpolation="nearest", aspect="auto")
            axis.set_xticks(range(len(lambdas)), [f"{value:g}" for value in lambdas])
            axis.set_yticks(range(len(gammas)), [f"{value:g}" for value in gammas])
            axis.set_xlabel(r"Class-to-pooled mixing $\lambda$")
            axis.set_ylabel(r"Eigenvalue shrinkage $\gamma$")
            axis.set_title(dataset)
            best = np.unravel_index(np.nanargmax(matrix), matrix.shape)
            axis.scatter(best[1], best[0], marker="o", facecolors="none",
                         edgecolors=MAGENTA, s=105, linewidths=2.0)
        colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("Training-fold balanced accuracy (%)")
        fig.suptitle("Covariance pooling does not improve the original 15-feature model",
                     fontsize=11, fontweight="bold")
        save_figure(fig, "rda_heatmaps")


def main() -> None:
    raw_signal_examples()
    performance_overview()
    confusion_matrices()
    subject_variation()
    confidence_coverage()
    rda_heatmaps()
    manifest = {
        "palette": {"blue": BLUE, "red": RED, "gray": GRAY,
                    "magenta": MAGENTA},
        "outputs": [
            "raw_signal_examples", "performance_overview", "confusion_matrices",
            "subject_variation", "confidence_coverage", "rda_heatmaps",
        ],
        "formats": ["pdf", "png"],
        "notes": "All figures use held-out results already stored in the repository. Bars start at zero. Confusion matrices are normalized within true class. Subject is the unit in the paired WISDM plot.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
