#!/usr/bin/env python3
"""Subject-disjoint model selection and visual audit for six-class GLRT.

The search never uses held-out test labels for configuration selection. It
exhaustively evaluates every declared feature-set, covariance, and shrinkage
combination using the same deterministic subject-disjoint folds.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Hashable, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glrt.six_class_glrt import (  # noqa: E402 - repository root is added above
    COVARIANCE_MODES,
    FEATURE_NAMES,
    FEATURE_SETS,
    UCI_ACTIVITIES,
    WISDM_ACTIVITIES,
    WISDM_TEST_USERS,
    classification_metrics,
    extract_features,
    fit_models,
    fit_standardizer,
    predict,
    read_uci_partition,
    read_wisdm_windows,
    select_features,
)


OUTPUT = ROOT / "analysis/six_class_optimization"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#555555")
TUNED_SHRINKAGES = (0.0, 0.01, 0.03, 0.10, 0.25, 0.50, 0.75, 1.0)
FOLDS = 3
RIDGE = 1e-6


@dataclass(frozen=True)
class Dataset:
    name: str
    class_order: tuple[str, ...]
    train_features: tuple[tuple[float, ...], ...]
    train_labels: tuple[str, ...]
    train_groups: tuple[int, ...]
    test_features: tuple[tuple[float, ...], ...]
    test_labels: tuple[str, ...]
    test_groups: tuple[int, ...]


@dataclass(frozen=True)
class Configuration:
    feature_set: str
    covariance_mode: str
    shrinkage: float

    @property
    def key(self) -> str:
        return f"{self.feature_set}|{self.covariance_mode}|s={self.shrinkage:g}"


def read_uci() -> Dataset:
    root = ROOT / "Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset"
    train_windows, train_labels = read_uci_partition(root, "train")
    test_windows, test_labels = read_uci_partition(root, "test")
    groups = tuple(int(value) for value in (root / "train/subject_train.txt").read_text(encoding="utf-8").split())
    test_groups = tuple(int(value) for value in (root / "test/subject_test.txt").read_text(encoding="utf-8").split())
    return Dataset("UCI HAR", UCI_ACTIVITIES,
                   tuple(extract_features(window, 50.0) for window in train_windows), tuple(train_labels), groups,
                   tuple(extract_features(window, 50.0) for window in test_windows), tuple(test_labels), test_groups)


def read_wisdm() -> Dataset:
    path = ROOT / "Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    windows, _ = read_wisdm_windows(path)
    train = [row for row in windows if row[0] not in WISDM_TEST_USERS]
    test = [row for row in windows if row[0] in WISDM_TEST_USERS]
    return Dataset("WISDM", WISDM_ACTIVITIES,
                   tuple(extract_features(row[2], 20.0) for row in train), tuple(row[1] for row in train),
                   tuple(row[0] for row in train),
                   tuple(extract_features(row[2], 20.0) for row in test), tuple(row[1] for row in test),
                   tuple(row[0] for row in test))


def group_folds(groups: Sequence[int], labels: Sequence[str], class_order: Sequence[str],
                folds: int = FOLDS) -> tuple[dict[int, int], list[int]]:
    """Greedily balance subject-disjoint folds by size and class composition."""
    group_counts: dict[int, dict[str, int]] = {}
    for group, label in zip(groups, labels):
        counts = group_counts.setdefault(group, {name: 0 for name in class_order})
        counts[label] += 1
    totals = {name: sum(counts[name] for counts in group_counts.values()) for name in class_order}
    target_size = len(groups) / folds
    target_class = {name: totals[name] / folds for name in class_order}
    fold_sizes = [0] * folds
    fold_classes = [{name: 0 for name in class_order} for _ in range(folds)]
    assignment: dict[int, int] = {}
    ordered = sorted(group_counts, key=lambda group: (-sum(group_counts[group].values()), group))
    for position, group in enumerate(ordered):
        counts = group_counts[group]
        if position < folds:
            fold = position
        else:
            def imbalance(candidate: int) -> tuple[float, int, int]:
                total_loss = 0.0
                for fold_index in range(folds):
                    added_size = sum(counts.values()) if fold_index == candidate else 0
                    projected_size = fold_sizes[fold_index] + added_size
                    total_loss += ((projected_size - target_size) / max(target_size, 1.0)) ** 2
                    for name in class_order:
                        added_class = counts[name] if fold_index == candidate else 0
                        projected_class = fold_classes[fold_index][name] + added_class
                        total_loss += ((projected_class - target_class[name]) /
                                       max(target_class[name], 1.0)) ** 2
                return total_loss, fold_sizes[candidate], candidate
            fold = min(range(folds), key=imbalance)
        assignment[group] = fold
        fold_sizes[fold] += sum(counts.values())
        for name in class_order:
            fold_classes[fold][name] += counts[name]
    return assignment, fold_sizes


def subset(vectors: Sequence[Sequence[float]], name: str) -> list[tuple[float, ...]]:
    return [select_features(vector, name) for vector in vectors]


def cross_validate(dataset: Dataset, config: Configuration,
                   assignment: dict[int, int]) -> dict[str, object]:
    vectors = subset(dataset.train_features, config.feature_set)
    predictions: list[str | None] = [None] * len(vectors)
    margins = [0.0] * len(vectors)
    for fold in range(FOLDS):
        train_indices = [index for index, group in enumerate(dataset.train_groups) if assignment[group] != fold]
        valid_indices = [index for index, group in enumerate(dataset.train_groups) if assignment[group] == fold]
        train_vectors = [vectors[index] for index in train_indices]
        train_labels = [dataset.train_labels[index] for index in train_indices]
        standardizer, models = fit_models(train_vectors, train_labels, dataset.class_order,
                                          config.shrinkage, RIDGE, config.covariance_mode)
        for index in valid_indices:
            result = predict(vectors[index], standardizer, models)
            predictions[index] = str(result.label)
            margins[index] = result.glr_margin
    if any(label is None for label in predictions):
        raise AssertionError("Cross-validation did not predict every training window")
    metrics = classification_metrics(dataset.train_labels, predictions, dataset.class_order)
    return {"config": config, "metrics": metrics, "predictions": predictions, "margins": margins}


def search(dataset: Dataset) -> tuple[list[dict[str, object]], dict[str, object], dict[int, int], list[int]]:
    assignment, fold_sizes = group_folds(dataset.train_groups, dataset.train_labels,
                                         dataset.class_order)
    results: list[dict[str, object]] = []
    total = len(FEATURE_SETS) * len(COVARIANCE_MODES) * len(TUNED_SHRINKAGES)
    print(f"{dataset.name}: exhaustive search — {total} configurations")
    for feature_set in FEATURE_SETS:
        for covariance_mode in COVARIANCE_MODES:
            for shrinkage in TUNED_SHRINKAGES:
                results.append(cross_validate(
                    dataset, Configuration(feature_set, covariance_mode, shrinkage), assignment
                ))
    ranked = sorted(results, key=result_key, reverse=True)
    return ranked, ranked[0], assignment, fold_sizes


def result_key(result: dict[str, object]) -> tuple[float, float, float]:
    metrics = result["metrics"]
    return metrics["balanced_accuracy"], metrics["macro_f1"], metrics["accuracy"]


def fit_final(dataset: Dataset, config: Configuration) -> dict[str, object]:
    train_vectors = subset(dataset.train_features, config.feature_set)
    test_vectors = subset(dataset.test_features, config.feature_set)
    standardizer, models = fit_models(train_vectors, dataset.train_labels, dataset.class_order,
                                      config.shrinkage, RIDGE, config.covariance_mode)
    results = [predict(vector, standardizer, models) for vector in test_vectors]
    predicted = [str(result.label) for result in results]
    metrics = classification_metrics(dataset.test_labels, predicted, dataset.class_order)
    return {"metrics": metrics, "predictions": predicted,
            "margins": [result.glr_margin for result in results],
            "log_likelihoods": [dict(result.log_likelihoods) for result in results]}


def write_tables(dataset: Dataset, ranked: list[dict[str, object]], selected: dict[str, object],
                 final: dict[str, object], assignment: dict[int, int], fold_sizes: list[int], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "configuration_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature_set", "dimensions", "covariance_mode", "shrinkage",
                                                       "validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1"])
        writer.writeheader()
        for rank, result in enumerate(ranked, 1):
            config, metrics = result["config"], result["metrics"]
            writer.writerow({"rank": rank, "feature_set": config.feature_set,
                             "dimensions": len(FEATURE_SETS[config.feature_set]), "covariance_mode": config.covariance_mode,
                             "shrinkage": config.shrinkage, "validation_accuracy": metrics["accuracy"],
                             "validation_balanced_accuracy": metrics["balanced_accuracy"], "validation_macro_f1": metrics["macro_f1"]})
    selected_config = selected["config"]
    fold_class_counts = {
        str(fold): {
            activity: sum(label == activity and assignment[group] == fold
                          for label, group in zip(dataset.train_labels, dataset.train_groups))
            for activity in dataset.class_order
        }
        for fold in range(FOLDS)
    }
    summary = {"dataset": dataset.name,
               "selection_objective": ["balanced_accuracy", "macro_f1", "accuracy"],
               "subject_folds": {str(group): fold for group, fold in sorted(assignment.items())},
               "fold_window_counts": fold_sizes,
               "fold_class_counts": fold_class_counts,
               "selected": {"feature_set": selected_config.feature_set, "feature_names": [FEATURE_NAMES[index] for index in FEATURE_SETS[selected_config.feature_set]],
                            "covariance_mode": selected_config.covariance_mode, "shrinkage": selected_config.shrinkage,
                            "ridge": RIDGE, "validation_metrics": selected["metrics"]},
               "test_metrics": final["metrics"]}
    (directory / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (directory / "optimized_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["window", "subject", "actual", "predicted", "correct", "twice_log_glr_margin", *[f"log_likelihood_{label}" for label in dataset.class_order]]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index, (subject, actual, predicted, margin, likelihoods) in enumerate(zip(dataset.test_groups, dataset.test_labels, final["predictions"], final["margins"], final["log_likelihoods"])):
            row = {"window": index, "subject": subject, "actual": actual, "predicted": predicted, "correct": actual == predicted, "twice_log_glr_margin": margin}
            row.update({f"log_likelihood_{label}": likelihoods[label] for label in dataset.class_order}); writer.writerow(row)
    standardized = fit_standardizer(dataset.train_features)
    z = [standardized.transform(vector) for vector in dataset.train_features]
    profiles = [[fmean(vector[index] for vector,label in zip(z,dataset.train_labels) if label==activity)
                 for index in range(len(FEATURE_NAMES))] for activity in dataset.class_order]
    with (directory / "class_feature_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["activity", *FEATURE_NAMES])
        for activity, values in zip(dataset.class_order, profiles): writer.writerow([activity, *values])
    correlations = matrix_correlation(z)
    with (directory / "strongest_feature_correlations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["feature_1", "feature_2", "pearson_r", "absolute_r"])
        pairs = [(FEATURE_NAMES[left], FEATURE_NAMES[right], correlations[left][right])
                 for left in range(len(FEATURE_NAMES)) for right in range(left+1,len(FEATURE_NAMES))]
        for left,right,value in sorted(pairs,key=lambda row:abs(row[2]),reverse=True): writer.writerow([left,right,value,abs(value)])
    with (directory / "accuracy_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["coverage", "windows", "accuracy", "minimum_margin"])
        ordered=sorted(zip(final["margins"],dataset.test_labels,final["predictions"]),reverse=True)
        for percent in range(10,101,10):
            count=math.ceil(len(ordered)*percent/100); retained=ordered[:count]
            writer.writerow([percent/100,count,sum(actual==predicted for _,actual,predicted in retained)/count,retained[-1][0]])
    with (directory / "test_subject_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["subject", "windows", "accuracy", "balanced_accuracy", "macro_f1"])
        for subject in sorted(set(dataset.test_groups)):
            indices=[index for index,group in enumerate(dataset.test_groups) if group==subject]
            actual=[dataset.test_labels[index] for index in indices]
            predicted=[final["predictions"][index] for index in indices]
            present=[activity for activity in dataset.class_order if activity in set(actual)]
            recalls=[]; f1_scores=[]
            for activity in present:
                true_positive=sum(truth==activity and guess==activity for truth,guess in zip(actual,predicted))
                support=sum(truth==activity for truth in actual)
                predicted_count=sum(guess==activity for guess in predicted)
                recall=true_positive/support
                precision=true_positive/predicted_count if predicted_count else 0.0
                recalls.append(recall)
                f1_scores.append(2*precision*recall/(precision+recall) if precision+recall else 0.0)
            writer.writerow([subject,len(indices),sum(a==p for a,p in zip(actual,predicted))/len(actual),
                             fmean(recalls),fmean(f1_scores)])


def svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            f'<title>{html.escape(title)}</title><desc>{html.escape(description)}</desc>',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:18px;font-weight:bold}.label{font-size:12px}.small{font-size:10px}</style>',
            f'<text class="title" x="{width/2}" y="27" text-anchor="middle">{html.escape(title)}</text>']


def save_svg(lines: list[str], path: Path) -> None:
    path.write_text("\n".join(lines + ["</svg>"]), encoding="utf-8")


def plot_ranking(ranked: list[dict[str, object]], path: Path) -> None:
    top = ranked[:20]; width, height = 1180, 650; x0, y0, chart_w, row_h = 410, 55, 700, 27
    lines = svg_start(width, height, "Top validation configurations", "Top twenty configurations ranked by subject-disjoint validation balanced accuracy.")
    minimum = min(result["metrics"]["balanced_accuracy"] for result in top) - .01
    for row, result in enumerate(top):
        config, score = result["config"], result["metrics"]["balanced_accuracy"]
        y = y0 + row * row_h; bar = chart_w * (score - minimum) / (1 - minimum)
        lines += [f'<text class="small" x="{x0-8}" y="{y+16}" text-anchor="end">{html.escape(config.key)}</text>',
                  f'<rect x="{x0}" y="{y+3}" width="{bar:.1f}" height="18" fill="{COLORS[row%len(COLORS)]}"/>',
                  f'<text class="small" x="{x0+bar+5:.1f}" y="{y+16}">{score:.3f}</text>']
    lines.append(f'<text class="label" x="{x0+chart_w/2}" y="{height-18}" text-anchor="middle">Out-of-fold balanced accuracy (axis begins at {minimum:.2f})</text>')
    save_svg(lines, path)


def heat_color(value: float, low: float, high: float) -> str:
    ratio = 0.5 if high == low else max(0.0, min(1.0, (value-low)/(high-low)))
    return f"rgb({int(245-170*ratio)},{int(248-75*ratio)},{int(250-25*ratio)})"


def diverging_color(value: float, limit: float) -> str:
    """Color negative values blue, zero white, and positive values vermilion."""
    ratio = max(-1.0, min(1.0, value / max(limit, 1e-12)))
    white = (247, 247, 247)
    endpoint = (0, 114, 178) if ratio < 0 else (213, 94, 0)
    amount = abs(ratio)
    rgb = tuple(round(white[index] + amount * (endpoint[index] - white[index])) for index in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def plot_family_heatmap(ranked: list[dict[str, object]], path: Path) -> None:
    best = {}
    for result in ranked:
        config = result["config"]; key = (config.feature_set, config.covariance_mode)
        best[key] = max(best.get(key, 0.0), result["metrics"]["balanced_accuracy"])
    values = list(best.values()); low, high = min(values), max(values)
    width, height, cell_w, cell_h = 980, 550, 155, 62; x0, y0 = 270, 95
    lines = svg_start(width, height, "Best validation score by model family", "Heatmap of best observed validation balanced accuracy for each feature and covariance family.")
    for col, mode in enumerate(COVARIANCE_MODES):
        lines.append(f'<text class="small" x="{x0+col*cell_w+cell_w/2}" y="{y0-12}" text-anchor="middle">{mode}</text>')
    for row, feature_set in enumerate(FEATURE_SETS):
        lines.append(f'<text class="label" x="{x0-10}" y="{y0+row*cell_h+cell_h/2+4}" text-anchor="end">{feature_set} ({len(FEATURE_SETS[feature_set])}D)</text>')
        for col, mode in enumerate(COVARIANCE_MODES):
            value = best[(feature_set, mode)]
            lines += [f'<rect x="{x0+col*cell_w}" y="{y0+row*cell_h}" width="{cell_w}" height="{cell_h}" fill="{heat_color(value,low,high)}" stroke="white"/>',
                      f'<text class="label" x="{x0+col*cell_w+cell_w/2}" y="{y0+row*cell_h+cell_h/2+4}" text-anchor="middle">{value:.3f}</text>']
    lines.append(f'<text class="small" x="{x0}" y="{height-20}">Shared color scale: {low:.3f} to {high:.3f}; higher is darker.</text>')
    save_svg(lines, path)


def plot_shrinkage(ranked: list[dict[str, object]], selected: Configuration, path: Path) -> None:
    points = sorted((result["config"].shrinkage, result["metrics"]["balanced_accuracy"])
                    for result in ranked if result["config"].feature_set == selected.feature_set
                    and result["config"].covariance_mode == selected.covariance_mode)
    width, height, x0, y0, chart_w, chart_h = 850, 480, 90, 65, 690, 330
    low, high = min(score for _, score in points)-.01, max(score for _, score in points)+.01
    lines = svg_start(width, height, "Shrinkage sensitivity for selected family", "Validation balanced accuracy across covariance shrinkage values.")
    coordinates=[]
    for shrinkage, score in points:
        x=x0+chart_w*shrinkage; y=y0+chart_h*(high-score)/(high-low); coordinates.append((x,y))
        lines += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#0072B2"/>', f'<text class="small" x="{x:.1f}" y="{y-9:.1f}" text-anchor="middle">{score:.3f}</text>']
    lines += [f'<polyline fill="none" stroke="#0072B2" stroke-width="2" points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in coordinates)}"/>',
              f'<rect x="{x0}" y="{y0}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#777"/>',
              f'<text class="label" x="{x0+chart_w/2}" y="{height-20}" text-anchor="middle">Covariance shrinkage λ</text>',
              f'<text class="small" x="{x0}" y="{y0+chart_h+18}">0</text>', f'<text class="small" x="{x0+chart_w}" y="{y0+chart_h+18}" text-anchor="end">1</text>']
    save_svg(lines,path)


def plot_confusion(metrics: dict[str, object], path: Path) -> None:
    labels=metrics["class_order"]; matrix=metrics["confusion_matrix"]; width,height,cell=900,700,78; x0,y0=290,105
    lines=svg_start(width,height,"Optimized held-out confusion matrix","Rows are actual activities and columns are predicted activities; cells show count and row percentage.")
    for col,label in enumerate(labels): lines.append(f'<text class="small" x="{x0+col*cell+cell/2}" y="{y0-10}" text-anchor="end" transform="rotate(-35 {x0+col*cell+cell/2} {y0-10})">{html.escape(label)}</text>')
    for row,label in enumerate(labels):
        total=sum(matrix[row]); lines.append(f'<text class="small" x="{x0-8}" y="{y0+row*cell+cell/2+4}" text-anchor="end">{html.escape(label)}</text>')
        for col,value in enumerate(matrix[row]):
            fraction=value/total if total else 0
            lines += [f'<rect x="{x0+col*cell}" y="{y0+row*cell}" width="{cell}" height="{cell}" fill="{heat_color(fraction,0,1)}" stroke="white"/>',
                      f'<text class="small" x="{x0+col*cell+cell/2}" y="{y0+row*cell+cell/2-3}" text-anchor="middle">{value}</text>',
                      f'<text class="small" x="{x0+col*cell+cell/2}" y="{y0+row*cell+cell/2+13}" text-anchor="middle">{fraction:.0%}</text>']
    save_svg(lines,path)


def plot_recalls(validation: dict[str, object], test: dict[str, object], path: Path) -> None:
    labels=test["class_order"]; width,height,x0,y0,chart_w=1100,520,230,70,790
    lines=svg_start(width,height,"Per-class recall: validation and held-out test","Grouped bars compare subject-disjoint validation recall with held-out recall; values are printed on bars.")
    for row,label in enumerate(labels):
        y=y0+row*62; vr=validation["per_class"][label]["recall"]; tr=test["per_class"][label]["recall"]
        lines += [f'<text class="label" x="{x0-10}" y="{y+24}" text-anchor="end">{html.escape(label)}</text>',
                  f'<rect x="{x0}" y="{y+4}" width="{chart_w*vr:.1f}" height="18" fill="#0072B2"/>',
                  f'<rect x="{x0}" y="{y+27}" width="{chart_w*tr:.1f}" height="18" fill="#D55E00"/>',
                  f'<text class="small" x="{x0+chart_w*vr+4:.1f}" y="{y+17}">CV {vr:.1%}</text>',
                  f'<text class="small" x="{x0+chart_w*tr+4:.1f}" y="{y+40}">test {tr:.1%}</text>']
    save_svg(lines,path)


def histogram(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    counts=[0]*(len(edges)-1)
    for value in values:
        index=min(len(counts)-1,max(0,int((value-edges[0])/(edges[-1]-edges[0])*len(counts)))) if edges[-1]>edges[0] else 0
        counts[index]+=1
    return counts


def plot_margins(actual: Sequence[str], predicted: Sequence[str], margins: Sequence[float], path: Path) -> None:
    transformed=[math.log1p(max(0.0, margin)) for margin in margins]
    correct=[margin for truth,guess,margin in zip(actual,predicted,transformed) if truth==guess]; wrong=[margin for truth,guess,margin in zip(actual,predicted,transformed) if truth!=guess]
    cap=sorted(transformed)[max(0,int(.99*len(transformed))-1)]; edges=[cap*index/20 for index in range(21)]
    series=(("Correct",correct,"#0072B2"),("Incorrect",wrong,"#D55E00")); width,height,x0,y0,chart_w,chart_h=950,500,85,70,790,330
    lines=svg_start(width,height,"GLR margin distributions","Histograms show fractions of correct and incorrect held-out predictions after log1p transformation; values above the 99th percentile are placed in the last bin.")
    normalized=[]
    for name,values,color in series:
        counts=histogram(values,edges)
        normalized.append((name, [count / max(len(values), 1) for count in counts], color, len(values)))
    peak=max((value for _,counts,_,_ in normalized for value in counts), default=1.0) or 1.0
    for series_index,(name,counts,color,sample_count) in enumerate(normalized):
        points=[]
        for index,count in enumerate(counts):
            x=x0+chart_w*(index+.5)/len(counts); y=y0+chart_h*(1-count/peak); points.append((x,y))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in points)}"/>')
        lines.append(f'<text class="label" x="{x0+chart_w-130}" y="{90+22*series_index}" fill="{color}">{name} (n={sample_count})</text>')
    lines += [f'<rect x="{x0}" y="{y0}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#777"/>', f'<text class="label" x="{x0+chart_w/2}" y="{height-20}" text-anchor="middle">log(1 + twice-log GLR margin), clipped at p99={cap:.2f}</text>']
    save_svg(lines,path)


def plot_coverage(actual: Sequence[str], predicted: Sequence[str], margins: Sequence[float], path: Path) -> None:
    ranked=sorted(zip(margins,actual,predicted),reverse=True); points=[]
    for percent in range(10,101,10):
        count=math.ceil(len(ranked)*percent/100); accuracy=sum(truth==guess for _,truth,guess in ranked[:count])/count; points.append((percent/100,accuracy))
    width,height,x0,y0,chart_w,chart_h=850,470,90,65,690,320; low=min(score for _,score in points)-.03
    lines=svg_start(width,height,"Accuracy–coverage trade-off","Accuracy among the highest-margin held-out predictions as progressively more windows are retained.")
    coordinates=[]
    for coverage,accuracy in points:
        x=x0+chart_w*(coverage-.1)/.9; y=y0+chart_h*(1-accuracy)/(1-low); coordinates.append((x,y))
        lines += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#009E73"/>', f'<text class="small" x="{x:.1f}" y="{y-8:.1f}" text-anchor="middle">{accuracy:.2f}</text>']
    lines += [f'<polyline fill="none" stroke="#009E73" stroke-width="2" points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in coordinates)}"/>', f'<rect x="{x0}" y="{y0}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#777"/>', f'<text class="label" x="{x0+chart_w/2}" y="{height-20}" text-anchor="middle">Coverage: fraction of windows retained by descending GLR margin</text>']
    save_svg(lines,path)


def plot_subject_accuracy(dataset: Dataset, predicted: Sequence[str], path: Path) -> None:
    subjects=sorted(set(dataset.test_groups)); scores=[]
    for subject in subjects:
        indices=[index for index,group in enumerate(dataset.test_groups) if group==subject]
        score=sum(dataset.test_labels[index]==predicted[index] for index in indices)/len(indices)
        scores.append((subject,score,len(indices)))
    width,height,x0,y0,chart_w,chart_h=1000,500,85,65,840,330
    lines=svg_start(width,height,"Held-out accuracy by subject","Each bar is one untouched test subject; labels show accuracy and window count.")
    bar_w=chart_w/len(scores)*.72
    for index,(subject,score,count) in enumerate(scores):
        x=x0+chart_w*(index+.5)/len(scores)-bar_w/2; y=y0+chart_h*(1-score)
        lines += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{chart_h*score:.1f}" fill="#0072B2"/>',
                  f'<text class="small" x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle">{score:.0%}</text>',
                  f'<text class="small" x="{x+bar_w/2:.1f}" y="{y0+chart_h+16:.1f}" text-anchor="middle">{subject}</text>',
                  f'<text class="small" x="{x+bar_w/2:.1f}" y="{y0+chart_h+29:.1f}" text-anchor="middle">n={count}</text>']
    lines += [f'<rect x="{x0}" y="{y0}" width="{chart_w}" height="{chart_h}" fill="none" stroke="#777"/>',
              f'<text class="label" x="{x0+chart_w/2}" y="{height-18}" text-anchor="middle">Held-out subject identifier</text>']
    save_svg(lines,path)


def matrix_correlation(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    columns=list(zip(*vectors)); means=[fmean(column) for column in columns]
    result=[]
    for row,left in enumerate(columns):
        line=[]
        for col,right in enumerate(columns):
            numerator=sum((a-means[row])*(b-means[col]) for a,b in zip(left,right)); denominator=math.sqrt(sum((a-means[row])**2 for a in left)*sum((b-means[col])**2 for b in right)); line.append(numerator/denominator if denominator else 0.0)
        result.append(line)
    return result


def plot_matrix(matrix: Sequence[Sequence[float]], row_labels: Sequence[str], col_labels: Sequence[str],
                title: str, description: str, path: Path, diverging: bool=False) -> None:
    if len(matrix) != len(row_labels) or any(len(row) != len(col_labels) for row in matrix):
        raise ValueError("Matrix dimensions must match row and column labels")
    cell=42 if len(col_labels)>10 else 58; x0,y0=235,145
    width=x0+cell*len(col_labels)+80; height=y0+cell*len(row_labels)+130
    lines=svg_start(width,height,title,description)
    low=min(min(row) for row in matrix); high=max(max(row) for row in matrix)
    limit=max(abs(low),abs(high))
    for row,label in enumerate(row_labels):
        lines.append(f'<text class="small" x="{x0-8}" y="{y0+row*cell+cell/2+4}" text-anchor="end">{html.escape(label)}</text>')
        for col,value in enumerate(matrix[row]):
            color=diverging_color(value,limit) if diverging else heat_color(value,low,high)
            lines.append(f'<rect x="{x0+col*cell}" y="{y0+row*cell}" width="{cell}" height="{cell}" fill="{color}" stroke="white"/>')
            if max(len(row_labels),len(col_labels))<=10: lines.append(f'<text class="small" x="{x0+col*cell+cell/2}" y="{y0+row*cell+cell/2+4}" text-anchor="middle">{value:.2f}</text>')
    for col,label in enumerate(col_labels): lines.append(f'<text class="small" x="{x0+col*cell+cell/2}" y="{y0-8}" text-anchor="end" transform="rotate(-45 {x0+col*cell+cell/2} {y0-8})">{html.escape(label)}</text>')
    lines.append(f'<text class="small" x="{x0}" y="{height-25}">Range shown: {low:.2f} to {high:.2f}.</text>'); save_svg(lines,path)


def plots(dataset: Dataset, ranked: list[dict[str, object]], selected: dict[str, object], final: dict[str, object], directory: Path) -> None:
    config=selected["config"]
    plot_ranking(ranked,directory/"01_configuration_ranking.svg")
    plot_family_heatmap(ranked,directory/"02_model_family_heatmap.svg")
    plot_shrinkage(ranked,config,directory/"03_shrinkage_sensitivity.svg")
    plot_confusion(final["metrics"],directory/"04_test_confusion_matrix.svg")
    plot_recalls(selected["metrics"],final["metrics"],directory/"05_per_class_recall.svg")
    plot_margins(dataset.test_labels,final["predictions"],final["margins"],directory/"06_glr_margin_distribution.svg")
    plot_coverage(dataset.test_labels,final["predictions"],final["margins"],directory/"07_accuracy_coverage.svg")
    standardized=fit_standardizer(dataset.train_features)
    z=[standardized.transform(vector) for vector in dataset.train_features]
    profiles=[[fmean(vector[index] for vector,label in zip(z,dataset.train_labels) if label==activity) for index in range(len(FEATURE_NAMES))] for activity in dataset.class_order]
    plot_matrix(profiles,dataset.class_order,FEATURE_NAMES,"Class feature profiles","Training-only class mean z-scores for all fifteen features.",directory/"08_class_feature_profiles.svg",True)
    plot_matrix(matrix_correlation(z),FEATURE_NAMES,FEATURE_NAMES,"Training feature correlations","Pearson correlations across training windows; repeated windows within subjects are not independent.",directory/"09_feature_correlations.svg",True)
    plot_subject_accuracy(dataset,final["predictions"],directory/"10_test_subject_accuracy.svg")


def run(dataset: Dataset) -> None:
    directory=OUTPUT/("uci_har" if dataset.name=="UCI HAR" else "wisdm"); directory.mkdir(parents=True,exist_ok=True)
    ranked,selected,assignment,fold_sizes=search(dataset); final=fit_final(dataset,selected["config"])
    write_tables(dataset,ranked,selected,final,assignment,fold_sizes,directory); plots(dataset,ranked,selected,final,directory)
    config=selected["config"]
    print(f"{dataset.name}: selected {config.key}")
    print(f"  validation balanced accuracy={selected['metrics']['balanced_accuracy']:.2%}; test balanced accuracy={final['metrics']['balanced_accuracy']:.2%}; test accuracy={final['metrics']['accuracy']:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "uci", "wisdm"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.dataset in ("all", "uci"):
        run(read_uci())
    if args.dataset in ("all", "wisdm"):
        run(read_wisdm())


if __name__ == "__main__": main()
