#!/usr/bin/env python3
"""Subject-independent continuous GLRT evaluation on UCI 341/HAPT."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from analysis.fundamental_signal_audit import panel_frame, polyline, save_svg, svg_header
except ModuleNotFoundError:
    from fundamental_signal_audit import panel_frame, polyline, save_svg, svg_header
from glrt.continuous_transition_glrt import ScanResult, scan_glrt, threshold_crossings


ROOT = Path(__file__).resolve().parent.parent
DATA_DEFAULT = ROOT / "Datasets/uci341_postural_transitions"
OUT_DEFAULT = ROOT / "analysis/continuous_glrt"
SAMPLE_RATE = 50
TRAIN_SUBJECTS = frozenset(set(range(1, 31)) - {4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29, 30})
VALIDATION_SUBJECTS = frozenset({4, 9, 14, 19, 24, 29})
TEST_SUBJECTS = frozenset({5, 10, 15, 20, 25, 30})
QUANTILES = (90, 95, 97, 98, 98.5, 99, 99.2, 99.3, 99.4, 99.5, 99.6,
             99.7, 99.8, 99.9, 99.95, 99.97, 99.99)
REFRACTORY_SECONDS = (1, 2, 3, 5)


@dataclass(frozen=True)
class Config:
    name: str
    sensor: str
    horizon_samples: int
    minimum_segment: int
    hop: int
    candidate_step: int
    variance_model: str


@dataclass
class Record:
    experiment: int
    user: int
    result: ScanResult
    activities: np.ndarray


CONFIGS = (
    Config("mean_acc_2s", "acc", 100, 25, 10, 5, "common"),
    Config("mean_gyro_2s", "gyro", 100, 25, 10, 5, "common"),
    Config("mean_both_2s", "both", 100, 25, 10, 5, "common"),
    Config("mean_variance_both_4s", "both", 200, 25, 10, 5, "separate"),
)


def load_labels(path: Path) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    grouped: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for experiment, user, activity, start, end in np.loadtxt(path, dtype=int):
        grouped[(int(experiment), int(user))].append((int(activity), int(start) - 1, int(end)))
    return grouped


def load_streams(raw: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    streams = {}
    for path in sorted(raw.glob("acc_exp*_user*.txt")):
        match = re.search(r"exp(\d+)_user(\d+)", path.name)
        if not match:
            continue
        key = tuple(map(int, match.groups()))
        acc = np.loadtxt(path)
        gyro = np.loadtxt(raw / f"gyro_exp{key[0]:02d}_user{key[1]:02d}.txt")
        if acc.shape != gyro.shape:
            raise ValueError(f"Sensor length mismatch for {key}")
        streams[key] = (acc, gyro)
    return streams


def activity_at_times(intervals: list[tuple[int, int, int]], times: np.ndarray) -> np.ndarray:
    activities = np.full(len(times), -1, dtype=int)
    for activity, start, end in intervals:
        activities[(times >= start) & (times < end)] = activity
    return activities


def compute_records(config: Config, streams: dict, labels: dict) -> dict[tuple[int, int], Record]:
    records = {}
    for (experiment, user), (acc, gyro) in streams.items():
        signal = {"acc": acc, "gyro": gyro, "both": np.hstack((acc, gyro))}[config.sensor]
        result = scan_glrt(
            signal, horizon=config.horizon_samples, minimum_segment=config.minimum_segment,
            hop=config.hop, candidate_step=config.candidate_step,
            variance_model=config.variance_model,
        )
        records[(experiment, user)] = Record(
            experiment, user, result,
            activity_at_times(labels[(experiment, user)], result.trigger_samples),
        )
    return records


def match_record(record: Record, intervals: list[tuple[int, int, int]], threshold: float,
                 refractory: int, tolerance: int = 50) -> dict[str, object]:
    events = [(start, end, activity) for activity, start, end in intervals if activity >= 7]
    stable_samples = sum(end - start for activity, start, end in intervals if 1 <= activity <= 6)
    predictions = threshold_crossings(record.result, record.activities, threshold, refractory)
    used: set[int] = set()
    delays: list[float] = []
    matched_predictions: set[int] = set()
    matched_activities: list[int] = []
    for prediction_index, (trigger, _change, _score) in enumerate(predictions):
        candidates = [
            (abs(trigger - (start + end) // 2), event_index, start)
            for event_index, (start, end, _activity) in enumerate(events)
            if event_index not in used and start - tolerance <= trigger <= end + tolerance
        ]
        if candidates:
            _, event_index, start = min(candidates)
            used.add(event_index)
            matched_predictions.add(prediction_index)
            delays.append((trigger - start) / SAMPLE_RATE)
            matched_activities.append(events[event_index][2])
    return {
        "tp": len(used), "fp": len(predictions) - len(used), "fn": len(events) - len(used),
        "stable_samples": stable_samples, "delays": delays, "predictions": predictions,
        "matched_prediction_indices": matched_predictions,
        "event_total_by_activity": dict(Counter(activity for _start, _end, activity in events)),
        "event_matched_by_activity": dict(Counter(matched_activities)),
    }


def aggregate(records: dict, labels: dict, subjects: frozenset[int], threshold: float,
              refractory: int) -> tuple[dict[str, float | int | None], dict[int, dict[str, object]]]:
    per_subject: dict[int, dict[str, object]] = {
        subject: {"tp": 0, "fp": 0, "fn": 0, "stable_samples": 0, "delays": [],
                  "event_total_by_activity": Counter(), "event_matched_by_activity": Counter()}
        for subject in subjects
    }
    for key, record in records.items():
        if record.user not in subjects:
            continue
        result = match_record(record, labels[key], threshold, refractory)
        target = per_subject[record.user]
        for field in ("tp", "fp", "fn", "stable_samples"):
            target[field] = int(target[field]) + int(result[field])
        target["delays"].extend(result["delays"])
        target["event_total_by_activity"].update(result["event_total_by_activity"])
        target["event_matched_by_activity"].update(result["event_matched_by_activity"])
    metrics = metrics_from_components(per_subject.values())
    total_by_activity = Counter()
    matched_by_activity = Counter()
    for values in per_subject.values():
        total_by_activity.update(values["event_total_by_activity"])
        matched_by_activity.update(values["event_matched_by_activity"])
    metrics["recall_by_transition_id"] = {
        str(activity): matched_by_activity[activity] / total
        for activity, total in sorted(total_by_activity.items())
    }
    return metrics, per_subject


def metrics_from_components(components) -> dict[str, float | int | None]:
    parts = list(components)
    tp = sum(int(part["tp"]) for part in parts)
    fp = sum(int(part["fp"]) for part in parts)
    fn = sum(int(part["fn"]) for part in parts)
    stable = sum(int(part["stable_samples"]) for part in parts)
    delays = [float(value) for part in parts for value in part["delays"]]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "false_alarms_per_minute": fp / (stable / SAMPLE_RATE / 60) if stable else 0.0,
        "median_delay_seconds": median(delays) if delays else None,
    }


def bootstrap_subject_ci(per_subject: dict[int, dict[str, object]], iterations: int = 10000,
                         seed: int = 341) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    subjects = sorted(per_subject)
    draws = {name: [] for name in ("precision", "recall", "f1", "false_alarms_per_minute", "median_delay_seconds")}
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        metrics = metrics_from_components(per_subject[int(subject)] for subject in sampled)
        for name in draws:
            if metrics[name] is not None:
                draws[name].append(float(metrics[name]))
    return {name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            for name, values in draws.items()}


def stable_scores(records: dict, subjects: frozenset[int]) -> np.ndarray:
    return np.concatenate([
        record.result.scores[(record.activities >= 1) & (record.activities <= 6)]
        for record in records.values() if record.user in subjects
    ])


def select_operating_point(records: dict, labels: dict, budget: float | None,
                           objective: str) -> dict[str, object]:
    validation_scores = stable_scores(records, VALIDATION_SUBJECTS)
    candidates = []
    for quantile in QUANTILES:
        threshold = float(np.percentile(validation_scores, quantile))
        for seconds in REFRACTORY_SECONDS:
            refractory = seconds * SAMPLE_RATE
            metrics, _ = aggregate(records, labels, VALIDATION_SUBJECTS, threshold, refractory)
            if budget is not None and float(metrics["false_alarms_per_minute"]) > budget:
                continue
            if objective == "f1":
                rank = (float(metrics["f1"]), float(metrics["recall"]),
                        -float(metrics["false_alarms_per_minute"]))
            else:
                rank = (float(metrics["recall"]), float(metrics["precision"]),
                        -float(metrics["false_alarms_per_minute"]))
            candidates.append((rank, quantile, threshold, refractory, metrics))
    if not candidates:
        raise ValueError(f"No validation operating point meets budget {budget}")
    _, quantile, threshold, refractory, metrics = max(candidates, key=lambda row: row[0])
    return {"validation_quantile": quantile, "threshold": threshold,
            "refractory_samples": refractory, "validation_metrics": metrics}


def subject_rows(config_name: str, profile: str, per_subject: dict[int, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for subject, values in sorted(per_subject.items()):
        metrics = metrics_from_components([values])
        rows.append({"config": config_name, "profile": profile, "subject": subject, **metrics})
    return rows


def operating_curve(records: dict, labels: dict, subjects: frozenset[int], refractory: int) -> list[dict[str, float]]:
    reference = stable_scores(records, VALIDATION_SUBJECTS)
    points = []
    for quantile in QUANTILES:
        threshold = float(np.percentile(reference, quantile))
        metrics, _ = aggregate(records, labels, subjects, threshold, refractory)
        points.append({"quantile": quantile, "threshold": threshold,
                       "recall": float(metrics["recall"]),
                       "false_alarms_per_minute": float(metrics["false_alarms_per_minute"])})
    return points


def plot_operating_curves(curves: dict[str, list[dict[str, float]]], output: Path) -> None:
    width, height = 1050, 650
    lines = svg_header(width, height, "Continuous GLRT operating curves on held-out test subjects",
                       "Thresholds are validation stable-score quantiles; lower false alarms are better")
    x0, y0, panel_w, panel_h = 105, 95, 820, 410
    max_x = max(point["false_alarms_per_minute"] for curve in curves.values() for point in curve)
    max_x = min(max(max_x, 1.0), 3.0)
    panel_frame(lines, x0, y0, panel_w, panel_h, 0, 1, "event recall")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for (name, curve), color in zip(curves.items(), colors):
        ordered = sorted(curve, key=lambda point: point["false_alarms_per_minute"])
        points = []
        for point in ordered:
            x = x0 + panel_w * min(point["false_alarms_per_minute"], max_x) / max_x
            y = y0 + panel_h * (1 - point["recall"])
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{" ".join(points)}"/>')
    for tick in (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        if tick <= max_x:
            x = x0 + panel_w * tick / max_x
            lines += [f'<line x1="{x}" y1="{y0+panel_h}" x2="{x}" y2="{y0+panel_h+6}" stroke="#555"/>',
                      f'<text class="tiny" x="{x}" y="{y0+panel_h+22}" text-anchor="middle">{tick:g}</text>']
    lines.append(f'<text class="small" x="{x0+panel_w/2}" y="{y0+panel_h+52}" text-anchor="middle">false alarms per stable minute</text>')
    for index, ((name, _curve), color) in enumerate(zip(curves.items(), colors)):
        x = 130 + (index % 2) * 440
        y = 575 + (index // 2) * 24
        lines += [f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                  f'<text class="small" x="{x+36}" y="{y+4}">{html.escape(name)}</text>']
    save_svg(lines, output)


def plot_example_trace(records_by_config: dict[str, dict], labels: dict,
                       selected: dict[str, str], all_results: dict, output: Path) -> None:
    example_key = next(
        key for key in sorted(labels)
        if key[1] in TEST_SUBJECTS and any(activity >= 7 for activity, _start, _end in labels[key])
    )
    transitions = [(start, end) for activity, start, end in labels[example_key] if activity >= 7]
    section_start = max(0, transitions[0][0] - 5 * SAMPLE_RATE)
    section_end = min(transitions[min(2, len(transitions) - 1)][1] + 5 * SAMPLE_RATE,
                      max(end for _activity, _start, end in labels[example_key]))
    width, height = 1250, 650
    lines = svg_header(width, height, "Causal GLRT scores on one held-out continuous stream",
                       f"experiment {example_key[0]}, subject {example_key[1]}; yellow bands are transition intervals")
    colors = ("#0072B2", "#CC79A7")
    for panel, (profile, color) in enumerate(zip(("balanced", "low_false_alarm"), colors)):
        config_name = selected[profile]
        record = records_by_config[config_name][example_key]
        settings = all_results[config_name]["profiles"][profile]
        threshold = float(settings["threshold"])
        refractory = int(settings["refractory_samples"])
        keep = ((record.result.trigger_samples >= section_start)
                & (record.result.trigger_samples < section_end))
        times = record.result.trigger_samples[keep]
        ratios = record.result.scores[keep] / threshold
        x0, y0, panel_w, panel_h = 100, 100 + panel * 245, 1050, 165
        high = max(2.0, float(np.percentile(ratios, 99)) * 1.08)
        panel_frame(lines, x0, y0, panel_w, panel_h, 0, high, "score / threshold")
        for start, end in transitions:
            if end < section_start or start > section_end:
                continue
            shade_x = x0 + panel_w * (max(start, section_start) - section_start) / (section_end - section_start)
            shade_end = x0 + panel_w * (min(end, section_end) - section_start) / (section_end - section_start)
            lines.append(f'<rect x="{shade_x}" y="{y0}" width="{max(shade_end-shade_x, 1)}" height="{panel_h}" fill="#F0E442" opacity="0.25"/>')
        clipped = np.minimum(ratios, high).tolist()
        lines.append(polyline(clipped, x0, y0, panel_w, panel_h, 0, high, color, 1.5))
        threshold_y = y0 + panel_h * (high - 1) / high
        lines.append(f'<line x1="{x0}" y1="{threshold_y}" x2="{x0+panel_w}" y2="{threshold_y}" stroke="#111" stroke-dasharray="5 4"/>')
        alarms = threshold_crossings(record.result, record.activities, threshold, refractory)
        for trigger, _change, _score in alarms:
            if section_start <= trigger < section_end:
                x = x0 + panel_w * (trigger - section_start) / (section_end - section_start)
                lines.append(f'<circle cx="{x}" cy="{threshold_y}" r="5" fill="#D55E00" stroke="white"/>')
        lines += [
            f'<text class="label" x="{x0}" y="{y0-12}">{profile.replace("_", " ")}: {config_name}</text>',
            f'<text class="tiny" x="{x0}" y="{y0+panel_h+18}">0 s</text>',
            f'<text class="tiny" x="{x0+panel_w}" y="{y0+panel_h+18}" text-anchor="end">{(section_end-section_start)/SAMPLE_RATE:.1f} s</text>',
        ]
    lines += [
        f'<circle cx="{430}" cy="{590}" r="5" fill="#D55E00"/><text class="small" x="445" y="594">emitted rising-threshold alarm</text>',
        f'<text class="small" x="{width/2}" y="625" text-anchor="middle">Scores use only the trailing raw stream; labels are overlaid after detection for evaluation.</text>',
    ]
    save_svg(lines, output)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.dataset_root / "RawData"
    labels = load_labels(raw / "labels.txt")
    streams = load_streams(raw)

    all_results = {}
    records_by_config = {}
    subject_metrics_rows = []
    curves = {}
    for config in CONFIGS:
        records = compute_records(config, streams, labels)
        records_by_config[config.name] = records
        profiles = {
            "balanced": select_operating_point(records, labels, budget=1.0, objective="f1"),
            "low_false_alarm": select_operating_point(records, labels, budget=0.1, objective="recall"),
        }
        config_results = {"config": config.__dict__, "profiles": {}}
        for profile_name, selection in profiles.items():
            test_metrics, per_subject = aggregate(
                records, labels, TEST_SUBJECTS, float(selection["threshold"]),
                int(selection["refractory_samples"]),
            )
            config_results["profiles"][profile_name] = {
                **selection, "test_metrics": test_metrics,
                "test_subject_bootstrap_95ci": bootstrap_subject_ci(per_subject),
            }
            subject_metrics_rows.extend(subject_rows(config.name, profile_name, per_subject))
        all_results[config.name] = config_results
        curves[config.name] = operating_curve(
            records, labels, TEST_SUBJECTS,
            int(profiles["balanced"]["refractory_samples"]),
        )

    selected = {
        "balanced": max(
            (result for result in all_results.values()),
            key=lambda result: float(result["profiles"]["balanced"]["validation_metrics"]["f1"]),
        )["config"]["name"],
        "low_false_alarm": max(
            (result for result in all_results.values()),
            key=lambda result: (
                float(result["profiles"]["low_false_alarm"]["validation_metrics"]["recall"]),
                float(result["profiles"]["low_false_alarm"]["validation_metrics"]["precision"]),
            ),
        )["config"]["name"],
    }
    output = {
        "sampling_hz": SAMPLE_RATE,
        "subject_split": {"train": sorted(TRAIN_SUBJECTS),
                          "validation": sorted(VALIDATION_SUBJECTS), "test": sorted(TEST_SUBJECTS)},
        "event_match": "one-to-one; alarm trigger within transition interval expanded by 1 s",
        "false_alarm_denominator": "minutes annotated as basic activities 1-6",
        "selected_on_validation": selected,
        "results": all_results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "test_subject_metrics.csv", subject_metrics_rows)
    plot_operating_curves(curves, args.output_dir / "01_test_operating_curves.svg")
    plot_example_trace(records_by_config, labels, selected, all_results,
                       args.output_dir / "02_held_out_score_trace.svg")
    print(json.dumps({
        profile: {
            "config": config_name,
            "test": all_results[config_name]["profiles"][profile]["test_metrics"],
            "ci": all_results[config_name]["profiles"][profile]["test_subject_bootstrap_95ci"],
        } for profile, config_name in selected.items()
    }, indent=2))


if __name__ == "__main__":
    main()
