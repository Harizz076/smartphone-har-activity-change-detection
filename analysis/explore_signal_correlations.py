#!/usr/bin/env python3
"""Dependency-free exploratory signal/correlation plots for this HAR project.

Writes SVG figures and a JSON summary; it never changes the raw data or GLRT.
All findings are exploratory, calculated on training-side data only.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent.parent
UCI_LABELS = {1: "WALKING", 2: "WALKING UPSTAIRS", 3: "WALKING DOWNSTAIRS", 4: "SITTING", 5: "STANDING", 6: "LAYING"}
UCI_STILL = frozenset({4, 5, 6})
WISDM_STILL = frozenset({"Sitting", "Standing"})
TEST_USERS = frozenset(range(3, 37, 3))
FEATURES = ("log Var(x)", "log Var(y)", "log Var(z)", "log jerk", "log mean Var")
COLORS = ("#0072B2", "#D55E00", "#009E73")


def variance(values: list[float]) -> float:
    average = fmean(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)


def feature_vector(samples: list[tuple[float, float, float]]) -> list[float]:
    axis = [variance([row[column] for row in samples]) for column in range(3)]
    jerk = fmean(sum((current[column] - previous[column]) ** 2 for column in range(3))
                 for previous, current in zip(samples, samples[1:]))
    return [math.log(max(value, 1e-12)) for value in (*axis, jerk, fmean(axis))]


def correlation(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right))
    return numerator / denominator if denominator else 0.0


def scales(series: list[list[float]], x0: float, y0: float, width: float, height: float):
    flat = [value for values in series for value in values]
    low, high = min(flat), max(flat)
    padding = max((high - low) * .08, .01)
    low, high = low - padding, high + padding
    def point(index: int, value: float, count: int) -> tuple[float, float]:
        return x0 + width * index / max(count - 1, 1), y0 + height * (high - value) / (high - low)
    return point, low, high


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222} .small{font-size:11px}.label{font-size:12px}.title{font-size:18px;font-weight:bold}</style>',
            f'<text class="title" x="{width/2}" y="28" text-anchor="middle">{html.escape(title)}</text>']


def save_svg(lines: list[str], output: Path) -> None:
    output.write_text("\n".join(lines + ["</svg>"]), encoding="utf-8")


def plot_uci_windows(windows: list[list[tuple[float, float, float]]], labels: list[int], vectors: list[list[float]], output: Path) -> None:
    representatives = {}
    for label in UCI_LABELS:
        choices = sorted((vectors[index][-1], index) for index, actual in enumerate(labels) if actual == label)
        representatives[label] = choices[len(choices) // 2][1]
    width, height = 1320, 720
    lines = svg_header(width, height, "Representative UCI HAR total-acceleration windows")
    for position, label in enumerate(sorted(UCI_LABELS)):
        row, col = divmod(position, 3)
        x0, y0, panel_w, panel_h = 65 + col * 420, 70 + row * 310, 330, 205
        window = windows[representatives[label]]
        series = [[sample[axis] for sample in window] for axis in range(3)]
        point, low, high = scales(series, x0, y0, panel_w, panel_h)
        lines += [f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#999"/>',
                  f'<text class="label" x="{x0}" y="{y0-10}">{UCI_LABELS[label].title()}</text>',
                  f'<text class="small" x="{x0-7}" y="{y0+10}" text-anchor="end">{low:.2f}</text>',
                  f'<text class="small" x="{x0-7}" y="{y0+panel_h}" text-anchor="end">{high:.2f}</text>']
        for values, color, name in zip(series, COLORS, ("x", "y", "z")):
            path = " ".join(f"{px:.1f},{py:.1f}" for index, value in enumerate(values) for px, py in [point(index, value, len(values))])
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.1" points="{path}"/>')
            lines.append(f'<text class="small" x="{x0 + panel_w - 14 * (2 - COLORS.index(color))}" y="{y0+15}" fill="{color}">{name}</text>')
        lines.append(f'<text class="small" x="{x0+panel_w/2}" y="{y0+panel_h+18}" text-anchor="middle">2.56-second supplied window</text>')
    save_svg(lines, output)


def correlation_matrix(vectors: list[list[float]], keep: list[bool]) -> list[list[float]]:
    columns = [[vector[column] for vector, include in zip(vectors, keep) if include] for column in range(len(FEATURES))]
    return [[correlation(columns[row], columns[col]) for col in range(len(FEATURES))] for row in range(len(FEATURES))]


def heat_color(value: float) -> str:
    intensity = int(255 - 130 * abs(value))
    return f"rgb({255 if value >= 0 else intensity},{intensity},{intensity if value >= 0 else 255})"


def plot_correlations(still: list[list[float]], moving: list[list[float]], output: Path) -> dict[str, dict[str, object]]:
    width, height, size = 1320, 690, 72
    lines = svg_header(width, height, "UCI training-window feature correlations")
    results = {}
    for panel, (name, matrix) in enumerate((("Still", still), ("Moving", moving))):
        x0, y0 = 130 + panel * 650, 120
        lines.append(f'<text class="label" x="{x0+size*2.5}" y="{y0-28}" text-anchor="middle">{name}</text>')
        for row in range(len(FEATURES)):
            lines.append(f'<text class="small" x="{x0-8}" y="{y0+row*size+size/2+4}" text-anchor="end">{html.escape(FEATURES[row])}</text>')
            for col in range(len(FEATURES)):
                value = matrix[row][col]
                lines += [f'<rect x="{x0+col*size}" y="{y0+row*size}" width="{size}" height="{size}" fill="{heat_color(value)}" stroke="white"/>',
                          f'<text class="small" x="{x0+col*size+size/2}" y="{y0+row*size+size/2+4}" text-anchor="middle">{value:.2f}</text>']
        for col, feature in enumerate(FEATURES):
            lines.append(f'<text class="small" x="{x0+col*size+size/2}" y="{y0+size*5+16}" text-anchor="end" transform="rotate(-35 {x0+col*size+size/2} {y0+size*5+16})">{html.escape(feature)}</text>')
        # Do not highlight log mean variance here: it is derived from the three
        # axis variances, so that relationship is mechanically expected.
        candidates = [(abs(matrix[row][col]), row, col, matrix[row][col])
                      for row in range(4) for col in range(row)]
        _, row, col, value = max(candidates)
        results[name.lower()] = {"feature_a": FEATURES[row], "feature_b": FEATURES[col], "r": value}
    save_svg(lines, output)
    return results


def parse_wisdm_streams(path: Path, max_gap: int = 500_000_000):
    streams, current = [], []
    last_user = last_time = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            for record in line.strip().split(";"):
                parts = record.split(",")
                if len(parts) != 6: continue
                try:
                    user, timestamp = int(parts[0]), int(parts[2]); sample = tuple(float(value) for value in parts[3:6])
                except ValueError: continue
                if not all(math.isfinite(value) for value in sample): continue
                if last_user is not None and (user != last_user or timestamp < last_time or timestamp-last_time > max_gap):
                    if current: streams.append(current)
                    current = []
                current.append((user, parts[1], timestamp, sample)); last_user, last_time = user, timestamp
    if current: streams.append(current)
    return streams


def find_transition(streams):
    for stream in streams:
        for index in range(180, len(stream)-180):
            if stream[index-1][1] != stream[index][1]: return stream, index
    raise ValueError("No transition with enough surrounding samples")


def plot_transition(streams, output: Path) -> tuple[str, str]:
    stream, boundary = find_transition(streams)
    section = stream[boundary-180:boundary+180]
    samples = [row[3] for row in section]
    before, after = section[179][1], section[180][1]
    width, height = 1320, 680
    lines = svg_header(width, height, f"WISDM raw transition: {before} → {after}")
    x0, panel_w, panel_h = 75, 1160, 215
    series = [[sample[axis] for sample in samples] for axis in range(3)]
    point, low, high = scales(series, x0, 75, panel_w, panel_h)
    lines += [f'<rect x="{x0}" y="75" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#999"/>',
              f'<text class="small" x="{x0-8}" y="85" text-anchor="end">{low:.2f}</text>', f'<text class="small" x="{x0-8}" y="290" text-anchor="end">{high:.2f}</text>']
    for values, color in zip(series, COLORS):
        path = " ".join(f"{px:.1f},{py:.1f}" for index, value in enumerate(values) for px, py in [point(index, value, len(values))])
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width=".9" points="{path}"/>')
    feature = [feature_vector(samples[start:start+100])[-1] for start in range(0, len(samples)-99, 20)]
    fx0, fy0 = x0, 370
    fpoint, flow, fhigh = scales([feature], fx0, fy0, panel_w, 180)
    lines += [f'<rect x="{fx0}" y="{fy0}" width="{panel_w}" height="180" fill="#fafafa" stroke="#999"/>',
              f'<text class="label" x="{fx0}" y="{fy0-10}">Rolling log mean variance (5 s window, 1 s hop)</text>']
    path = " ".join(f"{px:.1f},{py:.1f}" for index, value in enumerate(feature) for px, py in [fpoint(index, value, len(feature))])
    lines.append(f'<polyline fill="none" stroke="#CC79A7" stroke-width="2" points="{path}"/>')
    transition_x = x0 + panel_w * 180 / 359
    for y, h in ((75, panel_h), (fy0, 180)):
        lines.append(f'<line x1="{transition_x:.1f}" y1="{y}" x2="{transition_x:.1f}" y2="{y+h}" stroke="#111" stroke-dasharray="5 4"/>')
    lines.append(f'<text class="small" x="{transition_x+5:.1f}" y="570">provided label boundary</text>')
    lines.append(f'<text class="small" x="{x0+panel_w/2}" y="620" text-anchor="middle">18 seconds around transition</text>')
    save_svg(lines, output)
    return before, after


def still_acf(streams: list[list[tuple[int, str, int, tuple[float, float, float]]]], max_lag: int = 8) -> list[float]:
    weighted = [0.0] * (max_lag+1); weights = [0] * (max_lag+1)
    for stream in streams:
        if stream[0][0] in TEST_USERS: continue
        begin = 0
        while begin < len(stream):
            end = begin + 1
            while end < len(stream) and stream[end][1] == stream[begin][1]: end += 1
            run = stream[begin:end]
            if run[0][1] in WISDM_STILL:
                values = [feature_vector([row[3] for row in run[start:start+100]])[-1] for start in range(0, len(run)-99, 20)]
                if len(values) > max_lag:
                    average = fmean(values); centered = [value-average for value in values]; denom = sum(value*value for value in centered)
                    if denom:
                        for lag in range(max_lag+1):
                            pairs = len(values)-lag
                            value = sum(centered[index]*centered[index+lag] for index in range(pairs)) / denom
                            weighted[lag] += value*pairs; weights[lag] += pairs
            begin = end
    return [weighted[lag]/weights[lag] if weights[lag] else 0.0 for lag in range(max_lag+1)]


def plot_acf(acf: list[float], output: Path) -> None:
    width, height, x0, y0, panel_w, panel_h = 860, 490, 90, 80, 690, 300
    lines = svg_header(width, height, "WISDM stillness rolling-feature autocorrelation")
    lines += [f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#999"/>',
              f'<line x1="{x0}" y1="{y0+panel_h/2}" x2="{x0+panel_w}" y2="{y0+panel_h/2}" stroke="#333"/>']
    for lag, value in enumerate(acf):
        bar_w = panel_w / len(acf) * .65; x = x0 + panel_w*(lag+.175)/len(acf); y = y0 + panel_h*(1-value)/2
        lines += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{abs(value)*panel_h/2:.1f}" fill="#0072B2"/>',
                  f'<text class="small" x="{x+bar_w/2:.1f}" y="{y0+panel_h+18}" text-anchor="middle">{lag}</text>',
                  f'<text class="small" x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle">{value:.2f}</text>']
    lines += [f'<text class="label" x="{x0+panel_w/2}" y="{y0+panel_h+48}" text-anchor="middle">Lag in 1-second rolling decisions</text>',
              f'<text class="small" x="{x0-10}" y="{y0+12}" text-anchor="end">1</text>', f'<text class="small" x="{x0-10}" y="{y0+panel_h/2+4}" text-anchor="end">0</text>']
    save_svg(lines, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uci-root", type=Path, default=ROOT/"Datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset")
    parser.add_argument("--wisdm-file", type=Path, default=ROOT/"Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"analysis/plots")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    axes = []
    for axis in "xyz":
        with (args.uci_root/"train"/"Inertial Signals"/f"total_acc_{axis}_train.txt").open(encoding="utf-8") as handle:
            axes.append([[float(value) for value in line.split()] for line in handle if line.strip()])
    windows = [list(zip(axes[0][index], axes[1][index], axes[2][index])) for index in range(len(axes[0]))]
    labels = [int(value) for value in (args.uci_root/"train/y_train.txt").read_text(encoding="utf-8").split()]
    vectors = [feature_vector(window) for window in windows]
    plot_uci_windows(windows, labels, vectors, args.output_dir/"uci_representative_windows.svg")
    highlights = plot_correlations(correlation_matrix(vectors, [label in UCI_STILL for label in labels]), correlation_matrix(vectors, [label not in UCI_STILL for label in labels]), args.output_dir/"uci_feature_correlations.svg")
    streams = parse_wisdm_streams(args.wisdm_file)
    transition = plot_transition(streams, args.output_dir/"wisdm_raw_transition.svg")
    acf = still_acf(streams); plot_acf(acf, args.output_dir/"wisdm_stillness_autocorrelation.svg")
    summary = {"uci_training_windows": len(labels), "uci_feature_correlation_highlights": highlights, "wisdm_streams": len(streams), "wisdm_transition_plotted": transition, "wisdm_still_acf": acf}
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
