#!/usr/bin/env python3
"""Fundamentals-first audit of the raw streams used by this project.

The script deliberately asks simple questions before fitting a detector:

* Are adjacent samples really adjacent in time?
* Is a label change continuously observed, or separated by a recording gap?
* What do x/y/z and magnitude look like around an admissible boundary?
* Do simple mean, variability, or jerk summaries visibly change?

It produces dependency-light SVG figures, CSV event records, and JSON facts.  It
does not train on, smooth over, or silently repair missing time intervals.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
WISDM_DEFAULT = ROOT / "Datasets/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
OUT_DEFAULT = ROOT / "analysis/fundamental_signal_audit"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


@dataclass(frozen=True)
class Sample:
    user: int
    label: str
    timestamp_ns: int
    x: float
    y: float
    z: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


def parse_wisdm(path: Path) -> tuple[list[Sample], int]:
    samples: list[Sample] = []
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            for record in line.strip().split(";"):
                fields = record.strip().split(",")
                if not record.strip():
                    continue
                if len(fields) != 6:
                    malformed += 1
                    continue
                try:
                    sample = Sample(
                        int(fields[0]), fields[1], int(fields[2]),
                        float(fields[3]), float(fields[4]), float(fields[5]),
                    )
                except ValueError:
                    malformed += 1
                    continue
                if not all(math.isfinite(value) for value in (sample.x, sample.y, sample.z)):
                    malformed += 1
                    continue
                samples.append(sample)
    return samples, malformed


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def svg_header(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.tiny{font-size:10px}.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:bold}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#555;stroke-width:1}</style>',
        f'<text class="title" x="{width / 2}" y="28" text-anchor="middle">{html.escape(title)}</text>',
    ]
    if subtitle:
        lines.append(f'<text class="small" x="{width / 2}" y="49" text-anchor="middle">{html.escape(subtitle)}</text>')
    return lines


def save_svg(lines: list[str], output: Path) -> None:
    output.write_text("\n".join(lines + ["</svg>"]), encoding="utf-8")


def polyline(values: list[float], x0: float, y0: float, width: float, height: float,
             low: float, high: float, color: str, stroke: float = 1.2) -> str:
    span = max(high - low, 1e-12)
    points = []
    for index, value in enumerate(values):
        x = x0 + width * index / max(len(values) - 1, 1)
        y = y0 + height * (high - value) / span
        points.append(f"{x:.2f},{y:.2f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="{stroke}" points="{" ".join(points)}"/>'


def panel_frame(lines: list[str], x0: float, y0: float, width: float, height: float,
                low: float, high: float, ylabel: str) -> None:
    lines.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#fafafa" stroke="#777"/>')
    for fraction in (0.25, 0.5, 0.75):
        y = y0 + height * fraction
        lines.append(f'<line class="grid" x1="{x0}" y1="{y}" x2="{x0 + width}" y2="{y}"/>')
    lines += [
        f'<text class="tiny" x="{x0 - 7}" y="{y0 + 4}" text-anchor="end">{high:.2f}</text>',
        f'<text class="tiny" x="{x0 - 7}" y="{y0 + height}" text-anchor="end">{low:.2f}</text>',
        f'<text class="small" x="18" y="{y0 + height / 2}" text-anchor="middle" transform="rotate(-90 18 {y0 + height / 2})">{html.escape(ylabel)}</text>',
    ]


def wisdm_boundaries(samples: list[Sample]) -> list[dict[str, object]]:
    boundaries: list[dict[str, object]] = []
    for index, (left, right) in enumerate(zip(samples, samples[1:])):
        if left.user == right.user and left.label != right.label:
            gap = (right.timestamp_ns - left.timestamp_ns) / 1e9
            boundaries.append({
                "left_index": index,
                "right_index": index + 1,
                "user": left.user,
                "before": left.label,
                "after": right.label,
                "gap_seconds": gap,
            })
    return boundaries


def has_regular_context(samples: list[Sample], boundary: dict[str, object], radius: int = 30,
                        max_step_s: float = 0.15) -> bool:
    index = int(boundary["right_index"])
    if index < radius or index + radius >= len(samples):
        return False
    section = samples[index - radius:index + radius]
    user = section[0].user
    if any(row.user != user for row in section):
        return False
    steps = [(b.timestamp_ns - a.timestamp_ns) / 1e9 for a, b in zip(section, section[1:])]
    return all(0 < step <= max_step_s for step in steps)


def plot_boundary_gap_audit(boundaries: list[dict[str, object]], output: Path) -> None:
    positive = [float(event["gap_seconds"]) for event in boundaries if float(event["gap_seconds"]) > 0]
    bins = [-1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7]
    transformed = [math.log10(value) for value in positive]
    counts = [0] * (len(bins) - 1)
    for value in transformed:
        for index in range(len(counts)):
            if bins[index] <= value < bins[index + 1]:
                counts[index] += 1
                break
    width, height = 1160, 650
    lines = svg_header(width, height, "WISDM label-boundary timing audit",
                       "A label change is usable as a continuous boundary only when the time gap is small")
    x0, y0, panel_w, panel_h = 95, 95, 980, 370
    panel_frame(lines, x0, y0, panel_w, panel_h, 0, max(counts) * 1.12, "number of label changes")
    bar_w = panel_w / len(counts) * 0.78
    for index, count in enumerate(counts):
        x = x0 + panel_w * (index + 0.11) / len(counts)
        h = panel_h * count / max(max(counts), 1)
        midpoint = (bins[index] + bins[index + 1]) / 2
        color = "#009E73" if 10 ** midpoint <= 0.5 else "#999999"
        lines.append(f'<rect x="{x:.1f}" y="{y0 + panel_h - h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        if count:
            lines.append(f'<text class="tiny" x="{x + bar_w / 2:.1f}" y="{y0 + panel_h - h - 5:.1f}" text-anchor="middle">{count}</text>')
    for seconds, label in ((0.05, "0.05 s"), (0.5, "0.5 s"), (5, "5 s"), (50, "50 s"), (500, "500 s"), (5000, "5000 s")):
        value = math.log10(seconds)
        if bins[0] <= value <= bins[-1]:
            x = x0 + panel_w * (value - bins[0]) / (bins[-1] - bins[0])
            lines += [f'<line class="axis" x1="{x}" y1="{y0 + panel_h}" x2="{x}" y2="{y0 + panel_h + 6}"/>',
                      f'<text class="tiny" x="{x}" y="{y0 + panel_h + 22}" text-anchor="middle">{label}</text>']
    continuous = sum(0 < float(event["gap_seconds"]) <= 0.5 for event in boundaries)
    nonpositive = sum(float(event["gap_seconds"]) <= 0 for event in boundaries)
    lines += [
        f'<text class="label" x="{x0}" y="535">{continuous} / {len(boundaries)} label changes have a positive gap ≤ 0.5 s</text>',
        f'<text class="small" x="{x0}" y="560">Those {continuous} events cover only upstairs↔downstairs and three users.</text>',
        f'<text class="small" x="{x0}" y="585">{nonpositive} boundaries have a non-positive timestamp step; they cannot be treated as continuous.</text>',
        f'<text class="small" x="{x0 + panel_w / 2}" y="{y0 + panel_h + 54}" text-anchor="middle">elapsed time between the final old-label sample and first new-label sample (log scale)</text>',
    ]
    save_svg(lines, output)


def select_examples(samples: list[Sample], boundaries: list[dict[str, object]], count: int = 4) -> list[dict[str, object]]:
    candidates = [event for event in boundaries
                  if 0 < float(event["gap_seconds"]) <= 0.5 and has_regular_context(samples, event)]
    selected: list[dict[str, object]] = []
    used: set[tuple[int, str, str]] = set()
    for event in candidates:
        key = (int(event["user"]), str(event["before"]), str(event["after"]))
        if key not in used:
            selected.append(event)
            used.add(key)
        if len(selected) == count:
            return selected
    for event in candidates:
        if event not in selected:
            selected.append(event)
        if len(selected) == count:
            break
    return selected


def plot_raw_examples(samples: list[Sample], events: list[dict[str, object]], output: Path,
                      radius: int = 30) -> None:
    width, height = 1320, 250 + 205 * len(events)
    lines = svg_header(width, height, "Raw WISDM acceleration around continuously sampled label boundaries",
                       "Vertical dashed line is the supplied label switch; 30 samples ≈ 1.5 s on each side")
    x0, panel_w, panel_h = 85, 1145, 145
    for panel, event in enumerate(events):
        center = int(event["right_index"])
        section = samples[center - radius:center + radius]
        values = [[getattr(row, axis) for row in section] for axis in ("x", "y", "z")]
        magnitude = [row.magnitude for row in section]
        all_values = [value for series in values + [magnitude] for value in series]
        low, high = percentile(all_values, 1), percentile(all_values, 99)
        pad = max((high - low) * 0.08, 0.1)
        low, high = low - pad, high + pad
        y0 = 90 + panel * 205
        panel_frame(lines, x0, y0, panel_w, panel_h, low, high, "acceleration (m/s²)")
        for series, color in zip(values + [magnitude], COLORS):
            clipped = [min(max(value, low), high) for value in series]
            lines.append(polyline(clipped, x0, y0, panel_w, panel_h, low, high, color, 1.0))
        boundary_x = x0 + panel_w * (radius - 0.5) / (2 * radius - 1)
        lines += [
            f'<line x1="{boundary_x}" y1="{y0}" x2="{boundary_x}" y2="{y0 + panel_h}" stroke="#111" stroke-width="1.5" stroke-dasharray="5 4"/>',
            f'<text class="label" x="{x0}" y="{y0 - 9}">user {event["user"]}: {html.escape(str(event["before"]))} → {html.escape(str(event["after"]))}; gap {float(event["gap_seconds"]):.3f} s</text>',
        ]
        for legend_index, (name, color) in enumerate(zip(("x", "y", "z", "|a|"), COLORS)):
            lx = x0 + 920 + legend_index * 55
            lines += [f'<line x1="{lx}" y1="{y0 - 13}" x2="{lx + 18}" y2="{y0 - 13}" stroke="{color}" stroke-width="3"/>',
                      f'<text class="tiny" x="{lx + 22}" y="{y0 - 9}">{name}</text>']
        lines += [f'<text class="tiny" x="{x0}" y="{y0 + panel_h + 17}">−1.5 s</text>',
                  f'<text class="tiny" x="{x0 + panel_w / 2}" y="{y0 + panel_h + 17}" text-anchor="middle">0</text>',
                  f'<text class="tiny" x="{x0 + panel_w}" y="{y0 + panel_h + 17}" text-anchor="end">+1.5 s</text>']
    lines.append(f'<text class="small" x="{width / 2}" y="{height - 25}" text-anchor="middle">Magnitude reduces orientation sensitivity, but it can hide posture-specific axis changes.</text>')
    save_svg(lines, output)


def rolling_features(section: list[Sample], window: int = 20) -> dict[str, list[float]]:
    magnitudes = [row.magnitude for row in section]
    jerk = [0.0] + [math.sqrt((b.x-a.x)**2 + (b.y-a.y)**2 + (b.z-a.z)**2)
                    for a, b in zip(section, section[1:])]
    half = window // 2
    mean_mag, sd_mag, rms_jerk = [], [], []
    for index in range(len(section)):
        left, right = max(0, index - half), min(len(section), index + half)
        m = magnitudes[left:right]
        j = jerk[left:right]
        average = fmean(m)
        mean_mag.append(average)
        sd_mag.append(math.sqrt(fmean([(value - average) ** 2 for value in m])))
        rms_jerk.append(math.sqrt(fmean([value * value for value in j])))
    return {"mean |a|": mean_mag, "SD |a|": sd_mag, "RMS jerk": rms_jerk}


def plot_simple_features(samples: list[Sample], event: dict[str, object], output: Path,
                         radius: int = 120) -> None:
    center = int(event["right_index"])
    section = samples[center - radius:center + radius]
    features = rolling_features(section)
    width, height = 1220, 735
    lines = svg_header(width, height, "Simple one-second summaries around one continuous WISDM boundary",
                       f'user {event["user"]}: {event["before"]} → {event["after"]}; features use no class model')
    x0, panel_w, panel_h = 100, 1030, 150
    for panel, (name, values) in enumerate(features.items()):
        y0 = 85 + panel * 195
        low, high = min(values), max(values)
        pad = max((high - low) * 0.08, 0.01)
        low, high = low - pad, high + pad
        panel_frame(lines, x0, y0, panel_w, panel_h, low, high, name)
        lines.append(polyline(values, x0, y0, panel_w, panel_h, low, high, COLORS[panel], 2.0))
        boundary_x = x0 + panel_w * (radius - 0.5) / (2 * radius - 1)
        lines.append(f'<line x1="{boundary_x}" y1="{y0}" x2="{boundary_x}" y2="{y0 + panel_h}" stroke="#111" stroke-width="1.5" stroke-dasharray="5 4"/>')
    lines += [
        f'<text class="tiny" x="{x0}" y="{650}">−6 s</text>',
        f'<text class="tiny" x="{x0 + panel_w / 2}" y="{650}" text-anchor="middle">0: supplied label switch</text>',
        f'<text class="tiny" x="{x0 + panel_w}" y="{650}" text-anchor="end">+6 s</text>',
        f'<text class="small" x="{width / 2}" y="695" text-anchor="middle">A useful GLRT should formalize a repeatable change visible in summaries like these—not manufacture one through labels.</text>',
    ]
    save_svg(lines, output)


def event_effects(samples: list[Sample], boundaries: list[dict[str, object]], radius: int = 30) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in boundaries:
        if not (0 < float(event["gap_seconds"]) <= 0.5 and has_regular_context(samples, event, radius)):
            continue
        center = int(event["right_index"])
        before, after = samples[center-radius:center], samples[center:center+radius]
        output = dict(event)
        for name, getter in (
            ("mean_magnitude", lambda block: fmean(row.magnitude for row in block)),
            ("sd_magnitude", lambda block: np.std([row.magnitude for row in block], ddof=0)),
            ("rms_jerk", lambda block: math.sqrt(fmean(
                (b.x-a.x)**2 + (b.y-a.y)**2 + (b.z-a.z)**2 for a, b in zip(block, block[1:])))),
        ):
            left, right = float(getter(before)), float(getter(after))
            output[f"before_{name}"] = left
            output[f"after_{name}"] = right
            output[f"delta_{name}"] = right - left
        rows.append(output)
    return rows


def write_events(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_shift_glr(left: list[float], right: list[float]) -> tuple[float, float]:
    """Exact fixed-boundary Gaussian GLRT with unknown common variance.

    H0 has one unknown mean; H1 has one unknown mean on each side.  Both
    hypotheses maximize over the Gaussian variance.  Thus
    2 log Lambda = N log(SSE0 / SSE1).
    """
    n1, n2 = len(left), len(right)
    mean1, mean2 = fmean(left), fmean(right)
    combined = left + right
    overall = fmean(combined)
    sse0 = sum((value - overall) ** 2 for value in combined)
    sse1 = sum((value - mean1) ** 2 for value in left)
    sse1 += sum((value - mean2) ** 2 for value in right)
    if sse0 <= 1e-12:
        return 0.0, 0.0
    sse1 = max(sse1, 1e-12 * max(sse0, 1.0))
    variance = sse1 / max(n1 + n2 - 2, 1)
    effect = abs(mean2 - mean1) / math.sqrt(variance)
    glr = (n1 + n2) * math.log(max(sse0 / sse1, 1.0))
    return glr, effect


def wisdm_window_sensitivity(samples: list[Sample], boundaries: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for radius in (5, 10, 20, 30, 40):
        for event in boundaries:
            if not (0 < float(event["gap_seconds"]) <= 0.5
                    and has_regular_context(samples, event, radius)):
                continue
            center = int(event["right_index"])
            before, after = samples[center-radius:center], samples[center:center+radius]
            for feature, getter in (
                ("x", lambda row: row.x), ("y", lambda row: row.y),
                ("z", lambda row: row.z), ("magnitude", lambda row: row.magnitude),
            ):
                glr, effect = mean_shift_glr([getter(row) for row in before],
                                             [getter(row) for row in after])
                rows.append({
                    "window_seconds_each_side": radius / 20,
                    "samples_each_side": radius,
                    "boundary_index": event["right_index"],
                    "user": event["user"],
                    "before": event["before"],
                    "after": event["after"],
                    "feature": feature,
                    "fixed_split_glr_2loglambda": glr,
                    "absolute_standardized_mean_gap": effect,
                })
    return rows


def plot_window_sensitivity(rows: list[dict[str, object]], output: Path) -> None:
    durations = sorted({float(row["window_seconds_each_side"]) for row in rows})
    features = ("x", "y", "z", "magnitude")
    width, height = 1120, 700
    lines = svg_header(width, height, "WISDM fixed-boundary mean-shift evidence versus window size",
                       "Median 2 log Λ; labels locate the boundary but are not inputs to the statistic")
    x0, y0, panel_w, panel_h = 100, 95, 900, 385
    medians: dict[str, list[float]] = {}
    for feature in features:
        medians[feature] = []
        for duration in durations:
            values = [float(row["fixed_split_glr_2loglambda"]) for row in rows
                      if row["feature"] == feature and float(row["window_seconds_each_side"]) == duration]
            medians[feature].append(median(values) if values else 0.0)
    high = max(value for values in medians.values() for value in values) * 1.12
    panel_frame(lines, x0, y0, panel_w, panel_h, 0, high, "median 2 log Λ")
    for feature, color in zip(features, COLORS):
        values = medians[feature]
        points = []
        for duration, value in zip(durations, values):
            x = x0 + panel_w * (duration - durations[0]) / (durations[-1] - durations[0])
            y = y0 + panel_h * (high - value) / high
            points.append(f"{x:.1f},{y:.1f}")
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{" ".join(points)}"/>')
    event_counts = []
    for duration in durations:
        event_counts.append(len({row["boundary_index"]
                                 for row in rows if float(row["window_seconds_each_side"]) == duration}))
    for duration, count in zip(durations, event_counts):
        x = x0 + panel_w * (duration - durations[0]) / (durations[-1] - durations[0])
        lines += [f'<line class="axis" x1="{x}" y1="{y0 + panel_h}" x2="{x}" y2="{y0 + panel_h + 6}"/>',
                  f'<text class="tiny" x="{x}" y="{y0 + panel_h + 22}" text-anchor="middle">{duration:g}</text>',
                  f'<text class="tiny" x="{x}" y="{y0 + panel_h + 40}" text-anchor="middle">n={count}</text>']
    for index, (feature, color) in enumerate(zip(features, COLORS)):
        lx = x0 + 145 * index
        lines += [f'<line x1="{lx}" y1="580" x2="{lx + 25}" y2="580" stroke="{color}" stroke-width="3"/>',
                  f'<text class="small" x="{lx + 32}" y="584">{feature}</text>']
    lines += [
        f'<text class="small" x="{x0 + panel_w / 2}" y="{y0 + panel_h + 70}" text-anchor="middle">seconds of data on each side of the supplied boundary</text>',
        f'<text class="small" x="{width / 2}" y="650" text-anchor="middle">Here the median evidence falls as the window grows: averaging dilutes a local shift and fewer uninterrupted events remain.</text>',
    ]
    save_svg(lines, output)


def wisdm_event_control_contrast(samples: list[Sample], boundaries: list[dict[str, object]],
                                 radius: int = 10) -> list[dict[str, object]]:
    features = ("x", "y", "z", "magnitude")

    def score_at(index: int) -> tuple[float, str]:
        before, after = samples[index-radius:index], samples[index:index+radius]
        scores = []
        for feature in features:
            getter = (lambda row: row.magnitude) if feature == "magnitude" else (lambda row, f=feature: getattr(row, f))
            score, _ = mean_shift_glr([getter(row) for row in before], [getter(row) for row in after])
            scores.append((score, feature))
        return max(scores)

    rows: list[dict[str, object]] = []
    valid_events = [event for event in boundaries
                    if 0 < float(event["gap_seconds"]) <= 0.5
                    and has_regular_context(samples, event, radius)]
    for event in valid_events:
        score, feature = score_at(int(event["right_index"]))
        rows.append({"group": "label boundary", "index": event["right_index"],
                     "user": event["user"], "label": f'{event["before"]}->{event["after"]}',
                     "max_channel_glr_2loglambda": score, "winning_channel": feature})

    users = {int(event["user"]) for event in valid_events}
    labels = {str(event["before"]) for event in valid_events} | {str(event["after"]) for event in valid_events}
    # A deterministic 1-second stride limits redundancy while retaining broad
    # coverage. Controls are restricted to the same users and stair labels.
    for index in range(radius, len(samples) - radius, 20):
        block = samples[index-radius:index+radius]
        if block[0].user not in users or block[0].label not in labels:
            continue
        if any(row.user != block[0].user or row.label != block[0].label for row in block):
            continue
        steps = [(right.timestamp_ns - left.timestamp_ns) / 1e9 for left, right in zip(block, block[1:])]
        if not all(0 < step <= 0.15 for step in steps):
            continue
        score, feature = score_at(index)
        rows.append({"group": "within-activity control", "index": index,
                     "user": block[0].user, "label": block[0].label,
                     "max_channel_glr_2loglambda": score, "winning_channel": feature})
    return rows


def plot_event_control_contrast(rows: list[dict[str, object]], output: Path) -> None:
    groups = ("within-activity control", "label boundary")
    values = {group: [float(row["max_channel_glr_2loglambda"]) for row in rows if row["group"] == group]
              for group in groups}
    quantiles = (50, 75, 90, 95)
    high = max(percentile(values[group], 97.5) for group in groups) * 1.12
    width, height = 1050, 630
    lines = svg_header(width, height, "Do WISDM label boundaries look more unusual than ordinary stair motion?",
                       "Maximum exact mean-shift GLRT across x, y, z, and magnitude; 0.5 s on each side")
    x0, y0, panel_w, panel_h = 105, 100, 820, 330
    panel_frame(lines, x0, y0, panel_w, panel_h, 0, high, "2 log Λ")
    group_x = {groups[0]: x0 + panel_w * 0.30, groups[1]: x0 + panel_w * 0.72}
    for group, color in zip(groups, ("#777777", "#D55E00")):
        x = group_x[group]
        for quantile in quantiles:
            value = percentile(values[group], quantile)
            y = y0 + panel_h * (high - min(value, high)) / high
            lines += [f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>',
                      f'<text class="small" x="{x + 12}" y="{y + 4}">p{quantile}: {value:.2f}</text>']
        lines += [f'<text class="label" x="{x}" y="{y0 + panel_h + 30}" text-anchor="middle">{html.escape(group)}</text>',
                  f'<text class="small" x="{x}" y="{y0 + panel_h + 50}" text-anchor="middle">n={len(values[group])}</text>']
    control_p95 = percentile(values[groups[0]], 95)
    event_above = sum(value > control_p95 for value in values[groups[1]])
    lines += [
        f'<text class="label" x="{width / 2}" y="525" text-anchor="middle">Only {event_above}/{len(values[groups[1]])} label boundaries exceed the matched control 95th percentile ({control_p95:.2f}).</text>',
        f'<text class="small" x="{width / 2}" y="552" text-anchor="middle">The labeled switch is not cleanly separable from normal within-activity stair oscillation by a raw mean shift.</text>',
        f'<text class="tiny" x="{width / 2}" y="590" text-anchor="middle">Controls: same three users and stair labels, regular timing, fixed boundaries sampled every 1 s. Observations are temporally dependent.</text>',
    ]
    save_svg(lines, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wisdm-file", type=Path, default=WISDM_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples, malformed = parse_wisdm(args.wisdm_file)
    boundaries = wisdm_boundaries(samples)
    same_user_steps = [(right.timestamp_ns - left.timestamp_ns) / 1e9
                       for left, right in zip(samples, samples[1:]) if left.user == right.user]
    positive_steps = [step for step in same_user_steps if step > 0]
    continuous = [event for event in boundaries if 0 < float(event["gap_seconds"]) <= 0.5]
    examples = select_examples(samples, boundaries)
    effects = event_effects(samples, boundaries)
    sensitivity = wisdm_window_sensitivity(samples, boundaries)
    contrast = wisdm_event_control_contrast(samples, boundaries)

    plot_boundary_gap_audit(boundaries, args.output_dir / "01_wisdm_boundary_gap_audit.svg")
    plot_raw_examples(samples, examples, args.output_dir / "02_wisdm_continuous_raw_examples.svg")
    long_context = [event for event in boundaries
                    if 0 < float(event["gap_seconds"]) <= 0.5
                    and has_regular_context(samples, event, radius=100)]
    if long_context:
        plot_simple_features(samples, long_context[0],
                             args.output_dir / "03_wisdm_simple_features.svg", radius=100)
    plot_window_sensitivity(sensitivity, args.output_dir / "04_wisdm_window_sensitivity.svg")
    plot_event_control_contrast(contrast, args.output_dir / "05_wisdm_event_control_contrast.svg")
    write_events(args.output_dir / "wisdm_continuous_boundary_effects.csv", effects)
    write_events(args.output_dir / "wisdm_window_sensitivity.csv", sensitivity)
    write_events(args.output_dir / "wisdm_event_control_contrast.csv", contrast)

    event_scores = [float(row["max_channel_glr_2loglambda"]) for row in contrast
                    if row["group"] == "label boundary"]
    control_scores = [float(row["max_channel_glr_2loglambda"]) for row in contrast
                      if row["group"] == "within-activity control"]
    control_p95 = percentile(control_scores, 95)

    facts = {
        "dataset": "WISDM Activity Prediction v1.1 (accelerometer only; not WISDM UCI ID 507)",
        "parsed_records": len(samples),
        "malformed_records": malformed,
        "users": len({row.user for row in samples}),
        "label_counts": dict(Counter(row.label for row in samples)),
        "nominal_sampling_hz": 20,
        "observed_positive_step_seconds": {
            "median": median(positive_steps),
            "p05": percentile(positive_steps, 5),
            "p95": percentile(positive_steps, 95),
        },
        "same_user_nonpositive_timestamp_steps": sum(step <= 0 for step in same_user_steps),
        "adjacent_label_changes": len(boundaries),
        "continuous_label_changes_gap_le_0_5s": len(continuous),
        "continuous_label_change_users": sorted({int(event["user"]) for event in continuous}),
        "continuous_label_change_pairs": {
            f"{before}->{after}": count for (before, after), count in
            sorted(Counter((str(event["before"]), str(event["after"])) for event in continuous).items())
        },
        "events_with_regular_1_5s_context_each_side": len(effects),
        "fixed_boundary_event_control_check_0_5s_each_side": {
            "label_boundary_n": len(event_scores),
            "label_boundary_median": median(event_scores),
            "within_activity_control_n": len(control_scores),
            "within_activity_control_median": median(control_scores),
            "within_activity_control_p95": control_p95,
            "label_boundaries_above_control_p95": sum(value > control_p95 for value in event_scores),
        },
        "interpretation": [
            "The file's median positive step agrees with the documented 20 Hz rate.",
            "Most adjacent label changes are separated by gaps and are not observed physical transitions.",
            "The rare near-contiguous label changes cover only stair-direction switches and three users.",
            "This WISDM release has no gyroscope; it is secondary evidence for continuous change detection.",
        ],
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
