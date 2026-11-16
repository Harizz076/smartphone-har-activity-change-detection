#!/usr/bin/env python3
"""Raw, continuous UCI 341/HAPT transition audit.

Labels are used only to locate diagnostic event intervals and controls.  The
computed scores themselves use raw accelerometer/gyroscope values only.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

try:
    from analysis.fundamental_signal_audit import (
        COLORS, mean_shift_glr, panel_frame, percentile, polyline, save_svg, svg_header,
    )
except ModuleNotFoundError:  # direct execution: python analysis/uci341_raw_audit.py
    from fundamental_signal_audit import (
        COLORS, mean_shift_glr, panel_frame, percentile, polyline, save_svg, svg_header,
    )


ROOT = Path(__file__).resolve().parent.parent
DATA_DEFAULT = ROOT / "Datasets/uci341_postural_transitions"
OUT_DEFAULT = ROOT / "analysis/fundamental_signal_audit"
TRANSITION_IDS = frozenset(range(7, 13))
POSTURE_IDS = frozenset((4, 5, 6))
FILE_RE = re.compile(r"acc_exp(\d+)_user(\d+)\.txt$")


@dataclass(frozen=True)
class Event:
    experiment: int
    user: int
    activity: int
    start: int  # zero-based, inclusive
    end: int  # zero-based, exclusive

    @property
    def duration_samples(self) -> int:
        return self.end - self.start


def read_activity_labels(path: Path) -> dict[int, str]:
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            labels[int(fields[0])] = " ".join(fields[1:])
    return labels


def read_events(path: Path) -> list[Event]:
    values = np.loadtxt(path, dtype=int)
    # Adopt one-based, inclusive sample numbers: adjacent annotations commonly
    # satisfy next_start = previous_end + 1, and Python's [start-1:end]
    # preserves exactly that set. The supplied README names start/end samples
    # but does not explicitly state the indexing origin, so report this choice.
    return [Event(int(exp), int(user), int(activity), int(start) - 1, int(end))
            for exp, user, activity, start, end in values]


def discover_streams(raw: Path) -> dict[tuple[int, int], tuple[Path, Path]]:
    streams = {}
    for acc_path in sorted(raw.glob("acc_exp*_user*.txt")):
        match = FILE_RE.match(acc_path.name)
        if not match:
            continue
        experiment, user = map(int, match.groups())
        gyro_path = raw / f"gyro_exp{experiment:02d}_user{user:02d}.txt"
        if not gyro_path.exists():
            raise FileNotFoundError(f"Missing pair for {acc_path.name}")
        streams[(experiment, user)] = (acc_path, gyro_path)
    return streams


def load_stream(pair: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray]:
    acc, gyro = np.loadtxt(pair[0]), np.loadtxt(pair[1])
    if acc.ndim != 2 or acc.shape[1] != 3 or gyro.shape != acc.shape:
        raise ValueError(f"Unaligned or invalid sensor pair: {pair}")
    if not np.isfinite(acc).all() or not np.isfinite(gyro).all():
        raise ValueError(f"Non-finite values in sensor pair: {pair}")
    return acc, gyro


def validate(events: list[Event], streams: dict[tuple[int, int], tuple[Path, Path]]) -> tuple[dict, dict]:
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    lengths = {}
    mismatches = []
    for key, pair in streams.items():
        data = load_stream(pair)
        cache[key] = data
        lengths[key] = len(data[0])
    for event in events:
        length = lengths[(event.experiment, event.user)]
        if not (0 <= event.start < event.end <= length):
            mismatches.append((event.experiment, event.user, event.start, event.end, length))
    if mismatches:
        raise ValueError(f"Out-of-bounds labels: {mismatches[:5]}")
    return cache, lengths


def select_examples(events: list[Event]) -> list[Event]:
    selected = []
    for activity in sorted(TRANSITION_IDS):
        choices = [event for event in events if event.activity == activity]
        target = median([event.duration_samples for event in choices])
        selected.append(min(choices, key=lambda event: (abs(event.duration_samples - target), event.user)))
    return selected


def sensor_series(values: np.ndarray) -> list[list[float]]:
    return [values[:, axis].tolist() for axis in range(3)] + [np.linalg.norm(values, axis=1).tolist()]


def plot_raw_examples(events: list[Event], labels: dict[int, str], cache: dict,
                      output: Path, context: int = 100) -> None:
    examples = select_examples(events)
    width, height = 1420, 110 + 315 * len(examples)
    lines = svg_header(width, height, "UCI 341 raw signals around all six postural-transition types",
                       "Shading is the supplied transition interval; 2 s of context is shown before and after")
    for row, event in enumerate(examples):
        acc, gyro = cache[(event.experiment, event.user)]
        left = max(0, event.start - context)
        right = min(len(acc), event.end + context)
        for col, (sensor_name, units, data) in enumerate((
            ("accelerometer", "g", acc[left:right]),
            ("gyroscope", "rad/s", gyro[left:right]),
        )):
            x0, y0, panel_w, panel_h = 75 + col * 690, 95 + row * 315, 610, 215
            series = sensor_series(data)
            low = min(value for values in series for value in values)
            high = max(value for values in series for value in values)
            pad = max((high - low) * 0.06, 0.01)
            low, high = low - pad, high + pad
            start_x = x0 + panel_w * (event.start - left) / max(right - left - 1, 1)
            end_x = x0 + panel_w * (event.end - left - 1) / max(right - left - 1, 1)
            panel_frame(lines, x0, y0, panel_w, panel_h, low, high, units)
            lines.append(f'<rect x="{start_x:.1f}" y="{y0}" width="{max(end_x-start_x, 1):.1f}" height="{panel_h}" fill="#F0E442" opacity="0.25"/>')
            for values, color in zip(series, COLORS):
                lines.append(polyline(values, x0, y0, panel_w, panel_h, low, high, color, 0.9))
            lines += [
                f'<line x1="{start_x}" y1="{y0}" x2="{start_x}" y2="{y0+panel_h}" stroke="#111" stroke-dasharray="4 3"/>',
                f'<line x1="{end_x}" y1="{y0}" x2="{end_x}" y2="{y0+panel_h}" stroke="#111" stroke-dasharray="4 3"/>',
                f'<text class="small" x="{x0}" y="{y0-10}">{sensor_name}; exp {event.experiment}, user {event.user}</text>',
            ]
        lines.append(f'<text class="label" x="{width/2}" y="{82 + row*315}" text-anchor="middle">{html.escape(labels[event.activity])} — {event.duration_samples/50:.2f} s</text>')
    for index, (name, color) in enumerate(zip(("x", "y", "z", "magnitude"), COLORS)):
        x = 440 + index * 145
        lines += [f'<line x1="{x}" y1="{height-25}" x2="{x+24}" y2="{height-25}" stroke="{color}" stroke-width="3"/>',
                  f'<text class="small" x="{x+31}" y="{height-21}">{name}</text>']
    save_svg(lines, output)


def plot_durations(events: list[Event], labels: dict[int, str], output: Path) -> None:
    width, height = 1150, 650
    lines = svg_header(width, height, "UCI 341 transition-duration audit",
                       "Intervals are explicit durations, not point labels; all values use the documented 50 Hz rate")
    x0, y0, panel_w, panel_h = 130, 95, 900, 390
    durations = {activity: [event.duration_samples / 50 for event in events if event.activity == activity]
                 for activity in sorted(TRANSITION_IDS)}
    high = max(max(values) for values in durations.values()) * 1.06
    panel_frame(lines, x0, y0, panel_w, panel_h, 0, high, "seconds")
    for index, activity in enumerate(sorted(durations)):
        values = durations[activity]
        x = x0 + panel_w * (index + 0.5) / len(durations)
        q25, med, q75 = (percentile(values, q) for q in (25, 50, 75))
        scale = lambda value: y0 + panel_h * (high - value) / high
        lines += [
            f'<line x1="{x}" y1="{scale(min(values))}" x2="{x}" y2="{scale(max(values))}" stroke="#555"/>',
            f'<rect x="{x-28}" y="{scale(q75)}" width="56" height="{scale(q25)-scale(q75)}" fill="#56B4E9" stroke="#333"/>',
            f'<line x1="{x-28}" y1="{scale(med)}" x2="{x+28}" y2="{scale(med)}" stroke="#111" stroke-width="2"/>',
            f'<text class="tiny" x="{x}" y="{y0+panel_h+23}" text-anchor="middle">{html.escape(labels[activity].replace("_", " "))}</text>',
            f'<text class="tiny" x="{x}" y="{y0+panel_h+42}" text-anchor="middle">n={len(values)}, med={med:.2f}s</text>',
        ]
    lines.append(f'<text class="small" x="{width/2}" y="590" text-anchor="middle">Boxes are interquartile ranges; whiskers are observed minima and maxima.</text>')
    save_svg(lines, output)


def max_sensor_glr(data: np.ndarray, split: int, radius: int) -> tuple[float, str]:
    before, after = data[split-radius:split], data[split:split+radius]
    candidates = []
    for axis, name in enumerate(("x", "y", "z")):
        score, _ = mean_shift_glr(before[:, axis].tolist(), after[:, axis].tolist())
        candidates.append((score, name))
    score, _ = mean_shift_glr(np.linalg.norm(before, axis=1).tolist(),
                              np.linalg.norm(after, axis=1).tolist())
    candidates.append((score, "magnitude"))
    return max(candidates)


def timing_sensitivity(events: list[Event], cache: dict) -> list[dict[str, object]]:
    rows = []
    for event in events:
        if event.activity not in TRANSITION_IDS:
            continue
        acc, gyro = cache[(event.experiment, event.user)]
        landmarks = {"onset": event.start, "midpoint": (event.start + event.end) // 2, "end": event.end}
        for seconds in (0.25, 0.5, 1.0, 2.0):
            radius = int(round(seconds * 50))
            for landmark, split in landmarks.items():
                if split - radius < 0 or split + radius > len(acc):
                    continue
                for sensor, data in (("accelerometer", acc), ("gyroscope", gyro)):
                    score, channel = max_sensor_glr(data, split, radius)
                    rows.append({
                        "experiment": event.experiment, "user": event.user,
                        "activity": event.activity, "landmark": landmark,
                        "seconds_each_side": seconds, "sensor": sensor,
                        "max_channel_glr_2loglambda": score, "winning_channel": channel,
                    })
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_timing_sensitivity(rows: list[dict[str, object]], output: Path) -> None:
    durations = sorted({float(row["seconds_each_side"]) for row in rows})
    width, height = 1270, 650
    lines = svg_header(width, height, "Where is the raw mean-shift evidence inside a labeled transition?",
                       "Median maximum-channel exact GLRT across all 358 transition intervals")
    for panel, sensor in enumerate(("accelerometer", "gyroscope")):
        x0, y0, panel_w, panel_h = 85 + panel * 615, 105, 520, 350
        curves = {}
        for landmark in ("onset", "midpoint", "end"):
            curves[landmark] = [median([float(row["max_channel_glr_2loglambda"]) for row in rows
                                        if row["sensor"] == sensor and row["landmark"] == landmark
                                        and float(row["seconds_each_side"]) == duration])
                                for duration in durations]
        high = max(value for curve in curves.values() for value in curve) * 1.10
        panel_frame(lines, x0, y0, panel_w, panel_h, 0, high, "median 2 log Λ")
        for landmark, color in zip(("onset", "midpoint", "end"), COLORS[:3]):
            points = []
            for duration, value in zip(durations, curves[landmark]):
                x = x0 + panel_w * (math.log2(duration) - math.log2(durations[0])) / (math.log2(durations[-1]) - math.log2(durations[0]))
                y = y0 + panel_h * (high - value) / high
                points.append(f"{x:.1f},{y:.1f}")
                lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{" ".join(points)}"/>')
        for duration in durations:
            x = x0 + panel_w * (math.log2(duration) - math.log2(durations[0])) / (math.log2(durations[-1]) - math.log2(durations[0]))
            lines.append(f'<text class="tiny" x="{x}" y="{y0+panel_h+20}" text-anchor="middle">{duration:g}s</text>')
        lines += [f'<text class="label" x="{x0+panel_w/2}" y="{y0-15}" text-anchor="middle">{sensor}</text>',
                  f'<text class="small" x="{x0+panel_w/2}" y="{y0+panel_h+48}" text-anchor="middle">context on each side of landmark</text>']
    for index, (landmark, color) in enumerate(zip(("onset", "midpoint", "end"), COLORS[:3])):
        x = 425 + index * 160
        lines += [f'<line x1="{x}" y1="545" x2="{x+26}" y2="545" stroke="{color}" stroke-width="3"/>',
                  f'<text class="small" x="{x+33}" y="549">{landmark}</text>']
    lines.append(f'<text class="small" x="{width/2}" y="600" text-anchor="middle">This diagnostic uses labels to test landmark definitions; an online detector does not know these locations.</text>')
    save_svg(lines, output)


def event_control_contrast(events: list[Event], cache: dict, radius: int = 25) -> list[dict[str, object]]:
    rows = []
    for event in events:
        if event.activity not in TRANSITION_IDS:
            continue
        acc, gyro = cache[(event.experiment, event.user)]
        landmarks = (event.start, (event.start + event.end) // 2, event.end)
        for sensor, data in (("accelerometer", acc), ("gyroscope", gyro)):
            candidates = []
            for split in landmarks:
                if radius <= split <= len(data) - radius:
                    score, channel = max_sensor_glr(data, split, radius)
                    candidates.append((score, split, channel))
            score, split, channel = max(candidates)
            rows.append({"group": "transition event", "experiment": event.experiment,
                         "user": event.user, "activity": event.activity,
                         "interval_start": event.start, "sensor": sensor,
                         "glr_2loglambda": score, "sample_index": split,
                         "winning_channel": channel})
    for interval in events:
        if interval.activity not in POSTURE_IDS:
            continue
        acc, gyro = cache[(interval.experiment, interval.user)]
        # Controls are at least one window from interval edges and spaced 2 s
        # apart to reduce, not eliminate, temporal dependence.
        for split in range(interval.start + radius, interval.end - radius + 1, 100):
            for sensor, data in (("accelerometer", acc), ("gyroscope", gyro)):
                score, channel = max_sensor_glr(data, split, radius)
                rows.append({"group": "stable-posture control", "experiment": interval.experiment,
                             "user": interval.user, "activity": interval.activity,
                             "interval_start": interval.start, "sensor": sensor,
                             "glr_2loglambda": score, "sample_index": split,
                             "winning_channel": channel})
    return rows


def plot_event_control(rows: list[dict[str, object]], output: Path) -> None:
    width, height = 1230, 690
    lines = svg_header(width, height, "Are labeled postural transitions distinct from stable-posture fluctuations?",
                       "0.5 s per side; max channel, and for events the best of onset/midpoint/end")
    for panel, sensor in enumerate(("accelerometer", "gyroscope")):
        x0, y0, panel_w, panel_h = 100 + panel * 600, 105, 485, 350
        groups = ("stable-posture control", "transition event")
        values = {group: [float(row["glr_2loglambda"]) for row in rows
                          if row["sensor"] == sensor and row["group"] == group] for group in groups}
        high = max(percentile(values[group], 97.5) for group in groups) * 1.08
        panel_frame(lines, x0, y0, panel_w, panel_h, 0, high, "2 log Λ")
        for group_index, (group, color) in enumerate(zip(groups, ("#777777", "#D55E00"))):
            x = x0 + panel_w * (0.28 + 0.48 * group_index)
            for quantile in (50, 75, 90, 95):
                value = percentile(values[group], quantile)
                y = y0 + panel_h * (high - min(value, high)) / high
                lines += [f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}"/>',
                          f'<text class="tiny" x="{x+9}" y="{y+3}">p{quantile} {value:.1f}</text>']
            lines += [f'<text class="tiny" x="{x}" y="{y0+panel_h+22}" text-anchor="middle">{html.escape(group)}</text>',
                      f'<text class="tiny" x="{x}" y="{y0+panel_h+39}" text-anchor="middle">n={len(values[group])}</text>']
        threshold = percentile(values[groups[0]], 95)
        hits = sum(value > threshold for value in values[groups[1]])
        lines += [f'<text class="label" x="{x0+panel_w/2}" y="{y0-16}" text-anchor="middle">{sensor}</text>',
                  f'<text class="small" x="{x0+panel_w/2}" y="{y0+panel_h+72}" text-anchor="middle">events above control p95: {hits}/{len(values[groups[1]])} ({100*hits/len(values[groups[1]]):.1f}%)</text>']
    lines += [
        f'<text class="small" x="{width/2}" y="610" text-anchor="middle">The event score is optimistic because it takes the best of three label-derived landmarks; deployment must scan without labels.</text>',
        f'<text class="tiny" x="{width/2}" y="638" text-anchor="middle">Controls span standing, sitting, and lying; repeated controls within streams are not independent.</text>',
    ]
    save_svg(lines, output)


def plot_winning_channels(rows: list[dict[str, object]], output: Path) -> None:
    width, height = 1050, 590
    lines = svg_header(width, height, "Is magnitude enough, or do individual axes carry the strongest change?",
                       "Winning raw channel among x, y, z, and magnitude for 358 transition-event scores")
    channels = ("x", "y", "z", "magnitude")
    sensors = ("accelerometer", "gyroscope")
    x0, y0, panel_w, panel_h = 105, 100, 835, 330
    totals = {sensor: sum(row["sensor"] == sensor and row["group"] == "transition event" for row in rows)
              for sensor in sensors}
    percentages = {sensor: [100 * sum(row["sensor"] == sensor and row["group"] == "transition event"
                                      and row["winning_channel"] == channel for row in rows) / totals[sensor]
                            for channel in channels] for sensor in sensors}
    high = max(value for values in percentages.values() for value in values) * 1.15
    panel_frame(lines, x0, y0, panel_w, panel_h, 0, high, "transition events (%)")
    group_width = panel_w / len(channels)
    bar_width = group_width * 0.30
    for channel_index, channel in enumerate(channels):
        for sensor_index, (sensor, color) in enumerate(zip(sensors, ("#0072B2", "#D55E00"))):
            value = percentages[sensor][channel_index]
            x = x0 + group_width * channel_index + group_width * (0.17 + 0.38 * sensor_index)
            height_px = panel_h * value / high
            lines += [f'<rect x="{x}" y="{y0+panel_h-height_px}" width="{bar_width}" height="{height_px}" fill="{color}"/>',
                      f'<text class="tiny" x="{x+bar_width/2}" y="{y0+panel_h-height_px-5}" text-anchor="middle">{value:.1f}%</text>']
        center = x0 + group_width * (channel_index + 0.5)
        lines.append(f'<text class="small" x="{center}" y="{y0+panel_h+24}" text-anchor="middle">{channel}</text>')
    for index, (sensor, color) in enumerate(zip(sensors, ("#0072B2", "#D55E00"))):
        x = 350 + index * 220
        lines += [f'<rect x="{x}" y="500" width="22" height="12" fill="{color}"/>',
                  f'<text class="small" x="{x+30}" y="511">{sensor}</text>']
    lines.append(f'<text class="small" x="{width/2}" y="550" text-anchor="middle">A magnitude-only detector discards every event whose strongest scalar change is axis-specific.</text>')
    save_svg(lines, output)


def plot_transition_type_rates(rows: list[dict[str, object]], labels: dict[int, str], output: Path) -> None:
    sensors = ("accelerometer", "gyroscope")
    thresholds = {
        sensor: percentile([float(row["glr_2loglambda"]) for row in rows
                            if row["sensor"] == sensor and row["group"] == "stable-posture control"], 95)
        for sensor in sensors
    }
    activities = sorted(TRANSITION_IDS)
    rates = {}
    for sensor in sensors:
        rates[sensor] = []
        for activity in activities:
            scores = [float(row["glr_2loglambda"]) for row in rows
                      if row["sensor"] == sensor and row["group"] == "transition event"
                      and int(row["activity"]) == activity]
            rates[sensor].append(100 * sum(score > thresholds[sensor] for score in scores) / len(scores))
    width, height = 1250, 630
    lines = svg_header(width, height, "Simple GLRT sensitivity differs by transition type",
                       "Event score above each sensor's stable-posture 95th percentile; optimistic best landmark")
    x0, y0, panel_w, panel_h = 110, 100, 1025, 355
    panel_frame(lines, x0, y0, panel_w, panel_h, 0, 100, "events above threshold (%)")
    group_width = panel_w / len(activities)
    bar_width = group_width * 0.30
    for activity_index, activity in enumerate(activities):
        for sensor_index, (sensor, color) in enumerate(zip(sensors, ("#0072B2", "#D55E00"))):
            value = rates[sensor][activity_index]
            x = x0 + group_width * activity_index + group_width * (0.17 + 0.38 * sensor_index)
            height_px = panel_h * value / 100
            lines += [f'<rect x="{x}" y="{y0+panel_h-height_px}" width="{bar_width}" height="{height_px}" fill="{color}"/>',
                      f'<text class="tiny" x="{x+bar_width/2}" y="{y0+panel_h-height_px-5}" text-anchor="middle">{value:.1f}</text>']
        center = x0 + group_width * (activity_index + 0.5)
        name = labels[activity].replace("_", " ")
        lines.append(f'<text class="tiny" x="{center}" y="{y0+panel_h+22}" text-anchor="middle">{html.escape(name)}</text>')
    for index, (sensor, color) in enumerate(zip(sensors, ("#0072B2", "#D55E00"))):
        x = 410 + index * 230
        lines += [f'<rect x="{x}" y="535" width="22" height="12" fill="{color}"/>',
                  f'<text class="small" x="{x+30}" y="546">{sensor}</text>']
    lines.append(f'<text class="small" x="{width/2}" y="590" text-anchor="middle">A class-agnostic detector can retain both sensors; optional class naming should remain a later stage.</text>')
    save_svg(lines, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.dataset_root / "RawData"
    labels = read_activity_labels(args.dataset_root / "activity_labels.txt")
    events = read_events(raw / "labels.txt")
    streams = discover_streams(raw)
    cache, lengths = validate(events, streams)
    transitions = [event for event in events if event.activity in TRANSITION_IDS]

    plot_raw_examples(transitions, labels, cache, args.output_dir / "06_uci341_raw_transition_examples.svg")
    plot_durations(transitions, labels, args.output_dir / "07_uci341_transition_durations.svg")
    timing = timing_sensitivity(transitions, cache)
    plot_timing_sensitivity(timing, args.output_dir / "08_uci341_timing_window_sensitivity.svg")
    contrast = event_control_contrast(events, cache)
    plot_event_control(contrast, args.output_dir / "09_uci341_event_control_contrast.svg")
    plot_winning_channels(contrast, args.output_dir / "10_uci341_winning_channels.svg")
    plot_transition_type_rates(contrast, labels, args.output_dir / "11_uci341_transition_type_rates.svg")
    write_rows(args.output_dir / "uci341_timing_window_sensitivity.csv", timing)
    write_rows(args.output_dir / "uci341_event_control_contrast.csv", contrast)

    contrast_summary = {}
    hit_keys = {}
    for sensor in ("accelerometer", "gyroscope"):
        controls = [float(row["glr_2loglambda"]) for row in contrast
                    if row["sensor"] == sensor and row["group"] == "stable-posture control"]
        transition_scores = [float(row["glr_2loglambda"]) for row in contrast
                             if row["sensor"] == sensor and row["group"] == "transition event"]
        threshold = percentile(controls, 95)
        contrast_summary[sensor] = {
            "transition_n": len(transition_scores), "control_n": len(controls),
            "transition_median": median(transition_scores), "control_median": median(controls),
            "control_p95": threshold,
            "transitions_above_control_p95": sum(value > threshold for value in transition_scores),
            "transition_winning_channel_counts": dict(Counter(
                str(row["winning_channel"]) for row in contrast
                if row["sensor"] == sensor and row["group"] == "transition event"
            )),
            "per_transition_type_above_control_p95_percent": {
                labels[activity]: 100 * sum(
                    float(row["glr_2loglambda"]) > threshold for row in contrast
                    if row["sensor"] == sensor and row["group"] == "transition event"
                    and int(row["activity"]) == activity
                ) / sum(
                    1 for row in contrast if row["sensor"] == sensor
                    and row["group"] == "transition event" and int(row["activity"]) == activity
                ) for activity in sorted(TRANSITION_IDS)
            },
        }
        hit_keys[sensor] = {
            (int(row["experiment"]), int(row["user"]), int(row["activity"]), int(row["interval_start"]))
            for row in contrast if row["sensor"] == sensor and row["group"] == "transition event"
            and float(row["glr_2loglambda"]) > threshold
        }
    contrast_summary["complementarity_at_sensor_specific_control_p95"] = {
        "detected_by_both": len(hit_keys["accelerometer"] & hit_keys["gyroscope"]),
        "accelerometer_only": len(hit_keys["accelerometer"] - hit_keys["gyroscope"]),
        "gyroscope_only": len(hit_keys["gyroscope"] - hit_keys["accelerometer"]),
        "detected_by_either": len(hit_keys["accelerometer"] | hit_keys["gyroscope"]),
    }
    summary = {
        "dataset": "UCI 341 / HAPT raw inertial data",
        "sampling_hz": 50,
        "streams": len(streams), "subjects": len({event.user for event in events}),
        "total_raw_samples_per_sensor": sum(lengths.values()),
        "accel_gyro_length_mismatches": 0,
        "labeled_intervals": len(events), "transition_intervals": len(transitions),
        "transition_counts": {labels[activity]: count for activity, count in
                              sorted(Counter(event.activity for event in transitions).items())},
        "transition_duration_seconds": {
            labels[activity]: {
                "median": median([event.duration_samples / 50 for event in transitions if event.activity == activity]),
                "min": min(event.duration_samples / 50 for event in transitions if event.activity == activity),
                "max": max(event.duration_samples / 50 for event in transitions if event.activity == activity),
            } for activity in sorted(TRANSITION_IDS)
        },
        "event_control_check": contrast_summary,
        "endpoint_convention": "adopted one-based inclusive input endpoints; Python slice is [start-1:end]",
    }
    (args.output_dir / "uci341_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
