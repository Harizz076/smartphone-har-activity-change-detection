#!/usr/bin/env python3
"""Gravity-aligned, orientation-robust acceleration features."""
from __future__ import annotations

import math
from statistics import fmean
from typing import Sequence

from glrt.six_class_glrt import extract_features


GRAVITY_FEATURE_NAMES = (
    "gravity_magnitude",
    "log_dynamic_power",
    "log_vertical_power",
    "log_horizontal_power",
    "log_vertical_jerk",
    "log_horizontal_jerk",
    "magnitude_dominant_frequency_hz",
    "magnitude_spectral_entropy",
    "magnitude_log_band_power",
)


def _log_power(values: Sequence[float]) -> float:
    return math.log(max(fmean(value * value for value in values), 1e-12))


def _scalar_spectrum(values: Sequence[float], sample_rate: float,
                     minimum_hz: float = 0.3,
                     maximum_hz: float = 5.0) -> tuple[float, float, float]:
    count = len(values)
    average = fmean(values)
    centered = [value - average for value in values]
    first_bin = max(1, math.ceil(minimum_hz * count / sample_rate))
    last_bin = min(count // 2, math.floor(maximum_hz * count / sample_rate))
    if first_bin > last_bin:
        raise ValueError("Window is too short for the requested spectral band")
    powers = []
    for frequency_bin in range(first_bin, last_bin + 1):
        angle_step = 2.0 * math.pi * frequency_bin / count
        real = sum(value * math.cos(angle_step * index)
                   for index, value in enumerate(centered))
        imaginary = -sum(value * math.sin(angle_step * index)
                         for index, value in enumerate(centered))
        powers.append((frequency_bin, real * real + imaginary * imaginary))
    total_power = sum(power for _, power in powers)
    if total_power <= 0.0:
        return 0.0, 0.0, math.log(1e-12)
    dominant = max(powers, key=lambda item: item[1])[0] * sample_rate / count
    probabilities = [power / total_power for _, power in powers if power > 0.0]
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalized_entropy = entropy / math.log(len(powers)) if len(powers) > 1 else 0.0
    return dominant, normalized_entropy, math.log(max(total_power / (count * count), 1e-12))


def extract_gravity_features(
    samples: Sequence[tuple[float, float, float]],
    sample_rate: float,
) -> tuple[float, ...]:
    """Append gravity-aligned vertical/horizontal motion summaries.

    The window mean estimates the local gravity direction. Samples centered by
    that vector are projected onto the gravity unit vector; the residual norm
    is horizontal motion and is invariant to rotation about gravity.
    """
    if len(samples) < 3:
        raise ValueError("Feature extraction requires at least three samples")
    mean = tuple(fmean(sample[axis] for sample in samples) for axis in range(3))
    gravity = math.sqrt(sum(value * value for value in mean))
    if gravity <= 1e-12:
        unit = (0.0, 0.0, 1.0)
    else:
        unit = tuple(value / gravity for value in mean)

    vertical = []
    horizontal = []
    dynamic_magnitudes = []
    for sample in samples:
        centered = tuple(value - center for value, center in zip(sample, mean))
        projection = sum(value * direction for value, direction in zip(centered, unit))
        centered_power = sum(value * value for value in centered)
        horizontal_power = max(centered_power - projection * projection, 0.0)
        vertical.append(projection)
        horizontal.append(math.sqrt(horizontal_power))
        dynamic_magnitudes.append(math.sqrt(centered_power))

    vertical_jerk = [current - previous
                     for previous, current in zip(vertical, vertical[1:])]
    horizontal_jerk = [current - previous
                       for previous, current in zip(horizontal, horizontal[1:])]
    spectral = _scalar_spectrum(dynamic_magnitudes, sample_rate)
    extra = (
        gravity,
        _log_power(dynamic_magnitudes),
        _log_power(vertical),
        _log_power(horizontal),
        _log_power(vertical_jerk),
        _log_power(horizontal_jerk),
        *spectral,
    )
    result = (*extract_features(samples, sample_rate), *extra)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Gravity-aligned feature extraction produced non-finite values")
    return result
