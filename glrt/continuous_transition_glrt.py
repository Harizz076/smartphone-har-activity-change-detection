#!/usr/bin/env python3
"""Continuous multichannel Gaussian GLRT scans for inertial streams.

The detector is label-free.  At each causal update it examines a trailing
horizon, maximizes the likelihood ratio over admissible change locations, and
returns both the maximum score and estimated change sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanResult:
    trigger_samples: np.ndarray
    scores: np.ndarray
    change_samples: np.ndarray


def _validate(signal: np.ndarray, horizon: int, minimum_segment: int,
              hop: int, candidate_step: int, variance_model: str) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("signal must have shape (samples, channels)")
    if not np.isfinite(values).all():
        raise ValueError("signal contains non-finite values")
    if horizon <= 2 * minimum_segment:
        raise ValueError("horizon must exceed twice the minimum segment length")
    if min(minimum_segment, hop, candidate_step) < 1:
        raise ValueError("segment, hop, and candidate step must be positive")
    if variance_model not in {"common", "separate"}:
        raise ValueError("variance_model must be 'common' or 'separate'")
    return values


def scan_glrt(signal: np.ndarray, *, horizon: int, minimum_segment: int,
              hop: int = 1, candidate_step: int = 1,
              variance_model: str = "common") -> ScanResult:
    """Run a causal scan GLRT over a multichannel signal.

    Channels are modeled as conditionally independent Gaussians and share one
    candidate change location. Under H0, each channel has one unknown mean and
    variance. Under H1:

    * ``common``: each channel has separate pre/post means and one variance;
    * ``separate``: each channel has separate pre/post means and variances.

    Per-channel ``2 log Lambda`` values are summed at each candidate split, then
    maximized over the split. Empirical calibration is required because the
    channel-independence model is approximate and the scan changes the null.
    """
    values = _validate(signal, horizon, minimum_segment, hop, candidate_step, variance_model)
    sample_count, channels = values.shape
    if sample_count < horizon:
        return ScanResult(np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int))

    cumulative = np.vstack((np.zeros((1, channels)), np.cumsum(values, axis=0)))
    cumulative_sq = np.vstack((np.zeros((1, channels)), np.cumsum(values * values, axis=0)))
    trigger_ends = np.arange(horizon, sample_count + 1, hop, dtype=int)
    candidates = np.arange(minimum_segment, horizon - minimum_segment + 1,
                           candidate_step, dtype=int)
    scores = np.empty(len(trigger_ends), dtype=float)
    changes = np.empty(len(trigger_ends), dtype=int)

    for output_index, end in enumerate(trigger_ends):
        start = end - horizon
        total_sum = cumulative[end] - cumulative[start]
        total_sq = cumulative_sq[end] - cumulative_sq[start]
        sse0 = total_sq - total_sum * total_sum / horizon

        split_indices = start + candidates
        left_sum = cumulative[split_indices] - cumulative[start]
        left_sq = cumulative_sq[split_indices] - cumulative_sq[start]
        right_sum = total_sum - left_sum
        right_sq = total_sq - left_sq
        right_count = horizon - candidates
        sse_left = left_sq - left_sum * left_sum / candidates[:, None]
        sse_right = right_sq - right_sum * right_sum / right_count[:, None]

        sse_floor = np.maximum(sse0 * 1e-10, 1e-14 * horizon)
        variance_floor = sse_floor / horizon
        if variance_model == "common":
            sse1 = sse_left + sse_right
            channel_glr = horizon * np.log(
                np.maximum(sse0, sse_floor) / np.maximum(sse1, sse_floor)
            )
        else:
            channel_glr = (
                horizon * np.log(np.maximum(sse0 / horizon, variance_floor))
                - candidates[:, None] * np.log(
                    np.maximum(sse_left / candidates[:, None], variance_floor)
                )
                - right_count[:, None] * np.log(
                    np.maximum(sse_right / right_count[:, None], variance_floor)
                )
            )
        profile = np.maximum(channel_glr, 0.0).sum(axis=1)
        best = int(np.argmax(profile))
        scores[output_index] = profile[best]
        changes[output_index] = start + candidates[best]

    return ScanResult(trigger_ends - 1, scores, changes)


def threshold_crossings(result: ScanResult, annotated_activity: np.ndarray,
                        threshold: float, refractory_samples: int) -> list[tuple[int, int, float]]:
    """Emit rising-threshold alarms with refractory suppression.

    ``annotated_activity`` is not a detector input. It defines evaluable regions:
    negative values reset threshold state and produce no alarm, preventing
    unannotated dataset gaps from being scored as false alarms or transitions.
    """
    if len(annotated_activity) != len(result.scores):
        raise ValueError("annotation length must match scan outputs")
    alarms: list[tuple[int, int, float]] = []
    previously_above = False
    last_alarm = -10**18
    for trigger, score, change, activity in zip(
            result.trigger_samples, result.scores, result.change_samples, annotated_activity):
        if activity < 1:
            previously_above = False
            continue
        above = bool(score > threshold)
        if above and not previously_above and trigger - last_alarm >= refractory_samples:
            alarms.append((int(trigger), int(change), float(score)))
            last_alarm = int(trigger)
        previously_above = above
    return alarms
