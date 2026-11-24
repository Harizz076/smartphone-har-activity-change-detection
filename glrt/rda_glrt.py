#!/usr/bin/env python3
"""Friedman's two-parameter regularized Gaussian GLRT classifier.

This module extends the existing class-conditional Gaussian classifier with
the covariance mixing parameter that was missing from the earlier model.  It
implements Friedman (1989), equations 16 and 18: class scatter is mixed with
pooled within-class scatter, then the resulting covariance eigenvalues are
shrunk toward their mean.
"""
from __future__ import annotations

from statistics import fmean
from typing import Hashable, Sequence

from glrt.six_class_glrt import (
    GaussianClassModel,
    Standardizer,
    fit_standardizer,
    model_from_covariance,
)


def _scatter(vectors: Sequence[Sequence[float]],
             mean: Sequence[float]) -> list[list[float]]:
    dimensions = len(mean)
    return [[
        sum((vector[row] - mean[row]) * (vector[column] - mean[column])
            for vector in vectors)
        for column in range(dimensions)
    ] for row in range(dimensions)]


def fit_rda_models(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[Hashable],
    class_order: Sequence[Hashable],
    covariance_mixing: float,
    eigenvalue_shrinkage: float,
    ridge: float = 1e-6,
) -> tuple[Standardizer, tuple[GaussianClassModel, ...]]:
    """Fit Friedman's RDA covariance family with equal class priors.

    ``covariance_mixing`` is Friedman's lambda: zero is class-specific QDA
    and one uses pooled within-class scatter. ``eigenvalue_shrinkage`` is
    gamma: zero retains the mixed covariance and one makes it spherical.
    The caller must select both values without using held-out test labels.
    """
    if len(vectors) != len(labels) or not vectors:
        raise ValueError("Training features and labels must be non-empty and aligned")
    if len(set(class_order)) != len(class_order):
        raise ValueError("class_order contains duplicates")
    if not 0.0 <= covariance_mixing <= 1.0:
        raise ValueError("covariance_mixing must lie in [0, 1]")
    if not 0.0 <= eigenvalue_shrinkage <= 1.0:
        raise ValueError("eigenvalue_shrinkage must lie in [0, 1]")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    unknown = set(labels) - set(class_order)
    if unknown:
        raise ValueError(f"Training labels are absent from class_order: {unknown}")

    standardizer = fit_standardizer(vectors)
    standardized = [standardizer.transform(vector) for vector in vectors]
    groups = {
        label: [vector for vector, actual in zip(standardized, labels)
                if actual == label]
        for label in class_order
    }
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("Every class needs at least two training windows")

    dimensions = len(standardized[0])
    means = {
        label: tuple(fmean(vector[index] for vector in groups[label])
                     for index in range(dimensions))
        for label in class_order
    }
    scatters = {
        label: _scatter(groups[label], means[label]) for label in class_order
    }
    pooled_scatter = [[
        sum(scatters[label][row][column] for label in class_order)
        for column in range(dimensions)
    ] for row in range(dimensions)]
    total_count = len(standardized)

    models = []
    for label in class_order:
        class_count = len(groups[label])
        mixed_weight = ((1.0 - covariance_mixing) * class_count
                        + covariance_mixing * total_count)
        covariance = [[
            ((1.0 - covariance_mixing) * scatters[label][row][column]
             + covariance_mixing * pooled_scatter[row][column])
            / mixed_weight
            for column in range(dimensions)
        ] for row in range(dimensions)]
        models.append(model_from_covariance(
            label,
            class_count,
            means[label],
            covariance,
            eigenvalue_shrinkage,
            ridge,
        ))
    return standardizer, tuple(models)
