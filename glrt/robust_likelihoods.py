#!/usr/bin/env python3
"""Robust class-conditional likelihoods for supplementary HAR classification.

The primary project task remains continuous change detection (see CODEX.md).
These models are a separate six-class classification experiment comparing the
Gaussian plug-in GLRT with multivariate Student-t and independent Laplace
likelihoods.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, median
from typing import Hashable, Sequence

from glrt.six_class_glrt import (
    Prediction,
    Standardizer,
    cholesky,
    fit_standardizer,
    forward_solve,
)


@dataclass(frozen=True)
class StudentTClassModel:
    label: Hashable
    count: int
    location: tuple[float, ...]
    cholesky: tuple[tuple[float, ...], ...]
    log_determinant: float
    degrees_freedom: float
    em_iterations: int
    em_converged: bool

    def log_likelihood(self, vector: Sequence[float]) -> float:
        if len(vector) != len(self.location):
            raise ValueError("Feature vector dimension differs from Student-t model")
        residual = tuple(value - center for value, center in zip(vector, self.location))
        whitened = forward_solve(self.cholesky, residual)
        mahalanobis = sum(value * value for value in whitened)
        dimensions = len(vector)
        nu = self.degrees_freedom
        return (
            math.lgamma((nu + dimensions) / 2)
            - math.lgamma(nu / 2)
            - 0.5 * (dimensions * math.log(nu * math.pi) + self.log_determinant)
            - 0.5 * (nu + dimensions) * math.log1p(mahalanobis / nu)
        )


@dataclass(frozen=True)
class LaplaceClassModel:
    label: Hashable
    count: int
    location: tuple[float, ...]
    scale: tuple[float, ...]

    def log_likelihood(self, vector: Sequence[float]) -> float:
        if len(vector) != len(self.location):
            raise ValueError("Feature vector dimension differs from Laplace model")
        return -sum(
            math.log(2 * spread) + abs(value - center) / spread
            for value, center, spread in zip(vector, self.location, self.scale)
        )


RobustModel = StudentTClassModel | LaplaceClassModel


def covariance_mle(vectors: Sequence[Sequence[float]],
                   location: Sequence[float]) -> list[list[float]]:
    dimensions = len(location)
    return [[
        fmean((vector[row] - location[row]) * (vector[column] - location[column])
              for vector in vectors)
        for column in range(dimensions)
    ] for row in range(dimensions)]


def regularize_covariance(covariance: Sequence[Sequence[float]], shrinkage: float,
                          ridge: float, diagonal: bool = False) -> list[list[float]]:
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1]")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    dimensions = len(covariance)
    if dimensions == 0 or any(len(row) != dimensions for row in covariance):
        raise ValueError("Covariance must be a non-empty square matrix")
    target = fmean(covariance[index][index] for index in range(dimensions))
    if target <= 0 or not math.isfinite(target):
        target = 1.0
    return [[
        (0.0 if diagonal and row != column else (1 - shrinkage) * covariance[row][column])
        + (shrinkage * target + ridge if row == column else 0.0)
        for column in range(dimensions)
    ] for row in range(dimensions)]


def student_t_em(vectors: Sequence[Sequence[float]], degrees_freedom: float,
                 diagonal: bool = False, max_iterations: int = 30,
                 tolerance: float = 1e-5) -> tuple[tuple[float, ...], list[list[float]], int, bool]:
    """Estimate fixed-df multivariate-t location/scatter with the EM updates.

    A small internal ridge is used only to evaluate Mahalanobis distances. The
    returned scatter is unshrunk; the classification model regularizes it in a
    separate, explicitly tuned step.
    """
    if degrees_freedom <= 0:
        raise ValueError("degrees_freedom must be positive")
    if len(vectors) < 2:
        raise ValueError("Student-t fitting needs at least two vectors")
    dimensions = len(vectors[0])
    if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Student-t training vectors have inconsistent dimensions")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("Invalid EM convergence settings")
    location = tuple(fmean(vector[index] for vector in vectors) for index in range(dimensions))
    scatter = covariance_mle(vectors, location)
    if diagonal:
        scatter = [[scatter[row][column] if row == column else 0.0
                    for column in range(dimensions)] for row in range(dimensions)]
    converged = False
    for iteration in range(1, max_iterations + 1):
        internal = regularize_covariance(scatter, 0.01, 1e-6, diagonal)
        lower = cholesky(internal)
        weights = []
        for vector in vectors:
            residual = tuple(value - center for value, center in zip(vector, location))
            whitened = forward_solve(lower, residual)
            distance = sum(value * value for value in whitened)
            weights.append((degrees_freedom + dimensions) / (degrees_freedom + distance))
        weight_sum = sum(weights)
        updated_location = tuple(
            sum(weight * vector[index] for weight, vector in zip(weights, vectors)) / weight_sum
            for index in range(dimensions)
        )
        updated_scatter = [[
            sum(weight * (vector[row] - updated_location[row])
                * (vector[column] - updated_location[column])
                for weight, vector in zip(weights, vectors)) / len(vectors)
            for column in range(dimensions)
        ] for row in range(dimensions)]
        if diagonal:
            updated_scatter = [[updated_scatter[row][column] if row == column else 0.0
                                for column in range(dimensions)] for row in range(dimensions)]
        location_change = max(abs(new - old) for new, old in zip(updated_location, location))
        scatter_change = max(
            abs(updated_scatter[row][column] - scatter[row][column])
            for row in range(dimensions) for column in range(dimensions)
        )
        scale = max(1.0, max(abs(value) for row in scatter for value in row))
        location, scatter = updated_location, updated_scatter
        if max(location_change, scatter_change / scale) < tolerance:
            converged = True
            break
    return location, scatter, iteration, converged


def student_model_from_estimate(label: Hashable, count: int,
                                location: Sequence[float], scatter: Sequence[Sequence[float]],
                                degrees_freedom: float, shrinkage: float, ridge: float,
                                diagonal: bool, em_iterations: int,
                                em_converged: bool) -> StudentTClassModel:
    covariance = regularize_covariance(scatter, shrinkage, ridge, diagonal)
    lower = cholesky(covariance)
    log_determinant = 2 * sum(math.log(lower[index][index]) for index in range(len(location)))
    return StudentTClassModel(label, count, tuple(location), lower, log_determinant,
                              degrees_freedom, em_iterations, em_converged)


def fit_student_models(vectors: Sequence[Sequence[float]], labels: Sequence[Hashable],
                       class_order: Sequence[Hashable], degrees_freedom: float,
                       shrinkage: float, ridge: float, diagonal: bool = False,
                       max_iterations: int = 30) -> tuple[Standardizer, tuple[StudentTClassModel, ...]]:
    if len(vectors) != len(labels) or not vectors:
        raise ValueError("Training features and labels must be non-empty and aligned")
    standardizer = fit_standardizer(vectors)
    standardized = [standardizer.transform(vector) for vector in vectors]
    models = []
    for label in class_order:
        group = [vector for vector, actual in zip(standardized, labels) if actual == label]
        location, scatter, iterations, converged = student_t_em(
            group, degrees_freedom, diagonal, max_iterations
        )
        models.append(student_model_from_estimate(
            label, len(group), location, scatter, degrees_freedom, shrinkage,
            ridge, diagonal, iterations, converged
        ))
    return standardizer, tuple(models)


def fit_laplace_models(vectors: Sequence[Sequence[float]], labels: Sequence[Hashable],
                       class_order: Sequence[Hashable], shrinkage: float,
                       ridge: float) -> tuple[Standardizer, tuple[LaplaceClassModel, ...]]:
    if len(vectors) != len(labels) or not vectors:
        raise ValueError("Training features and labels must be non-empty and aligned")
    if not 0 <= shrinkage <= 1 or ridge <= 0:
        raise ValueError("Invalid Laplace regularization")
    standardizer = fit_standardizer(vectors)
    standardized = [standardizer.transform(vector) for vector in vectors]
    models = []
    for label in class_order:
        group = [vector for vector, actual in zip(standardized, labels) if actual == label]
        if len(group) < 2:
            raise ValueError(f"Class {label!r} needs at least two training windows")
        dimensions = len(group[0])
        location = tuple(median(vector[index] for vector in group) for index in range(dimensions))
        raw_scale = [fmean(abs(vector[index] - location[index]) for vector in group)
                     for index in range(dimensions)]
        target = fmean(raw_scale) or 1.0
        scale = tuple((1 - shrinkage) * value + shrinkage * target + ridge for value in raw_scale)
        models.append(LaplaceClassModel(label, len(group), location, scale))
    return standardizer, tuple(models)


def predict_robust(vector: Sequence[float], standardizer: Standardizer,
                   models: Sequence[RobustModel]) -> Prediction:
    if len(models) < 2:
        raise ValueError("Multi-hypothesis classification requires at least two models")
    standardized = standardizer.transform(vector)
    scores = tuple((model.label, model.log_likelihood(standardized)) for model in models)
    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    return Prediction(ranked[0][0], 2 * (ranked[0][1] - ranked[1][1]), scores)
