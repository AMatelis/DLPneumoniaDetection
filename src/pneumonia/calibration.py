
"""Probability calibration utilities for pneumonia classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def _validate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary labels and predicted probabilities."""
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if probabilities.size == 0:
        raise ValueError("Probabilities cannot be empty.")

    if probabilities.size != labels.size:
        raise ValueError(
            "Probabilities and labels must have the same length."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must contain only 0 and 1.")

    labels = labels.astype(int)

    if np.unique(labels).size != 2:
        raise ValueError(
            "Calibration requires examples from both classes."
        )

    return probabilities, labels


def fit_probability_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    """Fit a sigmoid calibrator on a separate calibration set.

    Args:
        probabilities: Raw model probabilities for the positive class.
        labels: Ground-truth binary labels (0 or 1).

    Returns:
        A fitted logistic regression calibration model.
    """
    probabilities, labels = _validate_probabilities(
        probabilities, labels
    )

    # Convert probabilities to logits so logistic regression can
    # learn a sigmoid mapping from raw scores to calibrated scores.
    epsilon = 1e-7
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    calibrator = LogisticRegression()
    calibrator.fit(logits, labels)

    return calibrator


def apply_probability_calibrator(
    calibrator: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Apply a fitted calibrator to new model probabilities."""
    probabilities = np.asarray(
        probabilities, dtype=float
    ).reshape(-1)

    if probabilities.size == 0:
        raise ValueError("Probabilities cannot be empty.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    epsilon = 1e-7
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    return calibrator.predict_proba(logits)[:, 1]


def evaluate_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calculate calibration metrics and reliability-curve values."""
    probabilities, labels = _validate_probabilities(
        probabilities, labels
    )

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    fraction_positive, mean_predicted = calibration_curve(
        labels,
        probabilities,
        n_bins=n_bins,
        strategy="uniform",
    )

    return {
        "brier_score": float(
            brier_score_loss(labels, probabilities)
        ),
        "log_loss": float(
            log_loss(labels, probabilities, labels=[0, 1])
        ),
        "fraction_positive": fraction_positive.tolist(),
        "mean_predicted_probability": mean_predicted.tolist(),
        "n_bins": n_bins,
        "n_samples": int(labels.size),
    }