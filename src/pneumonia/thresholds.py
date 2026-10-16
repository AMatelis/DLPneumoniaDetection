
"""Classification threshold optimization for pneumonia detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pneumonia.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_THRESHOLDS = 100


def _validate_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize binary labels and prediction scores."""
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if probabilities.size == 0:
        raise ValueError("probabilities cannot be empty.")

    if probabilities.size != labels.size:
        raise ValueError(
            "probabilities and labels must have the same number of elements."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be between 0 and 1.")

    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError("labels must contain only binary values 0 and 1.")

    labels = labels.astype(int)

    if not np.any(labels == 0) or not np.any(labels == 1):
        logger.warning(
            "The labels contain only one class. Some metrics, including "
            "ROC AUC, cannot be calculated."
        )

    return probabilities, labels


def calculate_threshold_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """
    Calculate binary classification metrics at a chosen threshold.

    Label convention:
        0 = NORMAL
        1 = PNEUMONIA

    The positive class is PNEUMONIA.
    """
    probabilities, labels = _validate_inputs(probabilities, labels)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = float(recall_score(
        labels, predictions, zero_division=0
    ))

    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "youden_j": float(sensitivity + specificity - 1.0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(labels, probabilities)
        )
    else:
        metrics["roc_auc"] = None

    return metrics


def optimize_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metric: str = "youden_j",
    num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
    min_sensitivity: float | None = None,
    min_specificity: float | None = None,
) -> dict[str, Any]:
    """
    Select a classification threshold based on validation predictions.

    Args:
        probabilities: Model sigmoid scores for the positive class.
        labels: Binary ground-truth labels (0=normal, 1=pneumonia).
        metric: Metric to maximize. Supported: youden_j, f1, accuracy,
            balanced_accuracy, sensitivity, specificity, and precision.
        num_thresholds: Number of candidate thresholds to evaluate.
        min_sensitivity: Optional minimum acceptable sensitivity.
        min_specificity: Optional minimum acceptable specificity.

    Returns:
        The selected threshold, its metrics, and all candidate results.

    Note:
        Choose thresholds using a validation or dedicated calibration set,
        not the held-out test set.
    """
    probabilities, labels = _validate_inputs(probabilities, labels)

    allowed_metrics = {
        "youden_j",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
    }

    if metric not in allowed_metrics:
        raise ValueError(
            f"Unsupported metric {metric!r}. Choose from "
            f"{sorted(allowed_metrics)}."
        )

    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2.")

    for name, value in (
        ("min_sensitivity", min_sensitivity),
        ("min_specificity", min_specificity),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    # Include the conventional 0.5 cutoff in the candidate set.
    candidates = np.unique(
        np.append(
            np.linspace(0.01, 0.99, num_thresholds),
            DEFAULT_THRESHOLD,
        )
    )

    candidate_results: list[dict[str, Any]] = []

    for threshold in candidates:
        result = calculate_threshold_metrics(
            probabilities,
            labels,
            threshold=float(threshold),
        )

        if (
            min_sensitivity is not None
            and result["sensitivity"] < min_sensitivity
        ):
            continue

        if (
            min_specificity is not None
            and result["specificity"] < min_specificity
        ):
            continue

        candidate_results.append(result)

    if not candidate_results:
        raise ValueError(
            "No threshold satisfied the requested sensitivity and "
            "specificity constraints. Try relaxing them."
        )

    # Resolve ties by preferring the threshold closest to 0.5.
    best_result = max(
        candidate_results,
        key=lambda result: (
            result[metric],
            -abs(result["threshold"] - DEFAULT_THRESHOLD),
        ),
    )

    return {
        "optimization_metric": metric,
        "best_threshold": best_result["threshold"],
        "best_metrics": best_result,
        "num_candidates_evaluated": len(candidate_results),
        "candidate_results": candidate_results,
    }


def save_threshold_results(
    results: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Save threshold optimization results to JSON."""
    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "threshold_optimization.json"
        )
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)

    logger.info("Saved threshold optimization results to %s", output_path)
    return output_path


def optimize_and_save_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path | None = None,
    metric: str = "youden_j",
    num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
    min_sensitivity: float | None = None,
    min_specificity: float | None = None,
) -> dict[str, Any]:
    """Optimize a threshold and save the results."""
    results = optimize_threshold(
        probabilities=probabilities,
        labels=labels,
        metric=metric,
        num_thresholds=num_thresholds,
        min_sensitivity=min_sensitivity,
        min_specificity=min_specificity,
    )

    saved_path = save_threshold_results(results, output_path)
    results["results_path"] = str(saved_path)

    return results