
"""Model evaluation, metrics, plots, and optional calibration for pneumonia detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import balanced_accuracy_score

from pneumonia.calibration import (
    apply_probability_calibrator,
    evaluate_calibration,
    fit_probability_calibrator,
)
from pneumonia.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _output_directory(exp_dir: str | Path | None) -> Path:
    """Create and return the evaluation output directory."""
    if exp_dir is None:
        directory = PROJECT_ROOT / "outputs" / "evaluation"
    else:
        directory = Path(exp_dir)
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _get_labels(generator) -> np.ndarray:
    """Read integer labels from a Keras image generator."""
    labels = np.asarray(generator.classes, dtype=int).ravel()

    if labels.size == 0:
        raise ValueError("The evaluation generator contains no labels.")

    return labels


def _predict_probabilities(
    model,
    generator,
    use_mc_dropout: bool = False,
    num_mc_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Generate sigmoid probabilities, optionally using MC dropout."""
    if num_mc_samples < 1:
        raise ValueError("num_mc_samples must be at least 1.")

    generator.reset()

    if not use_mc_dropout:
        predictions = np.asarray(
            model.predict(generator, verbose=0)
        ).reshape(-1)

        return predictions, None, None

    logger.info("Running MC dropout with %d samples.", num_mc_samples)

    # Enable dropout at inference by calling the model with training=True.
    # This requires a model whose dropout layers are active under this call.
    probability_runs = []

    for sample_index in range(num_mc_samples):
        generator.reset()
        batch_predictions = []

        for batch_index in range(len(generator)):
            batch = generator[batch_index]
            images = batch[0]

            predictions = model(images, training=True)

            if hasattr(predictions, "numpy"):
                predictions = predictions.numpy()

            batch_predictions.extend(
                np.asarray(predictions).reshape(-1).tolist()
            )

        probability_runs.append(np.asarray(batch_predictions, dtype=float))

    runs = np.asarray(probability_runs)

    if runs.ndim != 2:
        raise ValueError(
            f"Unexpected MC prediction shape: {runs.shape}"
        )

    mean_probabilities = np.mean(runs, axis=0)
    prediction_variance = np.var(runs, axis=0)

    # Binary entropy, in nats, as a simple uncertainty measure.
    clipped = np.clip(mean_probabilities, 1e-7, 1.0 - 1e-7)
    entropy = -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )

    return mean_probabilities, entropy, prediction_variance


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute classification metrics at a selected decision threshold."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()

    if len(y_true) != len(probabilities):
        raise ValueError("Labels and predictions have different lengths.")

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain non-finite values.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Expected probabilities between 0 and 1.")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    predicted_labels = (probabilities >= threshold).astype(int)

    metrics = {
        "sample_count": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predicted_labels)
        ),
        "precision": float(
            precision_score(y_true, predicted_labels, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted_labels, zero_division=0)
        ),
        "specificity": float(
            recall_score(
                y_true,
                predicted_labels,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(y_true, predicted_labels, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predicted_labels, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predicted_labels,
            labels=[0, 1],
            target_names=["NORMAL", "PNEUMONIA"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["auc"] = None
        logger.warning(
            "ROC AUC is undefined because the labels contain only one class."
        )

    return metrics


def _optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_thresholds: int = 100,
) -> dict[str, Any]:
    """Choose the threshold with the best Youden J statistic."""
    if num_thresholds < 2:
        raise ValueError("num_thresholds must be at least 2.")

    if len(np.unique(y_true)) != 2:
        logger.warning(
            "Cannot optimize the threshold without both classes."
        )
        return {"threshold": 0.5, "youden_j": None}

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        labels = (probabilities >= threshold).astype(int)

        sensitivity = recall_score(
            y_true, labels, pos_label=1, zero_division=0
        )
        specificity = recall_score(
            y_true, labels, pos_label=0, zero_division=0
        )
        youden_j = sensitivity + specificity - 1.0

        if youden_j > best_score:
            best_score = float(youden_j)
            best_threshold = float(threshold)

    return {
        "threshold": best_threshold,
        "youden_j": best_score,
        "selection_method": "maximum Youden J on the supplied labels",
    }


def _save_training_curves(history, output_dir: Path) -> None:
    """Save training accuracy and AUC curves when history is available."""
    if history is None or not hasattr(history, "history"):
        return

    history_data = history.history
    available = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("auc", "val_auc", "AUC"),
        ("loss", "val_loss", "Loss"),
    ]

    for train_key, val_key, title in available:
        if train_key not in history_data:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history_data[train_key], label="Training")

        if val_key in history_data:
            ax.plot(history_data[val_key], label="Validation")

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"training_{train_key}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_evaluation_plots(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save confusion matrix and ROC curve when applicable."""
    predicted_labels = (
        probabilities >= metrics["threshold"]
    ).astype(int)

    matrix = confusion_matrix(y_true, predicted_labels, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    if len(np.unique(y_true)) == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, probabilities
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"AUC = {metrics['auc']:.4f}",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_dir / "roc_curve.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_calibration_plot(
    probabilities: np.ndarray,
    calibration_results: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot the reliability curve from calibration evaluation."""
    mean_predicted = calibration_results.get("mean_predicted_value", [])
    fraction_positive = calibration_results.get("fraction_of_positives", [])

    if not len(mean_predicted):
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Model",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "calibration_diagram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def evaluate_model(
    model,
    val_generator,
    history=None,
    use_mc_dropout: bool = False,
    optimize_thresholds: bool = False,
    calibrate: bool = False,
    exp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate a binary pneumonia classifier and save metrics and plots.

    Args:
        model: Trained Keras model with a sigmoid probability output.
        val_generator: Non-shuffled validation generator.
        history: Optional Keras training History object.
        use_mc_dropout: Estimate predictions using repeated stochastic
            forward passes. Requires dropout layers that remain active
            when the model is called with training=True.
        optimize_thresholds: Select a decision threshold using Youden J.
        calibrate: Fit a probability calibrator on these supplied labels.
            For scientifically valid results, supply a dedicated calibration
            set rather than the final test set.
        exp_dir: Output directory. Files are saved directly inside it.

    Returns:
        Dictionary of evaluation metrics and artifact paths.
    """
    output_dir = _output_directory(exp_dir)
    y_true = _get_labels(val_generator)

    predictions, uncertainties, variances = _predict_probabilities(
        model,
        val_generator,
        use_mc_dropout=use_mc_dropout,
        num_mc_samples=10,
    )

    if len(predictions) != len(y_true):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match "
            f"label count ({len(y_true)}). Check generator ordering and batches."
        )

    results: dict[str, Any] = {
        "evaluation_samples": int(len(y_true)),
        "mc_dropout_enabled": bool(use_mc_dropout),
    }

    threshold = 0.5

    if optimize_thresholds:
        threshold_results = _optimize_threshold(
            y_true,
            predictions,
            num_thresholds=100,
        )
        results["threshold_optimization"] = threshold_results
        threshold = threshold_results["threshold"]

    results["metrics"] = _classification_metrics(
        y_true,
        predictions,
        threshold=threshold,
    )

    # Calibration is deliberately reported as a separate analysis.
    # The calibrated probabilities are not silently substituted for the
    # primary, uncalibrated predictions.
    if calibrate:
        calibration_results = evaluate_calibration(
            predictions,
            y_true,
            n_bins=10,
        )

        calibrator = fit_probability_calibrator(
            predictions,
            y_true,
        )
        calibrated_probabilities = apply_probability_calibrator(
            calibrator,
            predictions,
        )

        results["calibration_before"] = calibration_results
        results["calibration_after"] = evaluate_calibration(
            calibrated_probabilities,
            y_true,
            n_bins=10,
        )

        _save_calibration_plot(
            predictions,
            results["calibration_before"],
            output_dir,
        )

        # Store the calibration parameters so the analysis can be reproduced.
        calibrator_path = output_dir / "probability_calibrator.json"
        calibrator_data = {
            "method": "logistic regression on probability logits",
            "classes": calibrator.classes_.tolist(),
            "coef": calibrator.coef_.tolist(),
            "intercept": calibrator.intercept_.tolist(),
        }

        with calibrator_path.open("w", encoding="utf-8") as file:
            json.dump(calibrator_data, file, indent=2)

    predictions_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": (
                predictions >= threshold
            ).astype(int),
            "probability": predictions,
        }
    )

    if uncertainties is not None:
        predictions_df["uncertainty_entropy"] = uncertainties
        predictions_df["uncertainty_variance"] = variances

    predictions_path = output_dir / "validation_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    _save_evaluation_plots(
        y_true,
        predictions,
        results["metrics"],
        output_dir,
    )
    _save_training_curves(history, output_dir)

    results["artifacts"] = {
        "predictions_csv": str(predictions_path),
        "output_directory": str(output_dir),
    }

    results_path = output_dir / "evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)

    logger.info("Evaluation metrics: %s", results["metrics"])
    logger.info("Evaluation results saved to %s", results_path)

    return results