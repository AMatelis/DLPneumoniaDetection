
"""Publication-oriented visualizations for pneumonia detection experiments.

Includes:
- Publication plotting style and figure saving
- Training curves
- ROC curves with bootstrap confidence intervals
- Uncertainty versus prediction error
- Grad-CAM image grids
- Dataset-shift performance comparisons
- Calibration curves
- Evaluation results summaries
- A function to generate figures from saved experiment outputs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Publication style and figure saving
# ---------------------------------------------------------------------

def setup_publication_style() -> None:
    """Apply consistent styling to research figures."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.15,
    )

    plt.rcParams.update(
        {
            "figure.figsize": (7, 5),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(
    figure: plt.Figure,
    save_path: str | Path,
    dpi: int = 300,
    formats: Sequence[str] | None = None,
) -> list[Path]:
    """Save a figure in one or more formats and close it."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = [save_path.suffix.lstrip(".") or "png"]

    saved_paths: list[Path] = []

    try:
        for file_format in formats:
            file_format = file_format.lower().lstrip(".")
            output_path = save_path.with_suffix(f".{file_format}")

            figure.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                format=file_format,
            )
            saved_paths.append(output_path)
            logger.info("Saved figure: %s", output_path)
    finally:
        plt.close(figure)

    return saved_paths


def _ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return a figure output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: str | Path) -> Any:
    """Load JSON data from a file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _validate_binary_inputs(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary labels and positive-class probabilities."""
    labels = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(y_prob, dtype=float).reshape(-1)

    if labels.size == 0:
        raise ValueError("Labels cannot be empty.")

    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal lengths.")

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    return labels.astype(int), probabilities


# ---------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------

def plot_training_curves(
    history: Any,
    output_dir: str | Path = "outputs/figures",
    prefix: str = "training",
    dpi: int = 300,
) -> list[Path]:
    """Plot accuracy, AUC, and loss curves from Keras training history."""
    history_dict = (
        history.history
        if hasattr(history, "history")
        else history
    )

    if not isinstance(history_dict, dict):
        raise TypeError("history must be a Keras History or dictionary.")

    output_dir = _ensure_output_dir(output_dir)
    saved: list[Path] = []

    plot_groups = [
        ("accuracy", "Accuracy", "accuracy"),
        ("auc", "AUC", "auc"),
        ("loss", "Loss", "loss"),
    ]

    setup_publication_style()

    for metric_key, title, ylabel in plot_groups:
        training_key = metric_key
        validation_key = f"val_{metric_key}"

        if training_key not in history_dict:
            continue

        train_values = np.asarray(history_dict[training_key], dtype=float)
        val_values = history_dict.get(validation_key)

        fig, ax = plt.subplots()

        ax.plot(
            np.arange(1, len(train_values) + 1),
            train_values,
            label=f"Training {title}",
        )

        if val_values is not None:
            val_values = np.asarray(val_values, dtype=float)
            ax.plot(
                np.arange(1, len(val_values) + 1),
                val_values,
                label=f"Validation {title}",
            )

        ax.set_title(f"Training and Validation {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.25)

        paths = save_figure(
            fig,
            output_dir / f"{prefix}_{metric_key}.png",
            dpi=dpi,
            formats=["png", "pdf"],
        )
        saved.extend(paths)

    return saved


# ---------------------------------------------------------------------
# ROC curves and bootstrap confidence intervals
# ---------------------------------------------------------------------

def plot_roc_with_ci(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    save_path: str | Path = "outputs/figures/roc_curve.png",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    dpi: int = 300,
) -> dict[str, Any]:
    """
    Plot a ROC curve and estimate an AUC confidence interval by bootstrap.

    Bootstrap resampling is performed at the example level. Resamples
    containing only one class are skipped.
    """
    labels, probabilities = _validate_binary_inputs(y_true, y_prob)

    if len(np.unique(labels)) != 2:
        raise ValueError("ROC analysis requires both classes.")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    fpr, tpr, _ = roc_curve(labels, probabilities)
    observed_auc = float(roc_auc_score(labels, probabilities))

    rng = np.random.default_rng(seed)
    bootstrap_aucs: list[float] = []
    n_examples = len(labels)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_examples, size=n_examples)
        sampled_labels = labels[indices]

        if len(np.unique(sampled_labels)) < 2:
            continue

        sampled_auc = roc_auc_score(
            sampled_labels,
            probabilities[indices],
        )
        bootstrap_aucs.append(float(sampled_auc))

    alpha = 1.0 - confidence_level

    if bootstrap_aucs:
        lower, upper = np.quantile(
            bootstrap_aucs,
            [alpha / 2, 1 - alpha / 2],
        )
        ci = [float(lower), float(upper)]
    else:
        ci = [None, None]
        logger.warning(
            "Could not calculate a bootstrap AUC interval."
        )

    setup_publication_style()
    fig, ax = plt.subplots()

    label = f"AUC = {observed_auc:.3f}"
    if ci[0] is not None:
        label += f" ({confidence_level:.0%} CI {ci[0]:.3f}-{ci[1]:.3f})"

    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set(
        title="Receiver Operating Characteristic",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate (Sensitivity)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")

    save_figure(fig, save_path, dpi=dpi, formats=["png", "pdf"])

    return {
        "auc": observed_auc,
        "confidence_level": confidence_level,
        "auc_confidence_interval": ci,
        "bootstrap_samples_requested": n_bootstrap,
        "bootstrap_samples_used": len(bootstrap_aucs),
    }


# ---------------------------------------------------------------------
# Uncertainty versus prediction error
# ---------------------------------------------------------------------

def plot_uncertainty_vs_error(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    uncertainty: Sequence[float] | np.ndarray,
    save_path: str | Path = "outputs/figures/uncertainty_vs_error.png",
    threshold: float = 0.5,
    dpi: int = 300,
) -> dict[str, float]:
    """
    Compare model uncertainty for correct and incorrect predictions.

    Uncertainty can be predictive entropy, predictive variance, or
    mutual information. The caller should specify which measure is used.
    """
    labels, probabilities = _validate_binary_inputs(y_true, y_prob)
    uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)

    if len(uncertainty) != len(labels):
        raise ValueError("uncertainty and labels must have equal lengths.")

    if not np.all(np.isfinite(uncertainty)):
        raise ValueError("uncertainty must contain only finite values.")

    predicted = (probabilities >= threshold).astype(int)
    incorrect = predicted != labels

    dataframe = pd.DataFrame(
        {
            "Prediction result": np.where(
                incorrect,
                "Incorrect",
                "Correct",
            ),
            "Uncertainty": uncertainty,
        }
    )

    setup_publication_style()
    fig, ax = plt.subplots()

    sns.boxplot(
        data=dataframe,
        x="Prediction result",
        y="Uncertainty",
        ax=ax,
    )

    sns.stripplot(
        data=dataframe,
        x="Prediction result",
        y="Uncertainty",
        ax=ax,
        alpha=0.35,
        size=3,
    )

    ax.set_title("Uncertainty by Prediction Correctness")
    save_figure(fig, save_path, dpi=dpi, formats=["png", "pdf"])

    correct_values = uncertainty[~incorrect]
    incorrect_values = uncertainty[incorrect]

    return {
        "n_examples": int(len(labels)),
        "n_correct": int(np.sum(~incorrect)),
        "n_incorrect": int(np.sum(incorrect)),
        "mean_uncertainty_correct": (
            float(np.mean(correct_values))
            if len(correct_values)
            else float("nan")
        ),
        "mean_uncertainty_incorrect": (
            float(np.mean(incorrect_values))
            if len(incorrect_values)
            else float("nan")
        ),
    }


# ---------------------------------------------------------------------
# Grad-CAM grids
# ---------------------------------------------------------------------

def plot_gradcam_grid(
    image_paths: Sequence[str | Path],
    heatmap_paths: Sequence[str | Path],
    labels: Sequence[str] | None = None,
    save_path: str | Path = "outputs/figures/gradcam_grid.png",
    max_images: int = 8,
    dpi: int = 300,
) -> Path:
    """
    Display saved original images alongside saved Grad-CAM heatmaps.

    This function expects image and heatmap files to already exist.
    It does not compute Grad-CAM itself.
    """
    if len(image_paths) != len(heatmap_paths):
        raise ValueError("image_paths and heatmap_paths must match in length.")

    if not image_paths:
        raise ValueError("At least one image is required.")

    if max_images < 1:
        raise ValueError("max_images must be at least 1.")

    count = min(len(image_paths), max_images)

    if labels is not None and len(labels) != len(image_paths):
        raise ValueError("labels must match the image count.")

    setup_publication_style()
    fig, axes = plt.subplots(
        count,
        2,
        figsize=(8, max(3, count * 2.5)),
        squeeze=False,
    )

    for index in range(count):
        original = plt.imread(image_paths[index])
        heatmap = plt.imread(heatmap_paths[index])

        axes[index, 0].imshow(original, cmap="gray" if original.ndim == 2 else None)
        axes[index, 0].set_title(
            f"Original: {labels[index]}"
            if labels is not None
            else "Original"
        )

        axes[index, 1].imshow(heatmap)
        axes[index, 1].set_title("Grad-CAM")

        axes[index, 0].axis("off")
        axes[index, 1].axis("off")

    fig.suptitle("Grad-CAM Visual Explanations")
    fig.tight_layout()

    paths = save_figure(
        fig,
        save_path,
        dpi=dpi,
        formats=["png", "pdf"],
    )
    return paths[0]


# ---------------------------------------------------------------------
# Dataset shift performance
# ---------------------------------------------------------------------

def plot_dataset_shift_performance(
    results: dict[str, Any] | pd.DataFrame,
    save_path: str | Path = "outputs/figures/dataset_shift.png",
    metric: str = "auc",
    dpi: int = 300,
) -> Path:
    """
    Plot performance across dataset-shift conditions.

    Supported input:
      - DataFrame with columns 'condition' and the requested metric.
      - Dictionary mapping condition names to dictionaries containing
        the requested metric.

    Pass measured results, not hardcoded or illustrative performance.
    """
    if isinstance(results, pd.DataFrame):
        dataframe = results.copy()
    elif isinstance(results, dict):
        rows = []
        for condition, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            if metric in metrics:
                rows.append(
                    {
                        "condition": str(condition),
                        metric: metrics[metric],
                    }
                )
        dataframe = pd.DataFrame(rows)
    else:
        raise TypeError("results must be a dictionary or DataFrame.")

    if dataframe.empty:
        raise ValueError(f"No results contain the metric '{metric}'.")

    if "condition" not in dataframe.columns:
        raise ValueError("Results must include a 'condition' column.")

    if metric not in dataframe.columns:
        raise ValueError(f"Results must include a '{metric}' column.")

    dataframe[metric] = pd.to_numeric(
        dataframe[metric],
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=[metric])

    if dataframe.empty:
        raise ValueError(f"No numeric values found for '{metric}'.")

    setup_publication_style()
    fig, ax = plt.subplots(
        figsize=(max(7, len(dataframe) * 1.1), 5)
    )

    sns.barplot(
        data=dataframe,
        x="condition",
        y=metric,
        hue="condition",
        legend=False,
        ax=ax,
    )

    ax.set_title(f"Dataset Shift: {metric.upper()} by Condition")
    ax.set_xlabel("Evaluation condition")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", labelrotation=25)

    if metric.lower() in {"auc", "accuracy", "balanced_accuracy", "f1"}:
        ax.set_ylim(0, 1)

    paths = save_figure(
        fig,
        save_path,
        dpi=dpi,
        formats=["png", "pdf"],
    )
    return paths[0]


# ---------------------------------------------------------------------
# Calibration curves
# ---------------------------------------------------------------------

def plot_calibration_curves(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    save_path: str | Path = "outputs/figures/calibration.png",
    n_bins: int = 10,
    strategy: str = "uniform",
    dpi: int = 300,
) -> dict[str, float]:
    """
    Plot a reliability diagram and calculate Brier score.

    Calibration should be evaluated on a separate validation or test
    partition that was not used to fit the calibrator.
    """
    labels, probabilities = _validate_binary_inputs(y_true, y_prob)

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    fraction_positive, mean_predicted = calibration_curve(
        labels,
        probabilities,
        n_bins=n_bins,
        strategy=strategy,
    )

    brier_score = float(np.mean((probabilities - labels) ** 2))

    setup_publication_style()
    fig, ax = plt.subplots()

    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Model",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.set(
        title="Calibration Curve",
        xlabel="Mean Predicted Probability",
        ylabel="Observed Positive Fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()

    save_figure(fig, save_path, dpi=dpi, formats=["png", "pdf"])

    return {
        "brier_score": brier_score,
        "n_bins": int(n_bins),
        "n_examples": int(len(labels)),
    }


# ---------------------------------------------------------------------
# Evaluation results summary
# ---------------------------------------------------------------------

def plot_results_summary(
    results: dict[str, Any] | str | Path,
    save_path: str | Path = "outputs/figures/results_summary.png",
    dpi: int = 300,
) -> Path:
    """
    Plot a compact summary of scalar evaluation metrics.

    `results` can be an evaluation dictionary or a path to its JSON file.
    Nested classification reports and confusion matrices are not treated
    as scalar metrics.
    """
    if isinstance(results, (str, Path)):
        results = _load_json(results)

    if not isinstance(results, dict):
        raise TypeError("results must be a dictionary or JSON file path.")

    candidates = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "auc",
        "brier_score",
    )

    metrics: dict[str, float] = {}

    for key in candidates:
        value = results.get(key)

        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(value):
                metrics[key] = float(value)

    if not metrics:
        raise ValueError(
            "No recognized scalar metrics found in evaluation results."
        )

    dataframe = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values()),
        }
    )

    setup_publication_style()
    fig, ax = plt.subplots(
        figsize=(8, max(4, len(metrics) * 0.45))
    )

    sns.barplot(
        data=dataframe,
        x="Value",
        y="Metric",
        hue="Metric",
        legend=False,
        ax=ax,
    )

    ax.set_title("Evaluation Metrics Summary")
    ax.set_xlabel("Score")
    ax.set_ylabel("")

    paths = save_figure(
        fig,
        save_path,
        dpi=dpi,
        formats=["png", "pdf"],
    )
    return paths[0]


# ---------------------------------------------------------------------
# Generate figures from an experiment directory
# ---------------------------------------------------------------------

def generate_all_figures(
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = 300,
) -> list[Path]:
    """
    Generate available figures from saved experiment artifacts.

    Looks for:
      - outputs/training_history.json
      - outputs/evaluation/evaluation_results.json
      - outputs/evaluation/predictions.csv
      - outputs/dataset_shift_results.json

    Only artifacts that exist and contain the required fields are used.
    """
    experiment_dir = Path(experiment_dir)

    if output_dir is None:
        output_dir = experiment_dir / "figures"

    output_dir = _ensure_output_dir(output_dir)
    generated: list[Path] = []

    history_path = experiment_dir / "outputs" / "training_history.json"
    if history_path.exists():
        history = _load_json(history_path)
        generated.extend(
            plot_training_curves(
                history,
                output_dir=output_dir,
                dpi=dpi,
            )
        )

    evaluation_path = (
        experiment_dir
        / "outputs"
        / "evaluation"
        / "evaluation_results.json"
    )

    predictions_path = (
        experiment_dir
        / "outputs"
        / "evaluation"
        / "predictions.csv"
    )

    if evaluation_path.exists():
        try:
            summary_path = output_dir / "results_summary.png"
            generated.append(
                plot_results_summary(
                    evaluation_path,
                    save_path=summary_path,
                    dpi=dpi,
                )
            )
        except (ValueError, TypeError, KeyError) as error:
            logger.warning("Could not plot evaluation summary: %s", error)

    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)

        required_columns = {"true_label", "probability"}
        if required_columns.issubset(predictions.columns):
            try:
                generated.extend(
                    plot_roc_with_ci(
                        predictions["true_label"],
                        predictions["probability"],
                        save_path=output_dir / "roc_curve.png",
                        dpi=dpi,
                    ).get("saved_paths", [])
                )
            except ValueError as error:
                logger.warning("Could not plot ROC curve: %s", error)

            try:
                calibration_path = output_dir / "calibration.png"
                plot_calibration_curves(
                    predictions["true_label"],
                    predictions["probability"],
                    save_path=calibration_path,
                    dpi=dpi,
                )
                generated.extend(
                    [
                        calibration_path,
                        calibration_path.with_suffix(".pdf"),
                    ]
                )
            except ValueError as error:
                logger.warning(
                    "Could not plot calibration curve: %s",
                    error,
                )

            uncertainty_columns = [
                column
                for column in (
                    "predictive_entropy",
                    "predictive_variance",
                    "mutual_information",
                )
                if column in predictions.columns
            ]

            if uncertainty_columns:
                predicted = (
                    predictions["probability"].to_numpy() >= 0.5
                ).astype(int)
                labels = predictions["true_label"].to_numpy().astype(int)
                errors = predicted != labels

                for column in uncertainty_columns:
                    try:
                        generated.append(
                            plot_uncertainty_vs_error(
                                labels,
                                predictions["probability"],
                                predictions[column],
                                save_path=(
                                    output_dir
                                    / f"uncertainty_vs_error_{column}.png"
                                ),
                                dpi=dpi,
                            ).get("saved_path", output_dir / f"uncertainty_vs_error_{column}.png")
                        )
                    except ValueError as error:
                        logger.warning(
                            "Could not plot %s: %s",
                            column,
                            error,
                        )

    shift_path = experiment_dir / "outputs" / "dataset_shift_results.json"
    if shift_path.exists():
        shift_results = _load_json(shift_path)

        # The caller's result schema may differ by experiment. Plot only
        # when entries contain directly comparable AUC values.
        try:
            plot_path = output_dir / "dataset_shift_auc.png"
            generated.append(
                plot_dataset_shift_performance(
                    shift_results,
                    save_path=plot_path,
                    metric="auc",
                    dpi=dpi,
                )
            )
        except (ValueError, TypeError, KeyError) as error:
            logger.warning(
                "Could not plot dataset-shift results: %s",
                error,
            )

    logger.info(
        "Generated %d figure artifact(s) in %s",
        len(generated),
        output_dir,
    )
    return generated