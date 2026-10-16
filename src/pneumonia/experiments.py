
"""Experiment orchestration for reproducible pneumonia detection research."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from pneumonia.config import ExperimentConfig, PROJECT_ROOT
from pneumonia.data import create_data_generators
from pneumonia.dataset_shift import run_dataset_shift_experiment
from pneumonia.evaluation import evaluate_model
from pneumonia.model import build_model

logger = logging.getLogger(__name__)

SUPPORTED_ABLATIONS = {
    "use_mc_dropout",
    "optimize_thresholds",
    "calibrate",
    "analyze_gradcam",
    "use_class_weights",
}


def set_random_seeds(seed: int = 42) -> None:
    """Set available random seeds for more reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_experiment_directory(
    experiment_name: str | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Create a unique experiment directory with standard subfolders."""
    if base_dir is None:
        base_path = PROJECT_ROOT / "outputs" / "experiments"
    else:
        base_path = Path(base_dir)
        if not base_path.is_absolute():
            base_path = PROJECT_ROOT / base_path

    base_path.mkdir(parents=True, exist_ok=True)

    if experiment_name:
        # Prevent path traversal and accidental nested output directories.
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_name).strip("._")
        if not safe_name:
            raise ValueError("experiment_name contains no usable characters.")
    else:
        safe_name = "pneumonia_experiment"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_path / f"{safe_name}_{timestamp}"
    suffix = 1

    while experiment_dir.exists():
        experiment_dir = base_path / f"{safe_name}_{timestamp}_{suffix}"
        suffix += 1

    for folder in ("models", "outputs", "plots", "logs"):
        (experiment_dir / folder).mkdir(parents=True, exist_ok=True)

    return experiment_dir


def _git_metadata() -> dict[str, Any]:
    """Collect Git metadata when the project is inside a Git repository."""
    metadata: dict[str, Any] = {
        "git_available": False,
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

        metadata.update(
            {
                "git_available": True,
                "git_commit": commit,
                "git_branch": branch or None,
                "git_dirty": bool(status),
            }
        )
    except (OSError, subprocess.SubprocessError):
        logger.info("Git metadata unavailable; continuing without it.")

    return metadata


def save_experiment_metadata(
    experiment_dir: str | Path,
    config: ExperimentConfig,
    seed: int = 42,
) -> Path:
    """Save configuration and environment metadata for an experiment."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_path = experiment_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, indent=2, default=str)

    metadata = {
        "experiment_name": config.get_value(
            "experiment", "name", "PneumoniaDetection_Baseline"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "config_file": str(config_path),
        **_git_metadata(),
    }

    metadata_path = experiment_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    return metadata_path


def _apply_ablations(
    config: ExperimentConfig,
    ablations: dict[str, bool] | None,
) -> None:
    """Validate and apply ablation overrides to the experiment config."""
    if not ablations:
        return

    for name, enabled in ablations.items():
        if name not in SUPPORTED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation '{name}'. Supported values: "
                f"{sorted(SUPPORTED_ABLATIONS)}"
            )

        if not isinstance(enabled, bool):
            raise TypeError(f"Ablation '{name}' must be True or False.")

        config.config.setdefault("ablations", {})[name] = enabled
        logger.info("Ablation override: %s = %s", name, enabled)


def _train_experiment(
    config: ExperimentConfig,
    experiment_dir: Path,
    num_epochs: int | None = None,
) -> dict[str, Any]:
    """Train a model, save artifacts, and run requested evaluation steps."""
    settings = config.to_dict()
    training_config = settings.get("training", {})
    ablations = settings.get("ablations", {})

    epochs = int(
        num_epochs
        if num_epochs is not None
        else training_config.get("num_epochs", 50)
    )

    if epochs < 1:
        raise ValueError("num_epochs must be at least 1.")

    seed = int(settings.get("reproducibility", {}).get("seed", 42))
    set_random_seeds(seed)

    logger.info("Creating training and validation generators.")
    train_generator, val_generator = create_data_generators(config=config)

    if train_generator.samples == 0 or val_generator.samples == 0:
        raise ValueError("Training and validation datasets must not be empty.")

    model = build_model()

    class_weight = None
    if ablations.get("use_class_weights", True):
        classes = np.unique(train_generator.classes)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=train_generator.classes,
        )
        class_weight = {
            int(class_id): float(weight)
            for class_id, weight in zip(classes, weights)
        }

    best_model_path = experiment_dir / "models" / "best_model.keras"

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=int(training_config.get("early_stopping_patience", 10)),
            restore_best_weights=True,
            mode="max",
        ),
        ModelCheckpoint(
            str(best_model_path),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.1,
            patience=int(training_config.get("reduce_lr_patience", 5)),
            mode="max",
        ),
    ]

    logger.info("Training for up to %d epochs.", epochs)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    final_model_path = experiment_dir / "models" / "final_model.keras"
    model.save(final_model_path)

    history_path = experiment_dir / "outputs" / "training_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)

    # Evaluate the best checkpoint when available, rather than assuming
    # the final training epoch was the best one.
    if best_model_path.exists():
        best_model = tf.keras.models.load_model(best_model_path)
    else:
        logger.warning(
            "Best checkpoint was not created; evaluating the current model."
        )
        best_model = model

    evaluation_results = None
    if settings.get("output", {}).get("run_evaluation", True):
        evaluation_results = evaluate_model(
            best_model,
            val_generator,
            history,
            use_mc_dropout=bool(ablations.get("use_mc_dropout", False)),
            optimize_thresholds=bool(
                ablations.get("optimize_thresholds", False)
            ),
            calibrate=bool(ablations.get("calibrate", False)),
            exp_dir=experiment_dir / "outputs",
        )

    gradcam_results = None
    if ablations.get("analyze_gradcam", False):
        try:
            from pneumonia.explainability import analyze_gradcam_quantitative

            gradcam_results = analyze_gradcam_quantitative(
                best_model,
                val_generator,
                num_samples=50,
                exp_dir=str(experiment_dir),
            )
        except ImportError as exc:
            logger.warning(
                "Grad-CAM was requested but its module is unavailable: %s",
                exc,
            )

    return {
        "status": "completed",
        "experiment_dir": str(experiment_dir),
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "training_history_path": str(history_path),
        "evaluation": evaluation_results,
        "gradcam": gradcam_results,
    }


def run_experiment(
    config_path: str | Path | None = None,
    experiment_name: str | None = None,
    ablations: dict[str, bool] | None = None,
    mode: str = "train",
    num_epochs: int | None = None,
) -> dict[str, Any]:
    """
    Run a configured training or dataset-shift experiment.

    Args:
        config_path: YAML configuration path.
        experiment_name: Optional name for the output directory.
        ablations: Optional overrides for supported ablation settings.
        mode: 'train' or 'dataset_shift'.
        num_epochs: Optional training epoch override.

    Returns:
        Dictionary containing experiment status, paths, and results.
    """
    if mode not in {"train", "dataset_shift"}:
        raise ValueError(
            "mode must be 'train' or 'dataset_shift'. "
            "Use the evaluation script to evaluate an existing model."
        )

    config = ExperimentConfig(config_path)
    _apply_ablations(config, ablations)

    seed = int(config.get("reproducibility.seed", 42))
    set_random_seeds(seed)

    experiment_name = experiment_name or config.get(
        "experiment.name", "PneumoniaDetection_Baseline"
    )

    experiment_dir = create_experiment_directory(experiment_name)

    save_experiment_metadata(
        experiment_dir,
        config,
        seed=seed,
    )

    logger.info("Experiment directory: %s", experiment_dir)
    logger.info("Mode: %s", mode)

    try:
        if mode == "train":
            results = _train_experiment(
                config,
                experiment_dir,
                num_epochs=num_epochs,
            )
        else:
            shift_config = config.to_dict()
            shift_settings = shift_config.get("dataset_shift", {})

            source_dataset = shift_settings.get("source_dataset", "A")
            target_dataset = shift_settings.get("target_dataset", "B")
            fine_tune = bool(shift_settings.get("fine_tune", False))
            fine_tune_ratio = shift_settings.get("split_ratio", 0.1)

            shift_results = run_dataset_shift_experiment(
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                fine_tune=fine_tune,
                config=config,
                output_dir=experiment_dir / "dataset_shift",
                fine_tune_ratio=fine_tune_ratio,
                epochs=num_epochs or int(
                    config.get("training.num_epochs", 50)
                ),
            )

            results = {
                "status": "completed",
                "experiment_dir": str(experiment_dir),
                "dataset_shift": shift_results,
            }

        results_path = experiment_dir / "experiment_results.json"
        with results_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, default=str, allow_nan=False)

        return results

    except Exception as exc:
        logger.exception("Experiment failed.")

        failure_results = {
            "status": "failed",
            "experiment_dir": str(experiment_dir),
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

        failure_path = experiment_dir / "experiment_failure.json"
        with failure_path.open("w", encoding="utf-8") as file:
            json.dump(failure_results, file, indent=2)

        # Preserve the original exception so callers and scripts know
        # that the experiment did not complete successfully.
        raise