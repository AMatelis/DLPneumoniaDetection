
"""Training pipeline for the ResNet50 pneumonia detection model."""

from __future__ import annotations

import json
import logging
import random
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
from pneumonia.model import build_model

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds to improve experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _load_config(
    config: ExperimentConfig | dict[str, Any] | str | Path | None = None,
) -> ExperimentConfig:
    """Return an ExperimentConfig from a supported input."""
    if config is None:
        return ExperimentConfig()

    if isinstance(config, ExperimentConfig):
        return config

    if isinstance(config, (str, Path)):
        return ExperimentConfig(config)

    if isinstance(config, dict):
        # Start with defaults so partial dictionaries remain usable.
        result = ExperimentConfig()
        _deep_merge(result.config, config)
        return result

    raise TypeError(
        "config must be an ExperimentConfig, dictionary, path, or None."
    )


def _deep_merge(
    target: dict[str, Any],
    updates: dict[str, Any],
) -> None:
    """Merge nested dictionaries without discarding unrelated defaults."""
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(target.get(key), dict)
        ):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _resolve_path(path: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _save_json(path: Path, data: Any) -> None:
    """Write JSON while converting common NumPy values."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.bool_):
            return bool(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=convert, allow_nan=False)


def calculate_class_weights(
    labels: np.ndarray,
) -> dict[int, float]:
    """
    Calculate balanced class weights from training labels only.

    The generator's class indices must match the model's label convention:
    NORMAL=0 and PNEUMONIA=1.
    """
    labels = np.asarray(labels).reshape(-1)

    if labels.size == 0:
        raise ValueError("Cannot calculate class weights from empty labels.")

    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    return {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }


def create_training_callbacks(
    output_dir: str | Path,
    config: ExperimentConfig | dict[str, Any] | None = None,
) -> list[tf.keras.callbacks.Callback]:
    """
    Create the project's training callbacks.

    Checkpoint selection follows the original project: highest validation AUC.
    """
    config_obj = _load_config(config)
    settings = config_obj.to_dict()
    training = settings.get("training", {})

    output_dir = _resolve_path(output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / "best_model.keras"

    early_stopping_patience = int(
        training.get("early_stopping_patience", 10)
    )
    reduce_lr_patience = int(
        training.get("reduce_lr_patience", 5)
    )

    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative.")

    if reduce_lr_patience < 0:
        raise ValueError("reduce_lr_patience cannot be negative.")

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.1,
            patience=reduce_lr_patience,
            mode="max",
            verbose=1,
        ),
    ]


def train(
    num_epochs: int | None = None,
    args: Any | None = None,
    config: ExperimentConfig | dict[str, Any] | str | Path | None = None,
    output_dir: str | Path | None = None,
    train_generator: Any | None = None,
    val_generator: Any | None = None,
    model: tf.keras.Model | None = None,
    run_evaluation: bool | None = None,
) -> dict[str, Any]:
    """
    Train the pneumonia classifier and optionally evaluate it.

    Args:
        num_epochs: Override the configured epoch count.
        args: Optional argparse Namespace for compatibility with the
            original script. Recognized fields include optimize_thresholds,
            calibrate, analyze_gradcam, and no_class_weights.
        config: ExperimentConfig, configuration dictionary, YAML path,
            or None to use the default configuration.
        output_dir: Directory for checkpoints, model, history, and outputs.
        train_generator: Optional pre-created training generator.
        val_generator: Optional pre-created validation generator.
        model: Optional compiled Keras model. If omitted, build_model()
            creates the model.
        run_evaluation: Whether to evaluate after training. Defaults to
            the output configuration setting, or True if not configured.

    Returns:
        Dictionary containing training status, artifact paths, history,
        and evaluation results when evaluation is enabled.
    """
    config_obj = _load_config(config)
    settings = config_obj.to_dict()

    training_config = settings.get("training", {})
    output_config = settings.get("output", {})
    ablations = settings.get("ablations", {})

    seed = int(settings.get("reproducibility", {}).get("seed", 42))
    set_random_seeds(seed)

    epochs = int(
        num_epochs
        if num_epochs is not None
        else training_config.get("num_epochs", 50)
    )
    if epochs < 1:
        raise ValueError("num_epochs must be at least 1.")

    if output_dir is None:
        configured_output = output_config.get(
            "base_dir", "outputs/training"
        )
        output_dir = _resolve_path(configured_output)
    else:
        output_dir = _resolve_path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("PNEUMONIA DETECTION MODEL TRAINING")
    logger.info("=" * 72)
    logger.info("Epochs: %d", epochs)
    logger.info("Output directory: %s", output_dir)

    # Save the exact configuration used for this run.
    config_path = output_dir / "config.json"
    _save_json(config_path, settings)

    if train_generator is None or val_generator is None:
        logger.info("Creating data generators.")

        generated_train, generated_val = create_data_generators(
            config=config_obj
        )

        if train_generator is None:
            train_generator = generated_train
        if val_generator is None:
            val_generator = generated_val

    if getattr(train_generator, "samples", 0) == 0:
        raise ValueError("The training generator contains no images.")

    if getattr(val_generator, "samples", 0) == 0:
        raise ValueError("The validation generator contains no images.")

    logger.info(
        "Training images: %d | Validation images: %d",
        train_generator.samples,
        val_generator.samples,
    )

    if model is None:
        logger.info("Building ResNet50 classifier.")
        model = build_model(config=config_obj)

    use_class_weights = bool(
        ablations.get("use_class_weights", True)
    )

    # Preserve the original project’s balanced class-weight behavior.
    # Only training labels are used to calculate the weights.
    class_weights = None

    if args is not None and getattr(args, "no_class_weights", False):
        use_class_weights = False

    if use_class_weights:
        class_weights = calculate_class_weights(
            train_generator.classes
        )
        logger.info("Using class weights: %s", class_weights)

    callbacks = create_training_callbacks(
        output_dir=output_dir,
        config=config_obj,
    )

    started_at = datetime.now(timezone.utc)

    logger.info("Starting model.fit().")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    finished_at = datetime.now(timezone.utc)

    # Save the final model separately from the best checkpoint.
    final_model_path = output_dir / "models" / "final_model.keras"
    model.save(final_model_path)

    best_model_path = output_dir / "models" / "best_model.keras"

    # EarlyStopping restores best weights, but prefer the checkpoint
    # explicitly when loading for downstream evaluation.
    if best_model_path.exists():
        evaluation_model = tf.keras.models.load_model(best_model_path)
    else:
        logger.warning(
            "No best checkpoint was saved. Using the trained model."
        )
        evaluation_model = model

    history_path = output_dir / "outputs" / "training_history.json"
    _save_json(history_path, history.history)

    # Record final and best observed values for each logged metric.
    history_summary: dict[str, Any] = {}

    for metric_name, values in history.history.items():
        if not values:
            continue

        numeric_values = [
            float(value)
            for value in values
            if np.isfinite(value)
        ]

        if not numeric_values:
            continue

        history_summary[metric_name] = {
            "last": numeric_values[-1],
            "best": (
                max(numeric_values)
                if "auc" in metric_name or "accuracy" in metric_name
                else min(numeric_values)
            ),
            "epochs_recorded": len(values),
        }

    _save_json(
        output_dir / "outputs" / "training_summary.json",
        {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (
                finished_at - started_at
            ).total_seconds(),
            "epochs_requested": epochs,
            "epochs_completed": len(history.epoch),
            "training_samples": int(train_generator.samples),
            "validation_samples": int(val_generator.samples),
            "class_weights": class_weights,
            "metrics": history_summary,
            "best_model_path": str(best_model_path),
            "final_model_path": str(final_model_path),
        },
    )

    if run_evaluation is None:
        run_evaluation = bool(
            output_config.get("run_evaluation", True)
        )

    evaluation_results = None

    if run_evaluation:
        logger.info("Running model evaluation.")

        # Imported here to avoid unnecessary imports when only training
        # is requested and to reduce circular-import risk.
        from pneumonia.evaluation import evaluate_model

        optimize_thresholds = bool(
            getattr(
                args,
                "optimize_thresholds",
                ablations.get("optimize_thresholds", False),
            )
        )
        calibrate = bool(
            getattr(
                args,
                "calibrate",
                ablations.get("calibrate", False),
            )
        )
        use_mc_dropout = bool(
            getattr(
                args,
                "use_mc_dropout",
                ablations.get("use_mc_dropout", False),
            )
        )

        evaluation_results = evaluate_model(
            evaluation_model,
            val_generator,
            history,
            use_mc_dropout=use_mc_dropout,
            optimize_thresholds=optimize_thresholds,
            calibrate=calibrate,
            exp_dir=output_dir / "outputs" / "evaluation",
        )

    gradcam_results = None
    analyze_gradcam = bool(
        getattr(
            args,
            "analyze_gradcam",
            ablations.get("analyze_gradcam", False),
        )
    )

    if analyze_gradcam:
        logger.info("Running quantitative Grad-CAM analysis.")

        from pneumonia.explainability import analyze_gradcam_quantitative

        gradcam_results = analyze_gradcam_quantitative(
            evaluation_model,
            val_generator,
            num_samples=50,
            exp_dir=output_dir / "outputs",
        )

    result = {
        "status": "completed",
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "history_path": str(history_path),
        "epochs_completed": len(history.epoch),
        "training_metrics": history_summary,
        "evaluation": evaluation_results,
        "gradcam": gradcam_results,
    }

    _save_json(
        output_dir / "outputs" / "training_results.json",
        result,
    )

    logger.info("Training completed successfully.")
    logger.info("Best checkpoint: %s", best_model_path)

    return result


def evaluate_with_mc_dropout(
    model_path: str | Path,
    config: ExperimentConfig | dict[str, Any] | str | Path | None = None,
    output_dir: str | Path | None = None,
    optimize_thresholds: bool = False,
    calibrate: bool = False,
) -> dict[str, Any]:
    """
    Load a trained model and run the project's MC-dropout evaluation.

    This function delegates prediction and metric generation to
    evaluation.py. The evaluation implementation must correctly enable
    dropout at inference time for uncertainty estimates to be meaningful.
    """
    config_obj = _load_config(config)
    model_path = _resolve_path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    _, val_generator = create_data_generators(config=config_obj)
    model = tf.keras.models.load_model(model_path)

    from pneumonia.evaluation import evaluate_model

    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "mc_dropout_evaluation"
    output_dir = _resolve_path(output_dir)

    return evaluate_model(
        model,
        val_generator,
        history=None,
        use_mc_dropout=True,
        optimize_thresholds=optimize_thresholds,
        calibrate=calibrate,
        exp_dir=output_dir,
    )