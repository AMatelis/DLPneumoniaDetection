
"""Dataset-shift experiments for pneumonia classification.

Compare a model trained on one dataset with its performance on another.
Optionally fine-tune on a subset of the target training data and evaluate
again on the target validation data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from pneumonia.config import ExperimentConfig, PROJECT_ROOT
from pneumonia.data import create_data_generators
from pneumonia.model import build_model

logger = logging.getLogger(__name__)

DATASET_NAMES = {
    "A": "Archive/ChestXRay",
    "B": "ExtraDataset",
}


def _load_config(
    config: ExperimentConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return configuration as a dictionary."""
    if config is None:
        return ExperimentConfig().to_dict()

    if isinstance(config, ExperimentConfig):
        return config.to_dict()

    if isinstance(config, dict):
        return config

    raise TypeError("config must be an ExperimentConfig, dictionary, or None.")


def _get_dataset_config(
    config: dict[str, Any],
    dataset_id: str,
) -> dict[str, Any]:
    """Get paths and metadata for Dataset A or Dataset B."""
    if dataset_id not in DATASET_NAMES:
        raise ValueError(
            f"Unknown dataset '{dataset_id}'. Choose 'A' or 'B'."
        )

    key = "dataset_a" if dataset_id == "A" else "dataset_b"
    dataset_config = config.get("data", {}).get(key, {})

    required_splits = ("train_dir", "val_dir")
    missing = [
        split for split in required_splits
        if not dataset_config.get(split)
    ]

    if missing:
        raise ValueError(
            f"Missing configuration for Dataset {dataset_id}: "
            + ", ".join(missing)
        )

    return dataset_config


def extract_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    """Calculate classification metrics from labels and probabilities."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    predictions = np.asarray(predictions, dtype=float).ravel()

    if not (
        len(y_true) == len(y_pred) == len(predictions)
    ):
        raise ValueError(
            "Labels, predicted labels, and probabilities must have equal lengths."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for an empty dataset.")

    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values.")

    results = {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["NORMAL", "PNEUMONIA"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # ROC AUC is undefined if the evaluation set contains only one class.
    if len(np.unique(y_true)) == 2:
        results["auc"] = float(roc_auc_score(y_true, predictions))
    else:
        results["auc"] = None
        logger.warning(
            "ROC AUC cannot be calculated because the evaluation set "
            "contains only one class."
        )

    return results


def _evaluate_model(
    model,
    generator,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a model on a generator without shuffling image order."""
    if generator.samples == 0:
        raise ValueError("Cannot evaluate on an empty generator.")

    generator.reset()
    probabilities = model.predict(generator, verbose=0).reshape(-1)

    # Some generators can produce an incomplete final batch. Match labels
    # to the predictions actually returned by the model.
    y_true = np.asarray(generator.classes[:len(probabilities)]).astype(int)
    y_pred = (probabilities >= threshold).astype(int)

    results = extract_metrics(y_true, y_pred, probabilities)
    results["threshold"] = float(threshold)

    return results


def _save_json(results: dict[str, Any], path: Path) -> None:
    """Save experiment results as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)


def run_dataset_shift_experiment(
    source_dataset: str,
    target_dataset: str,
    fine_tune: bool = False,
    config: ExperimentConfig | dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    fine_tune_ratio: float | None = None,
    epochs: int = 20,
) -> dict[str, Any]:
    """
    Train on a source dataset and evaluate on a target dataset.

    The experiment:
      1. Trains a model using the source training split.
      2. Evaluates on the source validation split.
      3. Evaluates on the target validation split before adaptation.
      4. Optionally fine-tunes on a subset of target training images.
      5. Evaluates on the same target validation split after fine-tuning.

    Dataset A and B must be configured in configs/default.yaml.

    Args:
        source_dataset: 'A' or 'B'.
        target_dataset: 'A' or 'B'; must differ from source_dataset.
        fine_tune: Whether to fine-tune using target training images.
        config: Optional ExperimentConfig or configuration dictionary.
        output_dir: Where to save models and results.
        fine_tune_ratio: Fraction of target training images to use.
            Defaults to dataset_shift.split_ratio from the config.
        epochs: Maximum training epochs for this experiment.

    Returns:
        Dictionary containing source, target, and optional adapted metrics.
    """
    source_dataset = source_dataset.upper()
    target_dataset = target_dataset.upper()

    if source_dataset not in DATASET_NAMES:
        raise ValueError("source_dataset must be 'A' or 'B'.")

    if target_dataset not in DATASET_NAMES:
        raise ValueError("target_dataset must be 'A' or 'B'.")

    if source_dataset == target_dataset:
        raise ValueError(
            "Source and target datasets must differ to measure dataset shift."
        )

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    config_data = _load_config(config)
    source_config = _get_dataset_config(config_data, source_dataset)
    target_config = _get_dataset_config(config_data, target_dataset)

    if fine_tune_ratio is None:
        fine_tune_ratio = float(
            config_data.get("dataset_shift", {}).get("split_ratio", 0.1)
        )

    if not 0 < fine_tune_ratio <= 1:
        raise ValueError("fine_tune_ratio must be greater than 0 and at most 1.")

    if output_dir is None:
        output_root = PROJECT_ROOT / "outputs" / (
            f"shift_{source_dataset}_to_{target_dataset}"
        )
    else:
        output_root = Path(output_dir)
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root

    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Dataset shift experiment: %s -> %s",
        DATASET_NAMES[source_dataset],
        DATASET_NAMES[target_dataset],
    )

    # Phase 1: Train on the source dataset.
    source_train, source_val = create_data_generators(
        train_dirs=[source_config["train_dir"]],
        val_dirs=[source_config["val_dir"]],
        config=config_data,
    )

    model = build_model()

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=int(
                config_data.get("training", {}).get(
                    "early_stopping_patience", 10
                )
            ),
            restore_best_weights=True,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.1,
            patience=int(
                config_data.get("training", {}).get(
                    "reduce_lr_patience", 5
                )
            ),
            mode="max",
        ),
    ]

    logger.info("Training on Dataset %s.", source_dataset)

    history = model.fit(
        source_train,
        validation_data=source_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Measure source and target performance before adaptation.
    logger.info("Evaluating on source validation data.")
    source_metrics = _evaluate_model(model, source_val)

    _, target_val = create_data_generators(
        train_dirs=[target_config["train_dir"]],
        val_dirs=[target_config["val_dir"]],
        config=config_data,
    )

    logger.info("Evaluating on target validation data before fine-tuning.")
    target_metrics = _evaluate_model(model, target_val)

    results = {
        "source_dataset": source_dataset,
        "source_dataset_name": DATASET_NAMES[source_dataset],
        "target_dataset": target_dataset,
        "target_dataset_name": DATASET_NAMES[target_dataset],
        "fine_tuning_enabled": fine_tune,
        "source_validation": source_metrics,
        "target_before_fine_tuning": target_metrics,
        "target_after_fine_tuning": None,
        "training_epochs_completed": len(history.history.get("loss", [])),
    }

    if source_metrics["auc"] is not None and target_metrics["auc"] is not None:
        results["auc_drop"] = (
            source_metrics["auc"] - target_metrics["auc"]
        )
    else:
        results["auc_drop"] = None

    # Phase 3: Optionally adapt the model using target training images.
    if fine_tune:
        logger.info(
            "Fine-tuning on %.1f%% of target training data.",
            fine_tune_ratio * 100,
        )

        target_train, _ = create_data_generators(
            train_dirs=[target_config["train_dir"]],
            val_dirs=[target_config["val_dir"]],
            fine_tune_ratio=fine_tune_ratio,
            config=config_data,
        )

        fine_tune_epochs = int(
            config_data.get("dataset_shift", {}).get(
                "fine_tune_epochs", min(5, epochs)
            )
        )

        if fine_tune_epochs < 1:
            raise ValueError("dataset_shift.fine_tune_epochs must be at least 1.")

        fine_tune_history = model.fit(
            target_train,
            validation_data=target_val,
            epochs=fine_tune_epochs,
            callbacks=[
                EarlyStopping(
                    monitor="val_auc",
                    patience=min(3, fine_tune_epochs),
                    restore_best_weights=True,
                    mode="max",
                )
            ],
            verbose=1,
        )

        logger.info("Evaluating target performance after fine-tuning.")
        adapted_metrics = _evaluate_model(model, target_val)

        results["target_after_fine_tuning"] = adapted_metrics
        results["fine_tuning_epochs_completed"] = len(
            fine_tune_history.history.get("loss", [])
        )

        if (
            target_metrics["auc"] is not None
            and adapted_metrics["auc"] is not None
        ):
            results["auc_change_after_fine_tuning"] = (
                adapted_metrics["auc"] - target_metrics["auc"]
            )
        else:
            results["auc_change_after_fine_tuning"] = None

        model.save(output_root / "adapted_model.keras")

    model.save(output_root / "source_model.keras")

    _save_json(results, output_root / "dataset_shift_results.json")

    logger.info("Results saved to %s", output_root)
    return results