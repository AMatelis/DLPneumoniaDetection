
"""Train the pneumonia detection model."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pneumonia.config import ExperimentConfig
from pneumonia.data import create_data_generators
from pneumonia.model import build_model
from pneumonia.evaluation import evaluate_model
from pneumonia.explainability import analyze_gradcam_quantitative


SEED = 42
EXPERIMENT_NAME = "PneumoniaDetection_Baseline"
EXPERIMENT_VERSION = "v1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class DummyHistory:
    """History object for evaluation when no training history is available."""

    def __init__(self):
        self.history = {
            "accuracy": [],
            "val_accuracy": [],
            "auc": [],
            "val_auc": [],
        }


def parse_args():
    """Parse training options."""
    parser = argparse.ArgumentParser(
        description="Train a ResNet50 pneumonia detection model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=EXPERIMENT_NAME,
        help="Name used to organize training outputs.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_model.h5",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "training",
        help="Directory for training logs and configuration.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting for imbalanced classes.",
    )
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Run threshold optimization after training.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run probability calibration after training.",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Run quantitative Grad-CAM analysis after training.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Train the model without running post-training evaluation.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    return args


def set_random_seeds(seed=SEED):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_training_config(args, config, class_weights=None):
    """Save the training settings used for this run."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_record = {
        "experiment_name": args.experiment_name,
        "experiment_version": EXPERIMENT_VERSION,
        "random_seed": SEED,
        "epochs_requested": args.epochs,
        "model_path": str(args.model_path.resolve()),
        "config_path": str(args.config.resolve()),
        "class_weights_enabled": not args.no_class_weights,
        "class_weights": class_weights,
        "threshold_optimization": args.optimize_thresholds,
        "calibration": args.calibrate,
        "gradcam": args.gradcam,
        "configuration": config.config,
    }

    config_path = args.output_dir / "training_config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(training_record, file, indent=4, default=str)

    logger.info("Saved training configuration to %s", config_path)


def main():
    args = parse_args()
    set_random_seeds()

    if not args.config.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {args.config}"
        )

    logger.info("Loading configuration from %s", args.config)
    config = ExperimentConfig(str(args.config))

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating training and validation generators...")
    train_generator, val_generator = create_data_generators()

    if train_generator is None or val_generator is None:
        raise RuntimeError(
            "Data generators were not created. Check your dataset paths."
        )

    if len(train_generator.classes) == 0:
        raise ValueError("The training dataset contains no labeled images.")

    if len(val_generator.classes) == 0:
        raise ValueError("The validation dataset contains no labeled images.")

    logger.info(
        "Training images: %d | Validation images: %d",
        len(train_generator.classes),
        len(val_generator.classes),
    )

    class_weights = None

    if not args.no_class_weights:
        classes = np.unique(train_generator.classes)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=train_generator.classes,
        )

        class_weights = {
            int(label): float(weight)
            for label, weight in zip(classes, weights)
        }

        logger.info("Using class weights: %s", class_weights)

    save_training_config(args, config, class_weights)

    logger.info("Building ResNet50 model...")
    model = build_model()

    training_config = config.config.get("training", {})

    early_stopping_patience = training_config.get(
        "early_stopping_patience", 10
    )
    reduce_lr_patience = training_config.get(
        "reduce_lr_patience", 5
    )

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(args.model_path),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.1,
            patience=reduce_lr_patience,
            mode="max",
            verbose=1,
        ),
    ]

    logger.info("Starting training for up to %d epochs...", args.epochs)

    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Save the final model as well as the best checkpoint.
    final_model_path = args.model_path.with_name(
        f"{args.model_path.stem}_final{args.model_path.suffix}"
    )
    model.save(str(final_model_path))

    history_path = args.output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=4)

    logger.info("Best checkpoint: %s", args.model_path)
    logger.info("Final-epoch model: %s", final_model_path)
    logger.info("Training history: %s", history_path)

    if args.skip_evaluation:
        logger.info("Skipping post-training evaluation.")
        return

    logger.info("Running post-training evaluation...")

    # Use the best saved checkpoint for evaluation.
    from tensorflow.keras.models import load_model

    best_model = load_model(str(args.model_path))

    evaluate_model(
        best_model,
        val_generator,
        history,
        optimize_thresholds=args.optimize_thresholds,
        calibrate=args.calibrate,
        exp_dir=str(args.output_dir),
    )

    if args.gradcam:
        logger.info("Running Grad-CAM analysis...")
        analyze_gradcam_quantitative(
            best_model,
            val_generator,
            num_samples=50,
            exp_dir=str(args.output_dir),
        )

    logger.info("Training and evaluation completed.")


if __name__ == "__main__":
    main()