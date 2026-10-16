
"""Configuration management for the pneumonia detection project."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Load, validate, and manage experiment configuration settings."""

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """Return default settings consistent with the original pipeline."""
        return {
            "experiment": {
                "name": "PneumoniaDetection_Baseline",
                "version": "v1.0.0",
                "description": (
                    "Pneumonia detection from chest X-ray images"
                ),
            },
            "reproducibility": {
                "seed": 42,
            },
            "training": {
                "num_epochs": 50,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "optimizer": "Adam",
                "loss": "binary_crossentropy",
                "early_stopping_patience": 10,
                "reduce_lr_patience": 5,
            },
            "model": {
                "architecture": "ResNet50",
                "pretrained_weights": "imagenet",
                "image_size": [224, 224],
                "model_path": "models/best_model.h5",
            },
            "data": {
                "dataset_a": {
                    "name": "Archive/ChestXRay",
                    "train_dir": "data/archive/chest_xray/train",
                    "val_dir": "data/archive/chest_xray/val",
                    "test_dir": "data/archive/chest_xray/test",
                },
                "dataset_b": {
                    "name": "ExtraDataset",
                    "train_dir": "data/extra_dataset/train",
                    "val_dir": "data/extra_dataset/val",
                    "test_dir": "data/extra_dataset/test",
                },
                "combine_datasets": True,
                "augmentation": {
                    "rotation_range": 20,
                    "width_shift_range": 0.2,
                    "height_shift_range": 0.2,
                    "zoom_range": 0.2,
                    "shear_range": 0.2,
                    "horizontal_flip": True,
                    "fill_mode": "nearest",
                },
            },
            "dataset_shift": {
                "split_ratio": 0.1,
            },
            "ablations": {
                "use_mc_dropout": True,
                "optimize_thresholds": True,
                "calibrate": True,
                "analyze_gradcam": True,
                "use_class_weights": True,
            },
            "evaluation": {
                "mc_dropout_samples": 10,
                "calibration_split": 0.5,
                "threshold_bins": 100,
                "reliability_bins": 10,
            },
            "output": {
                "save_predictions_csv": True,
                "save_json_configs": True,
                "create_plots": True,
                "plot_dpi": 300,
                "base_dir": "outputs",
            },
            "logging": {
                "level": "INFO",
            },
        }

    def __init__(
        self,
        config_path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        """Initialize configuration from defaults, YAML, and overrides."""
        self.config_path = (
            Path(config_path) if config_path is not None else None
        )
        self.config = self.get_default_config()

        if self.config_path is not None:
            self.load_yaml(self.config_path)

        if overrides:
            self._deep_update(self.config, overrides)

        self.validate()

    @staticmethod
    def _deep_update(
        destination: dict[str, Any],
        updates: dict[str, Any],
    ) -> None:
        """Recursively merge nested dictionaries."""
        for key, value in updates.items():
            if (
                key in destination
                and isinstance(destination[key], dict)
                and isinstance(value, dict)
            ):
                ExperimentConfig._deep_update(destination[key], value)
            else:
                destination[key] = copy.deepcopy(value)

    def load_yaml(self, path: str | Path) -> dict[str, Any]:
        """Load YAML settings and merge them with defaults."""
        yaml_path = Path(path)

        if not yaml_path.is_absolute():
            yaml_path = PROJECT_ROOT / yaml_path

        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {yaml_path}"
            )

        with yaml_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}

        if not isinstance(loaded, dict):
            raise ValueError(
                "The YAML configuration must contain a mapping."
            )

        self._deep_update(self.config, loaded)
        self.config_path = yaml_path

        return self.config

    def validate(self) -> None:
        """Validate essential configuration values."""
        training = self.config["training"]
        model = self.config["model"]
        evaluation = self.config["evaluation"]
        dataset_shift = self.config["dataset_shift"]

        if int(training["num_epochs"]) < 1:
            raise ValueError("training.num_epochs must be at least 1.")

        if int(training["batch_size"]) < 1:
            raise ValueError("training.batch_size must be at least 1.")

        if float(training["learning_rate"]) <= 0:
            raise ValueError(
                "training.learning_rate must be greater than 0."
            )

        image_size = model["image_size"]
        if (
            not isinstance(image_size, (list, tuple))
            or len(image_size) != 2
            or any(int(dimension) < 1 for dimension in image_size)
        ):
            raise ValueError(
                "model.image_size must contain two positive integers."
            )

        if int(evaluation["mc_dropout_samples"]) < 1:
            raise ValueError(
                "evaluation.mc_dropout_samples must be at least 1."
            )

        if int(evaluation["reliability_bins"]) < 1:
            raise ValueError(
                "evaluation.reliability_bins must be at least 1."
            )

        split_ratio = float(dataset_shift["split_ratio"])
        if not 0 < split_ratio < 1:
            raise ValueError(
                "dataset_shift.split_ratio must be between 0 and 1."
            )

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve a top-level configuration section."""
        return self.config.get(key, default)

    def get_value(
        self,
        section: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve a value from a configuration section."""
        section_config = self.config.get(section, {})

        if not isinstance(section_config, dict):
            return default

        return section_config.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the complete configuration."""
        return copy.deepcopy(self.config)

    def save(self, path: str | Path) -> Path:
        """Save the current configuration as JSON."""
        output_path = Path(path)

        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=4, default=str)

        logger.info("Configuration saved to %s", output_path)
        return output_path


def load_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience function returning configuration as a dictionary."""
    return ExperimentConfig(path).to_dict()