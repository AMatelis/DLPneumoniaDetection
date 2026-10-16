
"""Data loading and augmentation for chest X-ray pneumonia classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pneumonia.config import ExperimentConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def _resolve_path(path: str | Path) -> Path:
    """Resolve a dataset path relative to the project root when necessary."""
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _get_dataset_directories(
    config: dict[str, Any],
    split: str,
) -> list[Path]:
    """Get directories for a split from both configured datasets."""
    data_config = config.get("data", {})
    directories = []

    for dataset_key in ("dataset_a", "dataset_b"):
        dataset_config = data_config.get(dataset_key, {})
        directory = dataset_config.get(f"{split}_dir")

        if directory:
            directories.append(_resolve_path(directory))

    # Allow an explicit list of directories to override YAML paths.
    return directories


def _make_dataframe(
    directories: Sequence[str | Path],
    split_name: str,
) -> pd.DataFrame:
    """Collect labeled image paths from NORMAL and PNEUMONIA folders."""
    records = []

    for directory in directories:
        root = _resolve_path(directory)

        if not root.is_dir():
            logger.warning("%s directory does not exist: %s", split_name, root)
            continue

        for class_name in CLASS_NAMES:
            class_directory = root / class_name

            if not class_directory.is_dir():
                logger.warning(
                    "Missing class directory for %s: %s",
                    split_name,
                    class_directory,
                )
                continue

            for image_path in sorted(class_directory.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
                ):
                    records.append(
                        {
                            "filename": str(image_path),
                            "class": class_name,
                        }
                    )

    dataframe = pd.DataFrame(records, columns=["filename", "class"])

    if dataframe.empty:
        raise ValueError(
            f"No images found for {split_name}. Checked: "
            + ", ".join(str(_resolve_path(path)) for path in directories)
            + ". Each directory should contain NORMAL and PNEUMONIA folders."
        )

    return dataframe


def create_data_generators(
    train_dirs: Sequence[str | Path] | None = None,
    val_dirs: Sequence[str | Path] | None = None,
    fine_tune_ratio: float | None = None,
    config: ExperimentConfig | dict[str, Any] | None = None,
):
    """
    Create Keras generators for binary chest X-ray classification.

    Expected directory layout:
        dataset/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── val/
            ├── NORMAL/
            └── PNEUMONIA/

    Multiple training and validation directories can be combined.

    Args:
        train_dirs: Optional explicit list of training directories.
        val_dirs: Optional explicit list of validation directories.
        fine_tune_ratio: Optional fraction of training images to sample.
            Must be greater than 0 and at most 1.
        config: ExperimentConfig instance or configuration dictionary.

    Returns:
        A tuple of (train_generator, val_generator).
    """
    if config is None:
        config_data = ExperimentConfig().to_dict()
    elif isinstance(config, ExperimentConfig):
        config_data = config.to_dict()
    elif isinstance(config, dict):
        config_data = config
    else:
        raise TypeError(
            "config must be an ExperimentConfig instance, dictionary, or None."
        )

    data_config = config_data.get("data", {})
    training_config = config_data.get("training", {})
    model_config = config_data.get("model", {})

    image_size = model_config.get(
        "image_size",
        data_config.get("image_size", [224, 224]),
    )
    image_size = tuple(image_size)

    if len(image_size) != 2 or any(int(size) <= 0 for size in image_size):
        raise ValueError("image_size must contain two positive integers.")

    image_size = tuple(int(size) for size in image_size)
    batch_size = int(training_config.get("batch_size", 32))

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    seed = int(config_data.get("reproducibility", {}).get("seed", 42))
    augmentation = data_config.get("augmentation", {})

    if train_dirs is None:
        train_dirs = _get_dataset_directories(config_data, "train")

    if val_dirs is None:
        val_dirs = _get_dataset_directories(config_data, "val")

    if not train_dirs:
        raise ValueError(
            "No training directories configured. Set data.dataset_a.train_dir "
            "and/or data.dataset_b.train_dir in configs/default.yaml."
        )

    if not val_dirs:
        raise ValueError(
            "No validation directories configured. Set data.dataset_a.val_dir "
            "and/or data.dataset_b.val_dir in configs/default.yaml."
        )

    if fine_tune_ratio is not None:
        if not 0 < fine_tune_ratio <= 1:
            raise ValueError("fine_tune_ratio must be greater than 0 and at most 1.")

    train_df = _make_dataframe(train_dirs, "training")
    val_df = _make_dataframe(val_dirs, "validation")

    if fine_tune_ratio is not None and fine_tune_ratio < 1:
        train_df = train_df.groupby(
            "class", group_keys=False
        ).sample(
            frac=fine_tune_ratio,
            random_state=seed,
        )

        if train_df.empty:
            raise ValueError("No training images remain after sampling.")

    logger.info("Training images: %d", len(train_df))
    logger.info("Validation images: %d", len(val_df))
    logger.info("Image size: %s; batch size: %d", image_size, batch_size)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=float(augmentation.get("rotation_range", 20)),
        width_shift_range=float(
            augmentation.get("width_shift_range", 0.2)
        ),
        height_shift_range=float(
            augmentation.get("height_shift_range", 0.2)
        ),
        shear_range=float(augmentation.get("shear_range", 0.2)),
        zoom_range=float(augmentation.get("zoom_range", 0.2)),
        horizontal_flip=bool(
            augmentation.get("horizontal_flip", True)
        ),
        fill_mode=str(augmentation.get("fill_mode", "nearest")),
    )

    # Validation images are only rescaled, not augmented.
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    generator_options = {
        "x_col": "filename",
        "y_col": "class",
        "classes": CLASS_NAMES,
        "class_mode": "binary",
        "target_size": image_size,
        "batch_size": batch_size,
        "color_mode": "rgb",
        "interpolation": "bilinear",
    }

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        shuffle=True,
        seed=seed,
        **generator_options,
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        shuffle=False,
        **generator_options,
    )

    logger.info(
        "Class mapping: %s",
        train_generator.class_indices,
    )

    return train_generator, val_generator