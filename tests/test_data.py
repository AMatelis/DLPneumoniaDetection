
"""Tests for pneumonia image data loading."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pneumonia.data import (
    CLASS_NAMES,
    _make_dataframe,
    _resolve_path,
)


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a small temporary dataset with both classes."""
    for split in ("train", "val"):
        for class_name in CLASS_NAMES:
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)

            # Create two small RGB images for each class.
            for index in range(2):
                image = Image.new(
                    "RGB",
                    (32, 32),
                    color=(index * 50, 100, 150),
                )
                image.save(class_dir / f"{class_name}_{index}.png")

    return tmp_path


def test_class_names():
    """The loader should use the expected binary class names."""
    assert CLASS_NAMES == ["NORMAL", "PNEUMONIA"]


def test_resolve_absolute_path(tmp_path):
    """Absolute paths should remain absolute."""
    result = _resolve_path(tmp_path)

    assert result == tmp_path


def test_make_dataframe_finds_images(sample_dataset):
    """The dataframe should include images from both class folders."""
    train_dir = sample_dataset / "train"

    dataframe = _make_dataframe([train_dir])

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 4
    assert {"filename", "class"}.issubset(dataframe.columns)


def test_make_dataframe_contains_both_classes(sample_dataset):
    """The dataframe should identify NORMAL and PNEUMONIA images."""
    train_dir = sample_dataset / "train"

    dataframe = _make_dataframe([train_dir])

    assert set(dataframe["class"].unique()) == set(CLASS_NAMES)


def test_make_dataframe_ignores_unsupported_files(sample_dataset):
    """Non-image files should not be included."""
    train_dir = sample_dataset / "train"
    normal_dir = train_dir / "NORMAL"

    (normal_dir / "notes.txt").write_text(
        "Not an image",
        encoding="utf-8",
    )

    dataframe = _make_dataframe([train_dir])

    assert len(dataframe) == 4
    assert not dataframe["filename"].str.endswith(".txt").any()


def test_make_dataframe_raises_without_images(tmp_path):
    """An empty dataset should raise an error."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises((ValueError, FileNotFoundError)):
        _make_dataframe([empty_dir])


def test_create_data_generators(sample_dataset):
    """The generator factory should return usable train and validation generators."""
    from pneumonia.data import create_data_generators
    from pneumonia.config import ExperimentConfig

    config = ExperimentConfig(
        overrides={
            "model": {
                "image_size": [32, 32],
            },
            "training": {
                "batch_size": 2,
            },
        }
    )

    train_dir = sample_dataset / "train"
    val_dir = sample_dataset / "val"

    train_generator, val_generator = create_data_generators(
        train_dirs=[train_dir],
        val_dirs=[val_dir],
        config=config,
    )

    assert train_generator is not None
    assert val_generator is not None
    assert train_generator.samples == 4
    assert val_generator.samples == 4


def test_generator_images_have_expected_shape(sample_dataset):
    """Generated batches should have the configured image dimensions."""
    from pneumonia.data import create_data_generators
    from pneumonia.config import ExperimentConfig

    config = ExperimentConfig(
        overrides={
            "model": {
                "image_size": [32, 32],
            },
            "training": {
                "batch_size": 2,
            },
        }
    )

    train_dir = sample_dataset / "train"
    val_dir = sample_dataset / "val"

    train_generator, _ = create_data_generators(
        train_dirs=[train_dir],
        val_dirs=[val_dir],
        config=config,
    )

    images, labels = next(train_generator)

    assert images.shape == (2, 32, 32, 3)
    assert labels.shape == (2,)
    assert np.all(np.isfinite(images))
    assert np.all(images >= 0.0)
    assert np.all(images <= 1.0)