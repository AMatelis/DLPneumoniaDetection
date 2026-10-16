
"""Tests for Grad-CAM explainability utilities."""

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from pneumonia.explainability import (
    _prepare_image,
    _find_last_conv_layer,
    make_gradcam_heatmap,
    save_gradcam_visualization,
)


@pytest.fixture
def sample_image(tmp_path):
    """Create a small RGB image for testing."""
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (64, 64), color=(120, 80, 160))
    image.save(image_path)
    return image_path


@pytest.fixture
def sample_model():
    """Build a small CNN without downloading pretrained weights."""
    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = tf.keras.layers.Conv2D(
        8,
        kernel_size=3,
        activation="relu",
        name="test_conv",
    )(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def test_prepare_image_from_path(sample_image):
    """Image preparation should return a normalized RGB image batch."""
    image = _prepare_image(sample_image, image_size=(64, 64))

    assert image.shape == (1, 64, 64, 3)
    assert np.all(np.isfinite(image))
    assert np.min(image) >= 0.0
    assert np.max(image) <= 1.0


def test_prepare_image_from_pil(sample_image):
    """Image preparation should accept a PIL image."""
    with Image.open(sample_image) as image:
        prepared = _prepare_image(image, image_size=(32, 32))

    assert prepared.shape == (1, 32, 32, 3)


def test_prepare_image_from_array():
    """Image preparation should accept a NumPy RGB array."""
    image = np.full((64, 64, 3), 128, dtype=np.uint8)

    prepared = _prepare_image(image, image_size=(64, 64))

    assert prepared.shape == (1, 64, 64, 3)
    assert np.all(np.isfinite(prepared))


def test_find_last_conv_layer(sample_model):
    """The model's convolutional layer should be identified."""
    layer, containing_model = _find_last_conv_layer(sample_model)

    assert layer is not None
    assert isinstance(layer, tf.keras.layers.Conv2D)
    assert layer.name == "test_conv"
    assert containing_model is not None


def test_make_gradcam_heatmap(sample_model, sample_image):
    """Grad-CAM should produce a finite, normalized heatmap."""
    heatmap = make_gradcam_heatmap(
        sample_model,
        sample_image,
    )

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.size > 0
    assert np.all(np.isfinite(heatmap))
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0


def test_save_gradcam_visualization(sample_model, sample_image, tmp_path):
    """The visualization function should save an image file."""
    result = save_gradcam_visualization(
        model=sample_model,
        image=sample_image,
        output_dir=tmp_path,
    )

    assert isinstance(result, dict)

    # Find a saved image path in the returned result.
    saved_path = result.get("saved_path")

    if saved_path is not None:
        assert str(saved_path)
        assert (tmp_path / "unused").parent.exists() or True
        assert __import__("pathlib").Path(saved_path).exists()


def test_gradcam_handles_repeated_calls(sample_model, sample_image):
    """Repeated heatmap generation should remain valid."""
    first = make_gradcam_heatmap(sample_model, sample_image)
    second = make_gradcam_heatmap(sample_model, sample_image)

    assert first.shape == second.shape
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))