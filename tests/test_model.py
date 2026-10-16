
"""Tests for pneumonia model construction and configuration."""

import pytest
import tensorflow as tf

from pneumonia.model import build_model, unfreeze_resnet_layers


@pytest.fixture
def model_config():
    """Provide a small model configuration for testing."""
    return {
        "model": {
            "architecture": "ResNet50",
            "pretrained_weights": None,
            "image_size": [64, 64],
            "model_path": "test_model.keras",
        },
        "training": {
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "batch_size": 2,
        },
    }


def test_build_model_returns_keras_model(model_config):
    """Model construction should return a Keras model."""
    model = build_model(config=model_config)

    assert isinstance(model, tf.keras.Model)


def test_build_model_input_shape(model_config):
    """The model should accept RGB images of the configured size."""
    model = build_model(config=model_config)

    assert tuple(model.input_shape) == (None, 64, 64, 3)


def test_build_model_output_shape(model_config):
    """The model should output one pneumonia probability per image."""
    model = build_model(config=model_config)

    assert tuple(model.output_shape) == (None, 1)


def test_build_model_uses_sigmoid_output(model_config):
    """The final layer should use sigmoid for binary classification."""
    model = build_model(config=model_config)

    assert model.layers[-1].activation == tf.keras.activations.sigmoid


def test_build_model_is_compiled(model_config):
    """The model should be compiled with a loss and optimizer."""
    model = build_model(config=model_config)

    assert model.optimizer is not None
    assert model.loss == "binary_crossentropy"


def test_build_model_freezes_backbone_by_default(model_config):
    """The pretrained feature extractor should initially be frozen."""
    model = build_model(config=model_config)

    backbone = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and "resnet50" in layer.name.lower()
        ),
        None,
    )

    assert backbone is not None
    assert backbone.trainable is False


def test_build_model_accepts_custom_dropout_rate(model_config):
    """A custom dropout rate should be accepted."""
    model = build_model(
        config=model_config,
        dropout_rate=0.3,
    )

    dropout_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout)
    ]

    assert dropout_layers
    assert dropout_layers[0].rate == pytest.approx(0.3)


def test_model_produces_probabilities(model_config):
    """Predictions should be finite values between zero and one."""
    model = build_model(config=model_config)

    images = tf.random.uniform((2, 64, 64, 3))
    predictions = model(images, training=False).numpy()

    assert predictions.shape == (2, 1)
    assert (predictions >= 0.0).all()
    assert (predictions <= 1.0).all()


def test_unfreeze_resnet_layers_returns_model(model_config):
    """Unfreezing layers should return the model."""
    model = build_model(config=model_config)

    result = unfreeze_resnet_layers(
        model,
        trainable_from=-10,
    )

    assert result is model


def test_unfreeze_resnet_layers_keeps_batch_norm_frozen(model_config):
    """Batch normalization layers should remain frozen during fine-tuning."""
    model = build_model(config=model_config)

    unfreeze_resnet_layers(
        model,
        trainable_from=-10,
    )

    backbone = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and "resnet50" in layer.name.lower()
        ),
        None,
    )

    assert backbone is not None

    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable is False