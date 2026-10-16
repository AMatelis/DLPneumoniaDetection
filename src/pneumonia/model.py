
"""Model architecture and construction for pneumonia detection."""

from __future__ import annotations

import logging
from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

from pneumonia.config import ExperimentConfig

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DROPOUT_RATE = 0.5


def _config_dict(
    config: ExperimentConfig | dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert supported configuration inputs to a dictionary."""
    if config is None:
        return ExperimentConfig().to_dict()

    if isinstance(config, ExperimentConfig):
        return config.to_dict()

    if isinstance(config, dict):
        return config

    raise TypeError(
        "config must be an ExperimentConfig, dictionary, or None."
    )


def build_model(
    config: ExperimentConfig | dict[str, Any] | None = None,
    pretrained_weights: str | None = None,
    freeze_base: bool = True,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> Model:
    """
    Build and compile the project's ResNet50 binary classifier.

    Architecture:
        Input -> ImageNet-pretrained ResNet50 -> Global Average Pooling
        -> Dense(128, ReLU) -> Dropout -> Dense(1, Sigmoid)

    Args:
        config: Optional ExperimentConfig or configuration dictionary.
        pretrained_weights: Backbone weights, normally "imagenet".
            Use None to initialize the backbone without pretrained weights.
        freeze_base: Whether to freeze the ResNet50 feature extractor.
        dropout_rate: Dropout probability for the classification head.

    Returns:
        A compiled Keras model for binary classification.
    """
    settings = _config_dict(config)

    model_settings = settings.get("model", {})
    training_settings = settings.get("training", {})

    image_size = model_settings.get(
        "image_size",
        settings.get("data", {}).get("image_size", DEFAULT_IMAGE_SIZE),
    )
    image_size = tuple(int(dimension) for dimension in image_size)

    if len(image_size) != 2 or any(dimension < 1 for dimension in image_size):
        raise ValueError(
            "model.image_size must contain two positive integers."
        )

    learning_rate = float(
        training_settings.get("learning_rate", DEFAULT_LEARNING_RATE)
    )
    if learning_rate <= 0:
        raise ValueError("The learning rate must be greater than zero.")

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the range [0, 1).")

    if pretrained_weights is None:
        pretrained_weights = model_settings.get(
            "pretrained_weights", "imagenet"
        )

    architecture = model_settings.get("architecture", "ResNet50")
    if architecture != "ResNet50":
        raise ValueError(
            f"Unsupported architecture: {architecture!r}. "
            "This module currently implements ResNet50."
        )

    logger.info(
        "Building ResNet50 model with input size %s and learning rate %g.",
        image_size,
        learning_rate,
    )

    # The base network extracts image features. The original project
    # freezes these layers during initial training.
    base_model = ResNet50(
        include_top=False,
        weights=pretrained_weights,
        input_shape=(image_size[0], image_size[1], 3),
    )
    base_model.trainable = not freeze_base

    inputs = layers.Input(
        shape=(image_size[0], image_size[1], 3),
        name="xray_input",
    )

    # Keep the pretrained feature extractor in inference mode when frozen.
    # This prevents its Batch Normalization statistics from being updated
    # during head training.
    features = base_model(inputs, training=False if freeze_base else None)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(features)

    x = layers.Dense(
        128,
        activation="relu",
        name="classification_dense",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="classification_dropout",
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="pneumonia_probability",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="pneumonia_resnet50",
    )

    optimizer_name = training_settings.get("optimizer", "Adam")
    if optimizer_name.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name!r}. "
            "This module currently supports Adam."
        )

    loss_name = training_settings.get(
        "loss", "binary_crossentropy"
    )
    if loss_name != "binary_crossentropy":
        raise ValueError(
            "This binary sigmoid classifier requires "
            "binary_crossentropy loss."
        )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
            ),
            tf.keras.metrics.AUC(
                name="auc",
            ),
        ],
    )

    logger.info(
        "Model built successfully. Trainable parameters: %s",
        f"{sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}",
    )

    return model


def unfreeze_resnet_layers(
    model: Model,
    trainable_from: int = -30,
) -> Model:
    """
    Enable fine-tuning of later ResNet50 layers.

    Call this only after initial training, then recompile the model with
    a suitably small learning rate before continuing training.

    Args:
        model: A model returned by build_model().
        trainable_from: Index in the ResNet50 layer list from which layers
            should become trainable. The default unfreezes approximately
            the last 30 layers.

    Returns:
        The same model with the requested backbone layers unfrozen.
    """
    base_model = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and layer.name.startswith("resnet50")
        ),
        None,
    )

    if base_model is None:
        raise ValueError(
            "Could not find the ResNet50 backbone in this model."
        )

    base_model.trainable = True

    if trainable_from < 0:
        trainable_from = max(
            0, len(base_model.layers) + trainable_from
        )

    if trainable_from >= len(base_model.layers):
        raise ValueError(
            "trainable_from must refer to a layer within the backbone."
        )

    for index, layer in enumerate(base_model.layers):
        # Keep Batch Normalization layers frozen during fine-tuning for
        # more stable behavior on relatively small datasets.
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = index >= trainable_from

    logger.info(
        "Enabled fine-tuning from ResNet50 layer index %d.",
        trainable_from,
    )

    return model