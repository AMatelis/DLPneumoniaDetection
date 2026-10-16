
"""Uncertainty estimation for the pneumonia detection model.

Implements Monte Carlo Dropout (MC Dropout), predictive entropy,
variance, mutual information, and confidence summaries using TensorFlow/Keras.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)


def _validate_probabilities(
    probabilities: np.ndarray,
    name: str = "probabilities",
) -> np.ndarray:
    """Validate an array of binary-class probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} contains non-finite values.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(f"{name} must be between 0 and 1.")

    return probabilities


def binary_entropy(
    probabilities: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Calculate binary entropy for probabilities of the positive class.

    Entropy is low near probabilities 0 and 1 and high near 0.5.
    The result is measured in nats.
    """
    probabilities = _validate_probabilities(probabilities)
    p = np.clip(probabilities, epsilon, 1.0 - epsilon)

    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def predictive_entropy(
    mc_probabilities: np.ndarray,
) -> np.ndarray:
    """
    Calculate entropy from the mean MC Dropout prediction.

    Args:
        mc_probabilities: Array shaped (samples, examples), containing
            repeated predicted probabilities for the positive class.

    Returns:
        One predictive entropy value per example.
    """
    mc_probabilities = _validate_probabilities(mc_probabilities)

    if mc_probabilities.ndim != 2:
        raise ValueError(
            "mc_probabilities must have shape "
            "(n_samples, n_examples)."
        )

    mean_probabilities = np.mean(mc_probabilities, axis=0)
    return binary_entropy(mean_probabilities)


def expected_entropy(
    mc_probabilities: np.ndarray,
) -> np.ndarray:
    """Calculate the mean entropy across MC Dropout predictions."""
    mc_probabilities = _validate_probabilities(mc_probabilities)

    if mc_probabilities.ndim != 2:
        raise ValueError(
            "mc_probabilities must have shape "
            "(n_samples, n_examples)."
        )

    return np.mean(binary_entropy(mc_probabilities), axis=0)


def predictive_variance(
    mc_probabilities: np.ndarray,
    ddof: int = 0,
) -> np.ndarray:
    """
    Calculate the variance of repeated predicted probabilities.

    Higher variance indicates greater variation between stochastic
    forward passes.
    """
    mc_probabilities = _validate_probabilities(mc_probabilities)

    if mc_probabilities.ndim != 2:
        raise ValueError(
            "mc_probabilities must have shape "
            "(n_samples, n_examples)."
        )

    if mc_probabilities.shape[0] <= ddof:
        raise ValueError("Not enough MC samples for the requested ddof.")

    return np.var(mc_probabilities, axis=0, ddof=ddof)


def mutual_information(
    mc_probabilities: np.ndarray,
) -> np.ndarray:
    """
    Estimate epistemic uncertainty using MC Dropout.

    Mutual information is predictive entropy minus expected entropy.
    It estimates how much uncertainty is associated with variation
    across the model's stochastic predictions.
    """
    total_uncertainty = predictive_entropy(mc_probabilities)
    average_uncertainty = expected_entropy(mc_probabilities)

    # Small negative values can occur due to floating-point rounding.
    return np.maximum(total_uncertainty - average_uncertainty, 0.0)


def _forward_with_dropout_enabled(
    model: tf.keras.Model,
    images: tf.Tensor,
) -> tf.Tensor:
    """
    Run a forward pass with Dropout active and BatchNormalization in
    inference mode for the project's Sequential-style model.

    This avoids passing training=True to the entire ResNet50 model,
    which could otherwise update or use training-mode BatchNormalization
    behavior during uncertainty estimation.

    The model must be a sequential-style stack. For arbitrary functional
    models, a model-specific stochastic forward pass is required.
    """
    if not isinstance(model, tf.keras.Sequential):
        raise TypeError(
            "MC Dropout currently expects the project's Sequential model. "
            "For a Functional Keras model, implement a model-specific "
            "forward pass that activates Dropout layers only."
        )

    outputs = images

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        if isinstance(layer, tf.keras.layers.Dropout):
            outputs = layer(outputs, training=True)
        else:
            # Keep nested ResNet50 and other layers in inference mode.
            outputs = layer(outputs, training=False)

    return outputs


def mc_dropout_predict(
    model: tf.keras.Model,
    images: np.ndarray | tf.Tensor,
    n_samples: int = 10,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate repeated stochastic predictions with MC Dropout.

    Args:
        model: Compiled Keras model built with Dropout layers.
        images: Batch of preprocessed images. Pixel scaling must match
            the training pipeline.
        n_samples: Number of stochastic forward passes.
        batch_size: Optional batch size for splitting a large input batch.

    Returns:
        Dictionary with:
            mc_probabilities: Shape (n_samples, n_examples).
            mean_probability: Mean probability for each example.
            predictive_entropy: Entropy of the mean probability.
            expected_entropy: Mean entropy across predictions.
            predictive_variance: Variance across predictions.
            mutual_information: Approximate epistemic uncertainty.
            confidence: Distance of mean probability from 0.5, scaled
                to the range [0, 1].
            predicted_class: Binary predictions at threshold 0.5.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    images = tf.convert_to_tensor(images, dtype=tf.float32)

    if images.shape.rank is None or images.shape.rank < 1:
        raise ValueError("images must have a batch dimension.")

    number_of_images = int(tf.shape(images)[0].numpy())

    if number_of_images == 0:
        raise ValueError("images contains no examples.")

    if batch_size is None:
        batch_size = number_of_images

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    sample_predictions: list[np.ndarray] = []

    for sample_index in range(n_samples):
        batch_predictions: list[np.ndarray] = []

        for start in range(0, number_of_images, batch_size):
            batch = images[start : start + batch_size]

            outputs = _forward_with_dropout_enabled(model, batch)
            outputs = tf.convert_to_tensor(outputs)

            # The original project uses a single sigmoid output.
            if outputs.shape.rank == 2 and outputs.shape[-1] == 1:
                outputs = tf.squeeze(outputs, axis=-1)

            if outputs.shape.rank != 1:
                raise ValueError(
                    "Expected one sigmoid probability per image. "
                    f"Received output shape {outputs.shape}."
                )

            batch_predictions.append(outputs.numpy())

        predictions = np.concatenate(batch_predictions, axis=0)
        predictions = _validate_probabilities(
            predictions,
            name="model predictions",
        )
        sample_predictions.append(predictions)

        logger.debug(
            "Completed MC Dropout sample %d/%d",
            sample_index + 1,
            n_samples,
        )

    mc_probabilities = np.stack(sample_predictions, axis=0)
    mean_probability = np.mean(mc_probabilities, axis=0)

    return {
        "mc_probabilities": mc_probabilities,
        "mean_probability": mean_probability,
        "predictive_entropy": predictive_entropy(mc_probabilities),
        "expected_entropy": expected_entropy(mc_probabilities),
        "predictive_variance": predictive_variance(mc_probabilities),
        "mutual_information": mutual_information(mc_probabilities),
        "confidence": np.abs(mean_probability - 0.5) * 2.0,
        "predicted_class": (mean_probability >= 0.5).astype(int),
    }


def uncertainty_from_generator(
    model: tf.keras.Model,
    generator: Any,
    n_samples: int = 10,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Estimate uncertainty for images in a Keras data generator.

    The generator is reset before processing. Its labels are returned
    when available. The number of processed batches can be limited for
    exploratory analysis.
    """
    if not hasattr(generator, "__len__") or not hasattr(generator, "__getitem__"):
        raise TypeError("generator must support len() and indexed batches.")

    if hasattr(generator, "reset"):
        generator.reset()

    total_batches = len(generator)

    if max_batches is not None:
        if max_batches < 1:
            raise ValueError("max_batches must be at least 1.")
        total_batches = min(total_batches, max_batches)

    if total_batches == 0:
        raise ValueError("The generator contains no batches.")

    all_images: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch_index in range(total_batches):
        batch = generator[batch_index]

        if not isinstance(batch, (tuple, list)) or len(batch) < 1:
            raise ValueError(
                "Expected each generator batch to contain images "
                "and optionally labels."
            )

        images = np.asarray(batch[0])
        all_images.append(images)

        if len(batch) >= 2:
            all_labels.append(np.asarray(batch[1]).reshape(-1))

    images = np.concatenate(all_images, axis=0)

    if hasattr(generator, "reset"):
        generator.reset()

    results = mc_dropout_predict(
        model,
        images,
        n_samples=n_samples,
    )

    if all_labels:
        labels = np.concatenate(all_labels, axis=0)

        if len(labels) != len(results["mean_probability"]):
            raise ValueError("The number of labels does not match predictions.")

        results["true_labels"] = labels.astype(int)

    return results


def create_uncertainty_dataframe(
    results: dict[str, np.ndarray],
    image_paths: list[str] | None = None,
) -> pd.DataFrame:
    """Convert uncertainty estimates into a row-per-image DataFrame."""
    required_keys = (
        "mean_probability",
        "predictive_entropy",
        "expected_entropy",
        "predictive_variance",
        "mutual_information",
        "confidence",
        "predicted_class",
    )

    for key in required_keys:
        if key not in results:
            raise KeyError(f"Missing required result: {key}")

    dataframe = pd.DataFrame(
        {
            "mean_pneumonia_probability": results["mean_probability"],
            "predictive_entropy": results["predictive_entropy"],
            "expected_entropy": results["expected_entropy"],
            "predictive_variance": results["predictive_variance"],
            "mutual_information": results["mutual_information"],
            "confidence": results["confidence"],
            "predicted_class": results["predicted_class"],
        }
    )

    if "true_labels" in results:
        dataframe["true_label"] = results["true_labels"]

    if image_paths is not None:
        if len(image_paths) != len(dataframe):
            raise ValueError(
                "image_paths must have one entry per prediction."
            )
        dataframe.insert(0, "image_path", image_paths)

    return dataframe


def save_uncertainty_results(
    results: dict[str, np.ndarray],
    output_path: str | Path,
    image_paths: list[str] | None = None,
) -> Path:
    """Save per-image uncertainty estimates as a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = create_uncertainty_dataframe(
        results,
        image_paths=image_paths,
    )
    dataframe.to_csv(output_path, index=False)

    logger.info("Saved uncertainty results to %s", output_path)
    return output_path