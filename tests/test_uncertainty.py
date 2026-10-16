
"""Tests for pneumonia prediction uncertainty utilities."""

import numpy as np
import pytest
import tensorflow as tf

from pneumonia.uncertainty import (
    binary_entropy,
    predictive_entropy,
    expected_entropy,
    predictive_variance,
    mutual_information,
    mc_dropout_predict,
    create_uncertainty_dataframe,
)


def test_binary_entropy_is_zero_for_certain_predictions():
    """Certain predictions should have near-zero entropy."""
    probabilities = np.array([0.0, 1.0])

    entropy = binary_entropy(probabilities)

    assert np.all(np.isfinite(entropy))
    np.testing.assert_allclose(entropy, 0.0, atol=1e-6)


def test_binary_entropy_is_maximum_at_half():
    """Binary entropy should be highest at probability 0.5."""
    probabilities = np.array([0.5])

    entropy = binary_entropy(probabilities)

    assert entropy[0] == pytest.approx(np.log(2), abs=1e-6)


def test_binary_entropy_is_symmetric():
    """Complementary probabilities should have equal entropy."""
    probabilities = np.array([0.2, 0.8])

    entropy = binary_entropy(probabilities)

    assert entropy[0] == pytest.approx(entropy[1])


def test_predictive_entropy_shape_and_range():
    """Predictive entropy should be finite and nonnegative."""
    probabilities = np.array([0.1, 0.5, 0.9])

    entropy = predictive_entropy(probabilities)

    assert entropy.shape == probabilities.shape
    assert np.all(np.isfinite(entropy))
    assert np.all(entropy >= 0.0)


def test_expected_entropy_for_identical_samples():
    """Expected entropy should match entropy when samples are identical."""
    samples = np.array([
        [0.2, 0.7],
        [0.2, 0.7],
        [0.2, 0.7],
    ])

    result = expected_entropy(samples)

    expected = binary_entropy(np.array([0.2, 0.7]))
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_predictive_variance_is_zero_for_identical_samples():
    """Identical predictions should have zero variance."""
    samples = np.array([
        [0.3, 0.8],
        [0.3, 0.8],
        [0.3, 0.8],
    ])

    variance = predictive_variance(samples)

    np.testing.assert_allclose(variance, 0.0, atol=1e-7)


def test_predictive_variance_is_positive_for_different_samples():
    """Different predictions should produce positive variance."""
    samples = np.array([
        [0.1, 0.8],
        [0.5, 0.4],
        [0.9, 0.2],
    ])

    variance = predictive_variance(samples)

    assert np.all(variance > 0.0)


def test_mutual_information_is_nonnegative():
    """Mutual information should be nonnegative, within numerical tolerance."""
    samples = np.array([
        [0.1, 0.9],
        [0.5, 0.5],
        [0.9, 0.1],
    ])

    result = mutual_information(samples)

    assert np.all(np.isfinite(result))
    assert np.all(result >= -1e-6)


def test_mutual_information_near_zero_for_identical_samples():
    """Identical Monte Carlo predictions should have near-zero mutual information."""
    samples = np.array([
        [0.25, 0.75],
        [0.25, 0.75],
        [0.25, 0.75],
    ])

    result = mutual_information(samples)

    np.testing.assert_allclose(result, 0.0, atol=1e-6)


@pytest.fixture
def dropout_model():
    """Create a small model with a dropout layer for testing."""
    inputs = tf.keras.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, activation="relu")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Sequential([
        inputs,
        tf.keras.Model(inputs, outputs),
    ])


def test_mc_dropout_predict_returns_expected_keys(dropout_model):
    """MC dropout should return predictions and uncertainty estimates."""
    images = np.random.default_rng(42).random((4, 8, 8, 3)).astype(
        np.float32
    )

    result = mc_dropout_predict(
        dropout_model,
        images,
        n_samples=3,
        batch_size=2,
    )

    expected_keys = {
        "mc_probabilities",
        "mean_probability",
        "predictive_entropy",
        "expected_entropy",
        "predictive_variance",
        "mutual_information",
        "confidence",
        "predicted_class",
    }

    assert expected_keys.issubset(result.keys())


def test_mc_dropout_predict_output_shapes(dropout_model):
    """Uncertainty outputs should correspond to the input images."""
    images = np.random.default_rng(1).random((4, 8, 8, 3)).astype(
        np.float32
    )

    result = mc_dropout_predict(
        dropout_model,
        images,
        n_samples=3,
        batch_size=2,
    )

    assert result["mean_probability"].shape == (4,)
    assert result["predictive_entropy"].shape == (4,)
    assert result["expected_entropy"].shape == (4,)
    assert result["predictive_variance"].shape == (4,)
    assert result["mutual_information"].shape == (4,)
    assert result["confidence"].shape == (4,)
    assert result["predicted_class"].shape == (4,)
    assert result["mc_probabilities"].shape == (3, 4)


def test_mc_dropout_probabilities_are_valid(dropout_model):
    """MC dropout probabilities should be finite and between zero and one."""
    images = np.random.default_rng(7).random((3, 8, 8, 3)).astype(
        np.float32
    )

    result = mc_dropout_predict(
        dropout_model,
        images,
        n_samples=3,
    )

    probabilities = result["mean_probability"]

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_create_uncertainty_dataframe():
    """Uncertainty results should be convertible to a DataFrame."""
    result = {
        "mean_probability": np.array([0.2, 0.8]),
        "predictive_entropy": np.array([0.5, 0.4]),
        "predictive_variance": np.array([0.01, 0.02]),
        "mutual_information": np.array([0.03, 0.04]),
        "confidence": np.array([0.8, 0.8]),
        "predicted_class": np.array([0, 1]),
    }

    dataframe = create_uncertainty_dataframe(result)

    assert len(dataframe) == 2
    assert "mean_probability" in dataframe.columns
    assert "predicted_class" in dataframe.columns