
"""Tests for pneumonia model training utilities."""

import numpy as np
import pytest
import tensorflow as tf

from pneumonia.training import (
    set_random_seeds,
    calculate_class_weights,
    create_training_callbacks,
)


def test_set_random_seeds_is_reproducible():
    """Setting the same seed should reproduce TensorFlow random values."""
    set_random_seeds(42)
    first = tf.random.uniform((5,)).numpy()

    set_random_seeds(42)
    second = tf.random.uniform((5,)).numpy()

    np.testing.assert_allclose(first, second)


def test_calculate_class_weights_balanced_classes():
    """Balanced classes should receive equal weights."""
    class MockGenerator:
        classes = np.array([0, 0, 1, 1])

    weights = calculate_class_weights(MockGenerator())

    assert 0 in weights
    assert 1 in weights
    assert weights[0] == pytest.approx(weights[1])


def test_calculate_class_weights_imbalanced_classes():
    """The minority class should receive a higher weight."""
    class MockGenerator:
        classes = np.array([0, 0, 0, 0, 1, 1])

    weights = calculate_class_weights(MockGenerator())

    assert weights[1] > weights[0]


def test_calculate_class_weights_returns_positive_values():
    """All calculated class weights should be positive and finite."""
    class MockGenerator:
        classes = np.array([0, 0, 0, 1, 1])

    weights = calculate_class_weights(MockGenerator())

    assert all(np.isfinite(value) for value in weights.values())
    assert all(value > 0 for value in weights.values())


def test_calculate_class_weights_rejects_empty_classes():
    """An empty class array should not produce class weights."""
    class MockGenerator:
        classes = np.array([], dtype=int)

    with pytest.raises((ValueError, IndexError)):
        calculate_class_weights(MockGenerator())


def test_create_training_callbacks_returns_callbacks(tmp_path):
    """Callback creation should return usable Keras callbacks."""
    callbacks = create_training_callbacks(
        output_dir=tmp_path,
    )

    assert isinstance(callbacks, list)
    assert callbacks
    assert all(
        isinstance(callback, tf.keras.callbacks.Callback)
        for callback in callbacks
    )


def test_create_training_callbacks_includes_early_stopping(tmp_path):
    """Training callbacks should include early stopping."""
    callbacks = create_training_callbacks(
        output_dir=tmp_path,
    )

    assert any(
        isinstance(callback, tf.keras.callbacks.EarlyStopping)
        for callback in callbacks
    )


def test_create_training_callbacks_includes_learning_rate_reduction(tmp_path):
    """Training callbacks should reduce the learning rate when appropriate."""
    callbacks = create_training_callbacks(
        output_dir=tmp_path,
    )

    assert any(
        isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau)
        for callback in callbacks
    )


def test_create_training_callbacks_includes_checkpoint(tmp_path):
    """Training callbacks should save the best model checkpoint."""
    callbacks = create_training_callbacks(
        output_dir=tmp_path,
    )

    checkpoints = [
        callback
        for callback in callbacks
        if isinstance(callback, tf.keras.callbacks.ModelCheckpoint)
    ]

    assert checkpoints
    assert all(checkpoint.save_best_only for checkpoint in checkpoints)


def test_class_weights_give_equal_weight_to_single_class():
    """A single-class generator should be handled explicitly."""
    class MockGenerator:
        classes = np.array([0, 0, 0])

    # The implementation may reject a missing class or return a weight
    # for the observed class. Either behavior should be intentional.
    try:
        weights = calculate_class_weights(MockGenerator())
    except ValueError:
        return

    assert isinstance(weights, dict)
    assert 0 in weights
    assert weights[0] > 0