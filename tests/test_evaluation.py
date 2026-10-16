
"""Tests for pneumonia model evaluation."""

import json

import numpy as np
import pytest

from pneumonia.evaluation import (
    _classification_metrics,
    _optimize_threshold,
    _get_labels,
)


def test_classification_metrics_perfect_predictions():
    """Perfect predictions should produce perfect classification metrics."""
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.80, 0.95])

    results = _classification_metrics(labels, probabilities, threshold=0.5)

    assert results["accuracy"] == pytest.approx(1.0)
    assert results["balanced_accuracy"] == pytest.approx(1.0)
    assert results["precision"] == pytest.approx(1.0)
    assert results["recall"] == pytest.approx(1.0)
    assert results["specificity"] == pytest.approx(1.0)
    assert results["f1_score"] == pytest.approx(1.0)
    assert results["auc"] == pytest.approx(1.0)


def test_classification_metrics_incorrect_predictions():
    """Completely incorrect predictions should have zero accuracy."""
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.80, 0.90, 0.10, 0.20])

    results = _classification_metrics(labels, probabilities, threshold=0.5)

    assert results["accuracy"] == pytest.approx(0.0)
    assert results["balanced_accuracy"] == pytest.approx(0.0)


def test_classification_metrics_confusion_matrix():
    """The confusion matrix should match the predicted classifications."""
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.80, 0.70, 0.20])

    results = _classification_metrics(labels, probabilities, threshold=0.5)

    expected = np.array([
        [1, 1],
        [1, 1],
    ])

    np.testing.assert_array_equal(
        np.asarray(results["confusion_matrix"]),
        expected,
    )


def test_classification_metrics_includes_sensitivity_and_specificity():
    """Sensitivity and specificity should be present in the results."""
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.60, 0.80, 0.30])

    results = _classification_metrics(labels, probabilities, threshold=0.5)

    assert "sensitivity" in results
    assert "specificity" in results
    assert 0.0 <= results["sensitivity"] <= 1.0
    assert 0.0 <= results["specificity"] <= 1.0


def test_optimize_threshold_returns_valid_threshold():
    """Threshold optimization should return a threshold between 0 and 1."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.10, 0.20, 0.35, 0.60, 0.75, 0.90])

    threshold = _optimize_threshold(probabilities, labels)

    assert 0.0 <= threshold <= 1.0


def test_optimize_threshold_separates_classes():
    """A useful threshold should separate clearly distinct probabilities."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.15, 0.25, 0.75, 0.85, 0.95])

    threshold = _optimize_threshold(probabilities, labels)
    predictions = (probabilities >= threshold).astype(int)

    assert np.mean(predictions == labels) == pytest.approx(1.0)


def test_get_labels_returns_generator_labels():
    """Labels should be extracted from a generator's classes."""
    class MockGenerator:
        classes = np.array([0, 1, 0, 1])

    labels = _get_labels(MockGenerator())

    np.testing.assert_array_equal(labels, [0, 1, 0, 1])


def test_get_labels_resets_generator_if_supported():
    """Label extraction should reset a generator when reset is available."""
    class MockGenerator:
        classes = np.array([0, 1])
        reset_called = False

        def reset(self):
            self.reset_called = True

    generator = MockGenerator()
    _get_labels(generator)

    assert generator.reset_called


def test_evaluate_model_saves_results(tmp_path):
    """Evaluation should save its JSON results and prediction CSV."""
    import pandas as pd

    from pneumonia.evaluation import evaluate_model

    class MockGenerator:
        classes = np.array([0, 0, 1, 1])
        class_indices = {"NORMAL": 0, "PNEUMONIA": 1}
        samples = 4
        batch_size = 4
        filenames = [
            "NORMAL/a.png",
            "NORMAL/b.png",
            "PNEUMONIA/c.png",
            "PNEUMONIA/d.png",
        ]

        def reset(self):
            pass

    class MockModel:
        def predict(self, generator, verbose=0):
            return np.array([[0.1], [0.2], [0.8], [0.9]])

    results = evaluate_model(
        model=MockModel(),
        val_generator=MockGenerator(),
        exp_dir=tmp_path,
    )

    assert isinstance(results, dict)
    assert (tmp_path / "evaluation_results.json").exists()
    assert (tmp_path / "predictions.csv").exists()

    with (tmp_path / "evaluation_results.json").open(
        encoding="utf-8"
    ) as file:
        saved_results = json.load(file)

    assert "accuracy" in saved_results

    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert len(predictions) == 4