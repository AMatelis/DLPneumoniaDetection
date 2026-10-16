
"""Tests for probability calibration."""

import numpy as np
import pytest

from pneumonia.calibration import (
    fit_probability_calibrator,
    apply_probability_calibrator,
    evaluate_calibration,
)


@pytest.fixture
def sample_data():
    """Provide sample probabilities and binary labels."""
    probabilities = np.array(
        [0.10, 0.20, 0.35, 0.45, 0.55, 0.65, 0.80, 0.90]
    )
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    return probabilities, labels


def test_fit_probability_calibrator(sample_data):
    """The calibrator should fit and return a fitted estimator."""
    probabilities, labels = sample_data

    calibrator = fit_probability_calibrator(
        probabilities,
        labels,
    )

    assert calibrator is not None
    assert hasattr(calibrator, "predict_proba")


def test_apply_probability_calibrator(sample_data):
    """Calibrated probabilities should be finite and in [0, 1]."""
    probabilities, labels = sample_data

    calibrator = fit_probability_calibrator(
        probabilities,
        labels,
    )
    calibrated = apply_probability_calibrator(
        calibrator,
        probabilities,
    )

    assert calibrated.shape == probabilities.shape
    assert np.all(np.isfinite(calibrated))
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_evaluate_calibration_returns_expected_metrics(sample_data):
    """Calibration evaluation should return its core metrics."""
    probabilities, labels = sample_data

    results = evaluate_calibration(
        probabilities,
        labels,
        n_bins=4,
    )

    assert "brier_score" in results
    assert "log_loss" in results
    assert "mean_predicted_value" in results
    assert "fraction_of_positives" in results

    assert 0.0 <= results["brier_score"] <= 1.0
    assert np.isfinite(results["log_loss"])


def test_evaluate_calibration_bin_arrays_match(sample_data):
    """Each calibration bin should have a paired predicted and observed value."""
    probabilities, labels = sample_data

    results = evaluate_calibration(
        probabilities,
        labels,
        n_bins=4,
    )

    predicted = np.asarray(results["mean_predicted_value"])
    observed = np.asarray(results["fraction_of_positives"])

    assert predicted.ndim == 1
    assert observed.ndim == 1
    assert len(predicted) == len(observed)
    assert len(predicted) > 0


def test_calibrator_rejects_mismatched_lengths():
    """Mismatched probabilities and labels should raise an error."""
    probabilities = np.array([0.2, 0.8, 0.4])
    labels = np.array([0, 1])

    with pytest.raises((ValueError, IndexError)):
        fit_probability_calibrator(probabilities, labels)


def test_calibrator_rejects_single_class_labels():
    """Calibration fitting requires positive and negative examples."""
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 0, 0])

    with pytest.raises(ValueError):
        fit_probability_calibrator(probabilities, labels)