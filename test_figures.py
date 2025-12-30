"""
Test script for publication figure generation functions.
Creates mock data and tests each figure generation function.
"""

import numpy as np
import pandas as pd
import json
import os
import tempfile
from generate_figures import (
    setup_publication_style, plot_roc_with_ci, plot_uncertainty_vs_error,
    plot_gradcam_grid, plot_calibration_curves, plot_results_summary
)

def create_mock_data():
    """Create mock data for testing figure generation."""
    np.random.seed(42)

    # Mock predictions data
    n_samples = 200
    y_true = np.random.randint(0, 2, n_samples)
    y_pred_prob = np.random.beta(2, 2, n_samples)  # Realistic probabilities
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Mock uncertainty data
    uncertainties = np.random.beta(1, 3, n_samples)  # Higher for incorrect predictions

    # Create temporary directory structure
    temp_dir = tempfile.mkdtemp()
    outputs_dir = os.path.join(temp_dir, "outputs")
    plots_dir = os.path.join(temp_dir, "plots")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Save mock predictions CSV
    predictions_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability': y_pred_prob,
        'uncertainty_entropy': uncertainties
    })
    predictions_csv = os.path.join(outputs_dir, "test_validation_predictions.csv")
    predictions_df.to_csv(predictions_csv, index=False)

    # Create mock Grad-CAM stats
    gradcam_stats = {
        "correct_predictions": {
            "count": 75,
            "statistics": {
                "mean_entropy": 0.234,
                "mean_activation_lung": 0.456,
                "mean_activation_non_lung": 0.123,
                "mean_activation_ratio_lung_vs_nonlung": 3.712,
                "mean_focal_specificity": 0.678,
                "std_activation_overall": 0.089
            }
        },
        "incorrect_predictions": {
            "count": 25,
            "statistics": {
                "mean_entropy": 0.456,
                "mean_activation_lung": 0.234,
                "mean_activation_non_lung": 0.345,
                "mean_activation_ratio_lung_vs_nonlung": 0.678,
                "mean_focal_specificity": 0.234,
                "std_activation_overall": 0.145
            }
        }
    }
    gradcam_json = os.path.join(outputs_dir, "test_gradcam_stats.json")
    with open(gradcam_json, 'w') as f:
        json.dump(gradcam_stats, f, indent=4)

    # Create mock calibration stats
    calibration_stats = {
        "uncalibrated": {
            "ece": 0.087,
            "bin_metrics": [
                {"avg_confidence": 0.1, "accuracy": 0.15, "bin_size": 20},
                {"avg_confidence": 0.3, "accuracy": 0.35, "bin_size": 25},
                {"avg_confidence": 0.5, "accuracy": 0.48, "bin_size": 30},
                {"avg_confidence": 0.7, "accuracy": 0.72, "bin_size": 28},
                {"avg_confidence": 0.9, "accuracy": 0.88, "bin_size": 22}
            ]
        },
        "temperature_scaling": {
            "ece": 0.034,
            "temperature": 1.234,
            "bin_metrics": [
                {"avg_confidence": 0.1, "accuracy": 0.12, "bin_size": 20},
                {"avg_confidence": 0.3, "accuracy": 0.32, "bin_size": 25},
                {"avg_confidence": 0.5, "accuracy": 0.51, "bin_size": 30},
                {"avg_confidence": 0.7, "accuracy": 0.71, "bin_size": 28},
                {"avg_confidence": 0.9, "accuracy": 0.91, "bin_size": 22}
            ]
        },
        "platt_scaling": {
            "ece": 0.028,
            "A": 1.056,
            "B": -0.123,
            "bin_metrics": [
                {"avg_confidence": 0.1, "accuracy": 0.11, "bin_size": 20},
                {"avg_confidence": 0.3, "accuracy": 0.31, "bin_size": 25},
                {"avg_confidence": 0.5, "accuracy": 0.52, "bin_size": 30},
                {"avg_confidence": 0.7, "accuracy": 0.73, "bin_size": 28},
                {"avg_confidence": 0.9, "accuracy": 0.92, "bin_size": 22}
            ]
        },
        "calibration_data_info": {
            "calibration_set_size": 100,
            "test_set_size": 100,
            "total_validation_size": 200
        }
    }
    calib_json = os.path.join(outputs_dir, "test_calibration_analysis.json")
    with open(calib_json, 'w') as f:
        json.dump(calibration_stats, f, indent=4)

    # Create mock metadata
    metadata = {
        "experiment_name": "TestPublicationFigures",
        "experiment_version": "v1.0.0",
        "timestamp": "2024-12-28T14:30:15",
        "random_seed": 42
    }
    metadata_json = os.path.join(outputs_dir, "metadata.json")
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=4)

    return temp_dir, y_true, y_pred_prob, uncertainties

def test_figure_generation():
    """Test all figure generation functions."""
    print("Testing publication figure generation...")

    # Create mock data
    temp_dir, y_true, y_pred_prob, uncertainties = create_mock_data()
    plots_dir = os.path.join(temp_dir, "plots")

    try:
        # Test Figure 1: ROC with CI
        print("Testing ROC curve with confidence intervals...")
        plot_roc_with_ci(y_true, y_pred_prob, output_dir=plots_dir, filename="test_fig1_roc")

        # Test Figure 2: Uncertainty vs Error
        print("Testing uncertainty vs error analysis...")
        y_pred = (y_pred_prob > 0.5).astype(int)
        plot_uncertainty_vs_error(y_true, y_pred, uncertainties,
                                 output_dir=plots_dir, filename="test_fig2_uncertainty")

        # Test Figure 3: Grad-CAM Grid
        print("Testing Grad-CAM comparison grid...")
        gradcam_json = os.path.join(temp_dir, "outputs", "test_gradcam_stats.json")
        plot_gradcam_grid(gradcam_json, output_dir=plots_dir, filename="test_fig3_gradcam")

        # Test Figure 5: Calibration Curves
        print("Testing calibration curves...")
        calib_json = os.path.join(temp_dir, "outputs", "test_calibration_analysis.json")
        plot_calibration_curves(calib_json, output_dir=plots_dir, filename="test_fig5_calibration")

        # Test Figure 6: Results Summary
        print("Testing results summary...")
        plot_results_summary(temp_dir, output_dir=plots_dir, filename="test_fig6_summary")

        # Check generated files
        generated_files = []
        for root, dirs, files in os.walk(plots_dir):
            for file in files:
                if file.endswith(('.png', '.pdf')):
                    generated_files.append(os.path.join(root, file))

        print(f"\n✓ Generated {len(generated_files)} figure files:")
        for f in generated_files:
            print(f"  - {os.path.basename(f)}")

        print(f"\n✓ All figures saved to: {plots_dir}")
        return True

    except Exception as e:
        print(f"✗ Figure generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    success = test_figure_generation()
    if success:
        print("\n🎉 All publication figure generation tests passed!")
    else:
        print("\n❌ Some figure generation tests failed.")