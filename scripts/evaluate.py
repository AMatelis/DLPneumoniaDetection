
"""Evaluate a trained pneumonia detection model."""

import argparse
import sys
from pathlib import Path

from tensorflow.keras.models import load_model


# Make the project's src/ directory importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pneumonia.data import create_data_generators
from pneumonia.evaluation import evaluate_model
from pneumonia.explainability import analyze_gradcam_quantitative


class DummyHistory:
    """Provide the history interface expected by evaluate_model."""

    def __init__(self):
        self.history = {
            "accuracy": [],
            "val_accuracy": [],
            "auc": [],
            "val_auc": [],
        }


def parse_args():
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained pneumonia detection model."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_model.h5",
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation",
        help="Directory for evaluation results.",
    )
    parser.add_argument(
        "--mc-dropout",
        action="store_true",
        help="Enable Monte Carlo dropout uncertainty analysis.",
    )
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Enable classification threshold optimization.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable probability calibration analysis.",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Run quantitative Grad-CAM analysis.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            "Train a model first or pass its location with --model."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print("Creating data generators...")
    _, val_generator = create_data_generators()

    print("Running evaluation...")
    evaluate_model(
        model,
        val_generator,
        DummyHistory(),
        use_mc_dropout=args.mc_dropout,
        optimize_thresholds=args.optimize_thresholds,
        calibrate=args.calibrate,
        exp_dir=str(args.output_dir),
    )

    if args.gradcam:
        print("Running Grad-CAM analysis...")
        analyze_gradcam_quantitative(
            model,
            val_generator,
            num_samples=50,
            exp_dir=str(args.output_dir),
        )

    print(f"Evaluation finished. Results directory: {args.output_dir}")


if __name__ == "__main__":
    main()