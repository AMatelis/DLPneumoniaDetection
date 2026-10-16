
"""Command-line interface for the pneumonia detection project."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="pneumonia",
        description="Pneumonia detection research pipeline.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands",
    )

    # Training command
    train_parser = subparsers.add_parser(
        "train",
        help="Train the pneumonia classification model.",
    )
    train_parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the YAML configuration file.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    train_parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name.",
    )
    train_parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_model.h5",
        help="Where to save the best model checkpoint.",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "training",
        help="Directory for training results.",
    )
    train_parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting during training.",
    )
    train_parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Enable decision-threshold optimization.",
    )
    train_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable probability calibration.",
    )
    train_parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Run quantitative Grad-CAM analysis.",
    )
    train_parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation after training.",
    )

    # Evaluation command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved model.",
    )
    evaluate_parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "best_model.h5",
        help="Path to the saved model.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation",
        help="Directory for evaluation results.",
    )
    evaluate_parser.add_argument(
        "--mc-dropout",
        action="store_true",
        help="Enable Monte Carlo dropout evaluation.",
    )
    evaluate_parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Optimize classification thresholds.",
    )
    evaluate_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Evaluate probability calibration.",
    )
    evaluate_parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Run quantitative Grad-CAM analysis.",
    )

    # Publication figures command
    figures_parser = subparsers.add_parser(
        "figures",
        help="Generate publication-quality figures.",
    )
    figures_parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Directory containing experiment results.",
    )

    return parser


def run_script(script_name: str, arguments: list[str]) -> int:
    """Run a project script using the current Python interpreter."""
    script_path = SCRIPTS_DIR / script_name

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required script not found: {script_path}"
        )

    # Importing runpy avoids depending on the user's working directory.
    import runpy

    original_argv = sys.argv[:]

    try:
        sys.argv = [str(script_path), *arguments]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = original_argv

    return 0


def main() -> int:
    """Parse arguments and dispatch to the requested command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs must be at least 1.")

        if not args.config.is_file():
            parser.error(f"Configuration file not found: {args.config}")

        forwarded = [
            "--config", str(args.config),
            "--epochs", str(args.epochs),
            "--model-path", str(args.model_path),
            "--output-dir", str(args.output_dir),
        ]

        if args.experiment_name:
            forwarded.extend(["--experiment-name", args.experiment_name])
        if args.no_class_weights:
            forwarded.append("--no-class-weights")
        if args.optimize_thresholds:
            forwarded.append("--optimize-thresholds")
        if args.calibrate:
            forwarded.append("--calibrate")
        if args.gradcam:
            forwarded.append("--gradcam")
        if args.skip_evaluation:
            forwarded.append("--skip-evaluation")

        return run_script("train.py", forwarded)

    if args.command == "evaluate":
        if not args.model.is_file():
            parser.error(f"Model file not found: {args.model}")

        forwarded = [
            "--model", str(args.model),
            "--output-dir", str(args.output_dir),
        ]

        if args.mc_dropout:
            forwarded.append("--mc-dropout")
        if args.optimize_thresholds:
            forwarded.append("--optimize-thresholds")
        if args.calibrate:
            forwarded.append("--calibrate")
        if args.gradcam:
            forwarded.append("--gradcam")

        return run_script("evaluate.py", forwarded)

    if args.command == "figures":
        if not args.experiment_dir.is_dir():
            parser.error(
                f"Experiment directory not found: {args.experiment_dir}"
            )

        return run_script(
            "generate_figures.py",
            ["--experiment-dir", str(args.experiment_dir)],
        )

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())