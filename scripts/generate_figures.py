
"""Generate publication-quality figures for a pneumonia detection experiment."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pneumonia.visualization import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication figures from experiment outputs."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Directory containing the experiment's saved results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()

    if not experiment_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}"
        )

    print(f"Generating figures from: {experiment_dir}")
    generate_all_figures(str(experiment_dir))
    print(f"Figure generation complete: {experiment_dir}")


if __name__ == "__main__":
    main()