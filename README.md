
# Pneumonia Detection from Chest X-Ray Images

A research-oriented deep learning project exploring pneumonia classification from chest X-ray images using transfer learning, model explainability, uncertainty estimation, and systematic evaluation.

The project is designed to support reproducible experimentation in biomedical AI, with an emphasis on understanding model predictions, evaluating reliability, and investigating generalization across datasets.

![Chest X-ray example](assets/chest-xray-example.png.jpeg)

## Overview

This project explores binary classification of chest X-ray images into `NORMAL` and `PNEUMONIA` categories using a ResNet50-based deep learning pipeline.

Key components include:

- **Transfer learning:** ResNet50 with custom classification layers.
- **Model evaluation:** ROC-AUC, precision, recall, F1 score, and confusion matrices.
- **Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM).
- **Uncertainty estimation:** Monte Carlo Dropout for predictive uncertainty analysis.
- **Decision-threshold optimization:** Analysis of classification thresholds beyond the default 0.5 cutoff.
- **Probability calibration:** Tools for examining the reliability of predicted probabilities.
- **Dataset-shift analysis:** Experiments investigating performance across different data sources.
- **Modular architecture:** Separate components for data processing, training, evaluation, visualization, and experimentation.

## Visualizations

The following figures illustrate the intended analysis and presentation workflow. Illustrative graphics are not substitutes for outputs generated from trained models and held-out evaluation data.

### Grad-CAM Explainability

Grad-CAM provides a visual representation of image regions that contribute to a model's prediction. It can help investigate model behavior, but highlighted regions do not establish that a prediction is medically correct or that the model relies on clinically meaningful features.

![Chest X-ray Grad-CAM Analysis](assets/Chest%20X-ray%20Grad-CAM%20Analysis.png)

### Model Evaluation

The evaluation workflow is designed to examine classification performance using complementary metrics and visualizations. ROC curves, confusion matrices, and summary metrics should be generated from actual model predictions on an appropriately held-out test set.

![Model Evaluation Dashboard](assets/Model%20Evaluation%20Dashboard.png)

**Research integrity:** Any illustrative predictions, curves, or placeholder metrics in these graphics must be replaced with genuine experimental outputs before being reported as research findings.

## Datasets

The project is configured to use two chest X-ray datasets.

### Dataset A: Chest X-Ray

- **Internal name:** `Archive/ChestXRay`
- **Expected location:** `data/archive/chest_xray/`
- **Purpose:** Primary chest X-ray dataset.

### Dataset B: Extra Dataset

- **Internal name:** `ExtraDataset`
- **Expected location:** `data/extra_dataset/`
- **Purpose:** Additional dataset for combined training and dataset-shift experiments.

The original dataset source URLs, exact versions, and licenses still need to be verified against the datasets used. They should be documented before publication or redistribution.

### Class Labels

The data pipeline expects two classes:

- `NORMAL`
- `PNEUMONIA`

### Local Directory Structure

Datasets are stored locally and are excluded from this repository.

```text
data/
├── archive/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
└── extra_dataset/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

Update `configs/default.yaml` if your local dataset paths differ from these defaults.

### Data Processing

The current data pipeline uses TensorFlow/Keras `ImageDataGenerator` to:

- Resize images to 224 × 224 pixels.
- Rescale pixel values by `1/255`.
- Apply training augmentation, including rotation, width and height shifts, shear, zoom, and horizontal flipping.
- Rescale validation images without applying training augmentations.

The baseline pipeline is configured to combine training and validation directories from both datasets. Dataset-shift experiments are intended to investigate performance when training and evaluation data come from different sources.

Before reporting experimental results, verify the dataset provenance, licensing, sample counts, patient-level overlap, and split methodology. Keep final test data separate from training, model selection, threshold optimization, and calibration.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AMatelis/DLPneumoniaDetection.git
cd DLPneumoniaDetection
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Alternatively, install the project in editable mode with development dependencies if supported by the current `pyproject.toml` configuration:

```bash
python -m pip install -e ".[dev]"
```

### 3. Verify the environment

Use Python 3.10 or later with a compatible TensorFlow version and the project's required dependencies.

## Usage

The repository includes entry-point scripts for training, evaluation, and figure generation. Confirm that each script is configured for your local datasets before running it.

### Train the model

```bash
python scripts/train.py
```

### Evaluate the model

```bash
python scripts/evaluate.py
```

### Generate research figures

```bash
python scripts/generate_figures.py
```

### Run tests

```bash
python -m pytest -v
```

Training and evaluation require the datasets to be available locally in the expected directory structure.

## Project Structure

```text
DLPneumoniaDetection/
├── assets/
│   ├── chest-xray-example.png.jpeg
│   ├── Chest X-ray Grad-CAM Analysis.png
│   └── Model Evaluation Dashboard.png
├── configs/
│   └── default.yaml
├── data/                     # Local datasets; excluded from Git
├── models/                   # Local model files
├── outputs/                  # Generated results and figures
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── generate_figures.py
├── src/
│   └── pneumonia/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── training.py
│       ├── evaluation.py
│       ├── uncertainty.py
│       ├── explainability.py
│       ├── calibration.py
│       ├── thresholds.py
│       ├── dataset_shift.py
│       ├── visualization.py
│       ├── experiments.py
│       ├── api.py
│       ├── dashboard.py
│       └── cli.py
├── tests/
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

## Research Capabilities

### Model Development and Evaluation

The project uses a pretrained ResNet50 backbone for binary classification and provides tools for training, evaluation, and visualization.

### Explainability

Grad-CAM visualizations can be used to investigate which image regions influence model predictions. These visualizations are exploratory and do not independently establish clinical validity.

### Predictive Uncertainty

Monte Carlo Dropout provides a framework for examining variation in model predictions across stochastic forward passes.

### Probability Calibration and Threshold Selection

Calibration tools assess whether predicted probabilities align with observed outcomes. Threshold-optimization tools support analysis of alternative classification operating points.

Calibration and threshold selection should use appropriate validation data rather than the final test set.

### Dataset Generalization

Dataset-shift experiments are intended to examine how performance changes across data sources. Reliable comparisons require appropriate data separation and consistent evaluation procedures.

## Research Integrity and Limitations

This repository documents a research-oriented software project, not a clinically validated diagnostic system.

- No performance figures should be interpreted as established results unless generated from the documented experimental pipeline.
- Test-set metrics should be reported only after evaluation on appropriately held-out data.
- Dataset sources, licensing, split methodology, and potential data leakage should be verified and documented.
- Grad-CAM visualizations and uncertainty estimates do not independently establish clinical reliability.
- Further validation would be necessary before any clinical application could be considered.

## Contributing

Contributions that improve reliability, reproducibility, and maintainability are welcome.

Potential areas for contribution include:

- Improving training stability and model evaluation.
- Adding explainability and probability-calibration methods.
- Evaluating generalization across datasets.
- Investigating dataset shift and potential data leakage.
- Expanding automated test coverage.
- Improving API, dashboard, and documentation functionality.

Please open an issue to discuss substantial changes before submitting a pull request.

## Support and Discussion

- **Bug reports:** Open a GitHub issue.
- **Feature requests:** Open an issue describing the proposed functionality.
- **Discussion:** Use GitHub Discussions if enabled for the repository.

## Intended Audience

- Students exploring machine learning and biomedical engineering.
- Researchers investigating deep learning for medical imaging.
- Developers studying uncertainty-aware machine learning.
- Contributors interested in reproducible medical AI research.

## Acknowledgments

I would like to thank Dr. Paul Yi, Associate Member of the Department of Radiology and Section Chief of Intelligent Imaging Informatics at St. Jude Children's Research Hospital, for his feedback on the project's methodology and radiology AI approaches. His guidance helped inform the methodological direction of this project.

## Disclaimer

This project is intended for research and educational purposes only. It is not a validated medical device and must not be used for clinical diagnosis or treatment decisions.
