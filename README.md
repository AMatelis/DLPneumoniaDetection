# Pneumonia Detection from Chest X Rays
# YOLO badge test

A research-oriented deep learning project for detecting pneumonia from chest X ray images. Built to be extended, improved, and collaborated on.

This project goes beyond simple classification: it emphasizes model reliability, uncertainty estimation, explainability, and decision threshold optimization for real-world medical AI.

## Overview

• Classifies chest X ray images as NORMAL or PNEUMONIA

• Uses ResNet50 transfer learning with custom classification layers

• Evaluates with AUC, precision, recall, F1 score, and confusion matrices

• Generates Grad CAM heatmaps for explainability

• Estimates prediction uncertainty using Monte Carlo Dropout

• Optimizes decision thresholds instead of relying on a fixed 0.5 cutoff

• Supports training, evaluation, experiments, and deployment from a single script

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/DL-Pneumonia-Detection.git


2. Install dependencies:

pip install -r requirements.txt


3. Make sure you have Python 3.10+ and TensorFlow/Keras installed.

## Usage

#### Train the model

python pneumoniadetection.py --mode train --num_epochs 50


#### Evaluate with uncertainty

python pneumoniadetection.py --mode evaluate --use_mc_dropout


#### Run experiments

python pneumoniadetection.py --mode experiment --config config.yaml


#### Launch API

python pneumoniadetection.py --mode api


#### Launch dashboard

python pneumoniadetection.py --mode dashboard

## Project Structure
DL-Pneumonia-Detection/

├── data/            # train, val, test image folders

├── models/          # saved models

├── experiments/     # logs, configs, reports

├── outputs/         # plots and analysis 

├── pneumoniadetection.py

├── config.yaml

└── requirements.txt

## Contributing

• We welcome contributions! Good starting points:

• Improve model performance or training stability

• Add new explainability or calibration methods

• Test on new datasets or detect dataset shift

• Refactor code or improve documentation

• Add multi-class classification

• Improve the dashboard or API

Please open an issue first to discuss your proposal before submitting a pull request.

## Support & Discussion

• Open an issue for bugs

• Use GitHub Discussions for feature requests or ideas

## Who This Project Is For

• Students building ML or biomedical AI portfolios

• Researchers exploring medical imaging

• Developers interested in uncertainty-aware AI

• Anyone looking for a modular, research-grade ML project

## Disclaimer

This project is for research and educational use only. It is not a medical device and must not be used for clinical diagnosis.
