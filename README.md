# Pneumonia Detection from Chest X Rays

An open, research oriented deep learning project for medical imaging. Built to be extended, improved, and challenged.

This repository focuses on detecting pneumonia from chest X ray images using modern deep learning techniques. It is not just about training a CNN and reporting accuracy. The emphasis is on reliability, interpretability, and real world usability.

If you are interested in medical AI, uncertainty aware models, or deployable ML systems, this project is meant for you.

## Why This Project Matters

Most pneumonia detection projects stop at classification performance. This one asks deeper questions:

How confident is the model in each prediction

What regions of the X ray influenced the decision

How should decision thresholds change in clinical settings

How does performance change under dataset shift

The goal is to provide a strong foundation for meaningful experimentation, collaboration, and real improvement.

## What This Project Does

Classifies chest X ray images as NORMAL or PNEUMONIA

Uses transfer learning with a ResNet50 backbone

Evaluates performance using AUC, precision, recall, F1 score, and confusion matrices

Generates Grad CAM heatmaps for model explainability

Estimates prediction uncertainty using Monte Carlo Dropout

Optimizes decision thresholds instead of relying on a fixed 0.5 cutoff

Supports training, evaluation, experiments, and deployment from a single script

Designed to be clean, modular, and easy to extend.

## Getting Started
Train the model
python pneumoniadetection.py --mode train --num_epochs 50

Evaluate with uncertainty estimation
python pneumoniadetection.py --mode evaluate --use_mc_dropout

Run structured experiments
python pneumoniadetection.py --mode experiment --config config.yaml

Launch the API
python pneumoniadetection.py --mode api

Launch the dashboard
python pneumoniadetection.py --mode dashboard

## Project Structure
DL-Pneumonia-Detection/
├── data/     # train, val, test image folders
├── models/              # saved models
├── experiments/         # logs, configs, reports
├── outputs/             # plots and analysis
├── pneumoniadetection.py
├── config.yaml
└── requirements.txt

## Ways to Contribute

Contributions are strongly encouraged. Some ideas:

Improve model performance or training stability

Add new explainability or calibration techniques

Test on additional datasets or analyze dataset shift

Refactor or improve documentation

Add multi class classification

Extend the dashboard or API

If you are not sure where to start, open an issue and describe what you want to work on.

## Who This Is For

Students building strong ML or biomedical portfolios

Researchers exploring medical imaging models

Developers interested in uncertainty aware AI

Contributors looking for a serious open source ML project

## Disclaimer

This project is for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis.
