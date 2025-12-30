Pneumonia Detection from Chest X Rays

A clean, research oriented deep learning project for detecting pneumonia from chest X ray images. This repository is meant to be built on, not just read. If you are interested in medical AI, model reliability, or deployment, this is a good place to contribute.

The model uses transfer learning with ResNet50 and focuses on more than just accuracy. It includes uncertainty estimation, explainability tools, and decision threshold analysis, which are all important in real medical settings.

What This Project Does

Classifies chest X ray images as NORMAL or PNEUMONIA

Trains using transfer learning with ResNet50

Evaluates using AUC, precision, recall, F1 score, and confusion matrices

Generates Grad CAM heatmaps to show what the model is focusing on

Estimates prediction uncertainty using Monte Carlo Dropout

Optimizes decision thresholds instead of using a fixed 0.5 cutoff

Supports experiments, evaluation, and deployment from one script

This is designed to be readable, modular, and easy to extend.

Why This Repo Exists

Most pneumonia detection projects stop at training a CNN and reporting accuracy. This one tries to answer harder questions:

How confident is the model in its predictions

What parts of the image influenced the decision

How should thresholds change in a clinical setting

How does performance change under dataset shifts

The goal is to make this a strong base for experimentation, research, and real improvements.

Getting Started
Train the model
python pneumoniadetection.py --mode train --num_epochs 50

Evaluate with uncertainty
python pneumoniadetection.py --mode evaluate --use_mc_dropout

Run structured experiments
python pneumoniadetection.py --mode experiment --config config.yaml

Launch the API
python pneumoniadetection.py --mode api

Launch the dashboard
python pneumoniadetection.py --mode dashboard

Project Structure
DL-Pneumonia-Detection/
├── data/                # train, val, test image folders
├── models/              # saved models
├── experiments/         # logs, configs, reports
├── outputs/             # plots and analysis
├── pneumoniadetection.py
├── config.yaml
└── requirements.txt

Ways to Contribute

Contributions are very welcome. Some good starting points:

Improve model performance or training stability

Add new explainability or calibration methods

Test on new datasets or detect dataset shift

Clean up code structure or documentation

Add multi class classification

Improve the dashboard or API

If you are unsure where to start, open an issue and describe what you want to work on.

Who This Is For

Students building serious ML or biomedical portfolios

Researchers experimenting with medical imaging models

Developers interested in uncertainty aware AI

Anyone who wants to contribute to an open, well structured ML project

Disclaimer

This project is for research and educational use only. It is not a medical device and should not be used for clinical diagnosis.
