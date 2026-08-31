# AI-Enhanced Intrusion Detection System (IDS)

## Overview

This dissertation project develops and evaluates an Explainability-Guided Feature Refinement Framework (EGFRF) for AI-based Intrusion Detection Systems.

The framework compares traditional signature-based intrusion detection using SNORT with multiple machine learning and deep learning models using the CICIDS2017 and UNSW-NB15 benchmark datasets.

The project also investigates whether explainability-guided feature refinement using SHAP can reduce the number of input features while maintaining strong intrusion detection performance and improving model interpretability.

## Objectives

- Compare traditional SNORT-based intrusion detection with AI-based IDS models.
- Evaluate the performance of machine learning and deep learning models for binary intrusion detection.
- Reduce the number of input features using SHAP-based feature importance.
- Compare model performance using all available numerical features against a refined set of the top 15 SHAP-ranked features.
- Investigate False Positive Rate (FPR) and False Negative Rate (FNR).
- Evaluate the effect of feature refinement on predictive performance and computational efficiency.
- Improve model explainability using SHAP and LIME.
- Evaluate model generalisation using training, validation, and testing partitions.

## Datasets

Two benchmark intrusion detection datasets are used:

### CICIDS2017

CICIDS2017 contains benign and malicious network traffic representing several modern attack scenarios.

For this project, the original labels are converted into a binary classification problem:

- `0` = Benign / Normal traffic
- `1` = Attack traffic

### UNSW-NB15

UNSW-NB15 contains normal network activity together with several categories of malicious traffic.

The dataset is also treated as a binary classification problem:

- `0` = Normal traffic
- `1` = Attack traffic

## Data Preprocessing

The preprocessing pipeline includes:

- Removal of exact duplicate records.
- Stratified sampling where required.
- Removal of identifier, label, and non-numerical columns.
- Conversion of infinite values into missing values.
- Removal of duplicate numerical feature vectors before dataset splitting.
- Stratified 70% training, 15% validation, and 15% testing split.
- Median imputation for missing values.
- Standardisation using `StandardScaler`.
- Fitting the imputer and scaler using the training data only to prevent data leakage.
- Verification that identical feature vectors do not overlap between the training and testing partitions.

## Models

The following AI models are evaluated:

- Logistic Regression
- Random Forest
- Feedforward Neural Network (FFNN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

A traditional SNORT signature-based IDS is also used as a baseline where suitable packet capture data is available.

## Explainability-Guided Feature Refinement

SHAP is used to estimate global feature importance.

An Extra Trees model is used as the SHAP feature-selection model, and features are ranked according to their mean absolute SHAP values.

The experiment compares two feature configurations:

- **All Features**
- **SHAP Refined Features — Top 15 Features**

The models are retrained using the refined feature set to evaluate whether comparable detection performance can be achieved with fewer input features.

## Explainability

Two explainability techniques are used:

### SHAP

SHAP provides global feature-level explanations and is used to rank the most influential network traffic features.

### LIME

LIME is used to generate local explanations for individual attack predictions, showing which refined features contributed to a particular classification.

## Experimental Setup

The main experiment is repeated using five random seeds:

- 42
- 43
- 44
- 45
- 46

Results are collected independently for each seed and summarised statistically.

The deep learning models use early stopping to reduce unnecessary training and limit overfitting.

An additional seed-42 validation experiment evaluates each model separately across:

- Training data
- Validation data
- Testing data

This analysis is used to examine model generalisation and identify potential differences between training and unseen-data performance.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- F2 Score
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Receiver Operating Characteristic Area Under the Curve (ROC-AUC)
- Training Time
- Inference Time

Confusion-matrix values are also recorded:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

## Statistical Evaluation

For the repeated experiments, the framework calculates:

- Mean performance
- Standard deviation
- 95% confidence intervals

An EGFRF ablation comparison is also performed to compare the change between all-feature and SHAP-refined models in terms of:

- Accuracy
- F1
- F2
- FPR
- FNR
- ROC-AUC
- Training time
- Inference time

## Repository Structure

```text
AI-IDS-Dissertation/
│
├── src/
│   └── IDS_V3/
│       │
│       ├── config.py
│       ├── main.py
│       │
│       ├── data/
│       │   ├── loader.py
│       │   └── preprocessing.py
│       │
│       ├── features/
│       │   └── shap_selection.py
│       │
│       ├── models/
│       │   └── classifiers.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── plots.py
│       │   └── partition_auc.py
│       │
│       ├── explainability/
│       │   └── lime_explainer.py
│       │
│       └── utils/
│           └── reproducibility.py
│
└── README.md
