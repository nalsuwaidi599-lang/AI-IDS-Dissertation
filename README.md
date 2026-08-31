# AI-Enhanced Intrusion Detection System (IDS)

## Overview

This dissertation project develops and evaluates an **Explainability-Guided Feature Refinement Framework (EGFRF)** for AI-based Intrusion Detection Systems (IDS).

The framework evaluates machine learning and deep learning models using the **CICIDS2017** and **UNSW-NB15** benchmark datasets. It also compares AI-based intrusion detection with the traditional signature-based **SNORT IDS** where suitable packet capture data is available.

The project investigates whether **SHAP-guided feature refinement** can reduce the number of input features while maintaining strong intrusion detection performance and improving model interpretability.

---

## Objectives

The main objectives of this project are to:

- Compare traditional SNORT-based intrusion detection with AI-based IDS models.
- Evaluate machine learning and deep learning approaches for binary intrusion detection.
- Reduce the number of input features using SHAP-based feature importance.
- Compare models trained using all numerical features with models trained using the top 15 SHAP-ranked features.
- Evaluate False Positive Rate (FPR) and False Negative Rate (FNR).
- Evaluate the effect of feature refinement on predictive performance and computational efficiency.
- Improve model explainability using SHAP and LIME.
- Examine model generalisation across training, validation, and testing data.

---

## Datasets

Two benchmark intrusion detection datasets are used.

### CICIDS2017

CICIDS2017 contains benign and malicious network traffic representing multiple attack scenarios.

The labels are converted into binary classes:

- `0` = Benign / Normal
- `1` = Attack

### UNSW-NB15

UNSW-NB15 contains normal network traffic together with different categories of malicious traffic.

The dataset is also treated as a binary classification problem:

- `0` = Normal
- `1` = Attack

---

## Data Preprocessing

The preprocessing pipeline includes:

- Removal of exact duplicate records.
- Stratified sampling where required.
- Removal of identifier, label, and non-numerical columns.
- Conversion of infinite values into missing values.
- Removal of duplicate numerical feature vectors before dataset splitting.
- Stratified **70% training, 15% validation, and 15% testing** split.
- Median imputation for missing values.
- Feature standardisation using `StandardScaler`.
- Fitting the imputer and scaler using the training data only to prevent data leakage.
- Verification that identical feature vectors do not overlap between the training and testing partitions.

---

## Models

The following AI models are evaluated:

- Logistic Regression
- Random Forest
- Feedforward Neural Network (FFNN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

A traditional **SNORT signature-based IDS** is also used as a baseline where suitable packet capture data is available.

---

## Explainability-Guided Feature Refinement

SHAP is used to estimate global feature importance.

An **Extra Trees classifier** is used as the SHAP feature-selection model. Features are ranked according to their mean absolute SHAP values.

The experiment compares two feature configurations:

1. **All Features**
2. **SHAP Refined Features — Top 15 Features**

The models are retrained using the refined feature set to determine whether strong intrusion detection performance can be maintained using fewer input features.

---

## Explainability

Two explainable AI techniques are used.

### SHAP

SHAP provides global feature-level explanations and is used to rank the most influential network traffic features.

### LIME

LIME provides local explanations for individual predictions and is used to demonstrate which refined features contributed to the classification of a specific network record.

---

## Experimental Setup

The main experiment is repeated using five random seeds:

- `42`
- `43`
- `44`
- `45`
- `46`

Results from the repeated experiments are used to calculate the mean, standard deviation, and 95% confidence intervals.

The CNN and LSTM models use early stopping to reduce unnecessary training and limit overfitting.

### Seed-42 Partition Analysis

An additional experiment using **seed 42** evaluates model performance separately across:

- Training data
- Validation data
- Testing data

ROC-AUC is calculated independently for each partition.

This additional analysis is used to examine model generalisation and identify differences between performance on training data and unseen validation/testing data.

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- F2 Score
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- ROC-AUC
- Training Time
- Inference Time

The following confusion-matrix values are also recorded:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

---

## Statistical Evaluation

For the repeated experiments, the framework calculates:

- Mean performance
- Standard deviation
- 95% confidence intervals

An EGFRF ablation comparison is also performed between the **All Features** and **SHAP Refined** configurations.

The comparison examines changes in:

- Accuracy
- F1 Score
- F2 Score
- FPR
- FNR
- ROC-AUC
- Training time
- Inference time

---

## Experimental Pipeline

```text
Dataset Loading
       ↓
Duplicate Removal and Data Cleaning
       ↓
Numerical Feature Preparation
       ↓
70% / 15% / 15% Stratified Split
       ↓
Median Imputation
       ↓
Standardisation
       ↓
Data Leakage Verification
       ↓
All-Feature Model Training
       ↓
SHAP Feature Ranking
       ↓
Top-15 Feature Refinement
       ↓
Refined Model Training
       ↓
Five-Seed Evaluation (42–46)
       ↓
Statistical Evaluation
       ↓
EGFRF Ablation Analysis
       ↓
ROC-AUC Evaluation
       ↓
Seed-42 Train / Validation / Test AUC Analysis
       ↓
LIME Local Explanation
```

---

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
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── preprocessing.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   └── shap_selection.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── classifiers.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── plots.py
│       │   └── partition_auc.py
│       │
│       ├── explainability/
│       │   ├── __init__.py
│       │   └── lime_explainer.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── reproducibility.py
│
└── README.md
```

---

## Technologies

The implementation uses:

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- SHAP
- LIME
- SNORT

---

## Reproducibility

The framework records experimental information including:

- Dataset used
- Source data files
- Number of records
- Duplicate removal statistics
- Training, validation, and testing sample counts
- Train-test identical feature overlap
- Number of features before and after refinement
- Feature reduction percentage
- Random seeds
- Python and library versions
- Execution platform
- GPU availability

Experimental results are exported to CSV and JSON files to support reproducibility and further analysis.

---

## Author

**Noor Jassim Al-Suwaidi**

MSc Cyber Security Dissertation

---

## Status

**Dissertation implementation completed.**

This repository contains the modular implementation of the experimental framework used for the dissertation.
