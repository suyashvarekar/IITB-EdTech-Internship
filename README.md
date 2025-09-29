# Multimodal Prediction of Mental Rotation Accuracy

## Overview

This project is part of the IITB EdTech Internship 2025, Track 1 - Educational Data Analysis (EDA).
The goal is to build machine learning models that predict the correctness of participants’ responses in a mental rotation task using multimodal physiological and behavioral data.

The modalities used include:

* **EEG (brainwave signals)**
* **Eye-Tracking data**
* **GSR (Galvanic Skin Response)**
* **Facial expressions (Action Units / affective states)**

The system performs data preprocessing, feature engineering, model training, evaluation, and interpretability analysis.

---

## Problem Statement

* **Task**: Binary classification (Correct vs Incorrect response).
* **Inputs**: Features engineered from EEG, Eye, GSR, and Facial modalities.
* **Output**: Whether a participant’s response is correct (1) or incorrect (0).

---

## Dataset Structure

Data is organized per student in the following structure:

```
/content/drive/MyDrive/STData/
│── 1/
│   ├── 1_PSY.csv
│   ├── 1_EEG.csv
│   ├── 1_GSR.csv
│   ├── 1_EYE.csv
│   ├── 1_IVT.csv
│   └── 1_TIVA.csv
│── 2/
│   ├── 2_PSY.csv
│   ├── ...
│── ...
│── 38/
```

* **PSY.csv**: Contains trial information with `routineStart`, `routineEnd`, and `verdict` columns.
* **EEG.csv**: Contains brainwave signals or band power values.
* **GSR.csv**: Contains skin conductance/resistance data.
* **EYE.csv, IVT.csv**: Contain eye-tracking data (gaze, pupil size, fixation, saccades).
* **TIVA.csv**: Contains facial emotion or Action Unit (AU) features.

---

## Pipeline

### 1. Data Preparation

* Aligns multimodal data with question start and end times.
* Extracts per-question features by aggregating within each time window.

### 2. Feature Engineering

* **EEG**: Mean/variance of Delta, Theta, Alpha, Beta, Gamma bands.
* **GSR**: Mean, standard deviation, temporal slope.
* **Eye-Tracking**: Fixation duration, saccade amplitude, pupil size.
* **Facial Expressions**: Action Units and emotion intensities.

### 3. Preprocessing

* Synchronization of timestamps across modalities.
* Aggregation using mean, standard deviation, min, max, slope.
* Label encoding: Correct = 1, Incorrect = 0.
* Handling class imbalance using SMOTE.

### 4. Modeling Approaches

* **Baseline Models**: Logistic Regression, Random Forest, XGBoost.
* **Intermediate Fusion**: Train per-modality models, stack predictions with a meta-classifier.
* **Advanced (Optional)**: Sequence models (LSTM/BiLSTM) with attention for raw time-series.

### 5. Evaluation

* Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
* Confusion Matrix analysis.
* SHAP-based interpretability for feature importance.

### 6. Experiments

* Feature selection with PCA and mutual information.
* Modality dropout experiments to test contribution of each modality.
* Per-participant vs cross-participant generalization.

---

## File Structure

```
project/
├── data/
│   ├── PSY.csv
│   ├── EEG.csv
│   ├── GSR.csv
│   ├── EYE.csv, IVT.csv
│   ├── TIVA.csv
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_modeling_fusion.ipynb
│   └── 05_analysis.ipynb
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── stacking_meta.joblib
├── README.md
```

---

## Results

* Baseline models achieve competitive accuracy and F1-score.
* Stacking across modalities improves performance compared to individual models.
* SHAP analysis highlights key contributing features across EEG, Eye, GSR, and Facial data.

---

## Future Work

* Incorporate raw time-series deep learning (LSTM/Transformer).
* Explore early prediction before participant response is complete.
* Analyze differences between feedback vs no-feedback conditions.
* Add demographic covariates for personalized modeling.

---

## Requirements

* Python 3.10+
* Libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `matplotlib`, `seaborn`

Install dependencies in Colab:

```bash
pip install xgboost imbalanced-learn shap
```

---

## Usage

1. Upload dataset to Google Drive in the specified folder structure.
2. Open the notebook in Google Colab.
3. Run cells in order:

   * Mount Drive
   * Preprocessing & Feature Engineering
   * Modeling & Evaluation
   * Interpretability & Experiments
4. Trained models are saved in `/STData/models_export`.
