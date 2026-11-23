# 📉 Customer Churn Prediction – Telco Dataset

This project explores the IBM Telco Customer Churn dataset to identify key factors behind customer attrition and build a predictive machine learning model. The goal is to help telecom companies anticipate and reduce customer churn.

---

## 🎯 Objective

- Analyze customer behavior to uncover churn patterns
- Engineer features to improve model performance
- Train a classification model to predict churn risk
- Explain model behavior using SHAP values

---

## 🛠️ Tech Stack

- Python
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- imbalanced-learn (SMOTE)

---

## 📦 Dataset Overview

- **Total Records:** ~7,043
- **Source:** IBM Sample Dataset
- **Target Variable:** `Churn` (Yes/No)

Key features include:
- Demographics: gender, senior citizen, dependents
- Services: Internet, phone, security, streaming
- Contract & Billing: tenure, contract type, payment method
- Charges: monthly charges, total charges

---

## 🔍 Analysis Workflow

### 1️⃣ Data Cleaning & Preprocessing
- Converted `TotalCharges` from object to numeric
- Removed rows with missing values
- Mapped `SeniorCitizen` from 0/1 to `Yes`/`No`
- Created tenure groups (e.g., `1–12`, `13–24`, etc.)
- Dropped irrelevant column (`customerID`)

### 2️⃣ Exploratory Data Analysis (EDA)
- Univariate and bivariate analysis of features vs churn
- Key insights:
  - Highest churn in month-to-month contracts
  - Fiber optic users churn more than DSL users
  - Electronic check payments show high churn
  - Low tenure = high churn risk

### 3️⃣ Modeling – Random Forest Classifier (leakage-free evaluation)
- Stratified 70-30 train-test split **before** any resampling, so the test set keeps the real ~26.6% churn rate
- SMOTE applied to training folds only, inside an `imblearn` Pipeline (preprocessing + SMOTE + classifier)
- Decision threshold chosen on out-of-fold training predictions, never on the test set
- Test set used once; compared against majority-class and logistic-regression baselines
- See `churn_prediction_corrected.py` for the full pipeline

> **Correction note:** an earlier version of this notebook applied SMOTEENN to the full dataset *before* the train-test split. Synthetic and cleaned rows leaked into the test set, which inflated the reported accuracy/F1 (~95%). Those figures are not valid estimates of real-world performance and have been replaced by the results below.

## 📊 Key Results (held-out test set, n = 2,110; 561 churners)

| Model                        | Churn precision | Churn recall | Churn F1 | ROC-AUC | PR-AUC |
|------------------------------|-----------------|--------------|----------|---------|--------|
| Majority class               | 0.00            | 0.00         | 0.00     | –       | –      |
| Logistic regression + SMOTE  | 0.53            | 0.76         | 0.62     | 0.842   | 0.644  |
| Random forest + SMOTE        | 0.54            | 0.74         | 0.62     | 0.837   | 0.634  |

| Item | Finding |
|------|---------|
| Top drivers | Contract type, tenure group, monthly charges |
| Insight | Short-tenure users on month-to-month contracts with high bills are most likely to churn |
