# 📉 Customer Churn Prediction – Telco Dataset

This project explores the IBM Telco Customer Churn dataset to identify key factors behind customer attrition and build a predictive machine learning model. The goal is to help telecom companies anticipate and reduce customer churn.

---

## 🎯 Objective

- Analyze customer behavior to uncover churn patterns
- Engineer features to improve model performance
- Train a classification model to predict churn risk
- Evaluate it honestly: split before resampling, tune the threshold without touching the test set, and compare against baselines

---

## 🛠️ Tech Stack

- Python
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- imbalanced-learn (SMOTE inside a `Pipeline`)

---

## 📦 Dataset Overview

- **Total Records:** 7,043 (7,032 after dropping 11 rows with blank `TotalCharges`)
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
- Removed the 11 rows with blank `TotalCharges`
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
- Stratified 70-30 train-test split **before** any resampling (`random_state=0`), so the test set keeps the real ~26.6% churn rate
- Preprocessing: `StandardScaler` on numeric columns, `OneHotEncoder` on categorical columns
- `SMOTE` applied to training folds only, inside an `imblearn` Pipeline (preprocessing → SMOTE → classifier)
- Random Forest: 300 trees, `min_samples_leaf=3`
- Decision threshold (0.397) chosen from 5-fold out-of-fold training predictions (best churn-class F1 0.629), never from the test set
- Test set used once; compared against a majority-class and a logistic-regression baseline
- Implemented in sections 8–9 of `churn_prediction.ipynb`

> **Correction note:** an earlier version of this notebook applied SMOTEENN to the full dataset *before* the train-test split. Synthetic and cleaned rows leaked into the test set, which inflated the reported accuracy/F1 (~95%). Those figures are not valid estimates of real-world performance and have been replaced by the results below.

## 📊 Key Results (held-out test set, n = 2,110; 561 churners)

| Model                        | Churn precision | Churn recall | Churn F1 | ROC-AUC | PR-AUC |
|------------------------------|-----------------|--------------|----------|---------|--------|
| Majority class               | 0.00            | 0.00         | 0.00     | –       | –      |
| Logistic regression + SMOTE  | 0.53            | 0.76         | 0.62     | 0.842   | 0.644  |
| Random forest + SMOTE        | 0.54            | 0.74         | 0.62     | 0.837   | 0.634  |

| Item | Finding |
|------|---------|
| Highest-churn segments (from EDA) | Month-to-month contracts, short tenure, higher monthly charges, fiber optic, electronic check |
| Insight | Short-tenure users on month-to-month contracts with high bills are most likely to churn |
| Model takeaway | Random Forest is no better than logistic regression on this dataset; recall of ~0.74 comes at ~0.54 precision |
