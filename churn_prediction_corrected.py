"""Telco customer churn: leakage-free evaluation.

Fixes vs. the original notebook (churn_prediction.ipynb):
  1. Split FIRST (stratified), resample ONLY the training folds. The original
     ran SMOTEENN on the full data and then split, so synthetic rows and
     ENN-cleaned rows leaked into the test set.
  2. Resampling lives inside an imblearn Pipeline, so cross-validation
     resamples each training fold only.
  3. The test set keeps the real class balance (~26.5% churn) and is touched
     once, after the model and decision threshold are fixed.
  4. The decision threshold is chosen on out-of-fold training predictions,
     never on the test set.
  5. Reports churn-class precision / recall / F1, ROC-AUC and PR-AUC, with a
     majority-class and a logistic-regression baseline for context.

Usage: python churn_prediction_corrected.py path/to/Telco_Customer_Churn.csv
"""
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 0

# ---------------------------------------------------------------- load/clean
df = pd.read_csv(sys.argv[1])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna().drop(columns=["customerID"])  # 11 blank TotalCharges rows
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])
print(f"rows={len(df)}  churn rate={y.mean():.3%}")

# ------------------------------------------------- split FIRST, before any fit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])


def make(model, resample=True):
    steps = [("prep", prep)]
    if resample:
        steps.append(("smote", SMOTE(random_state=SEED)))  # train folds only
    steps.append(("clf", model))
    return Pipeline(steps)


rf = make(RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                 random_state=SEED, n_jobs=-1))
lr = make(LogisticRegression(max_iter=2000), resample=True)
majority = make(DummyClassifier(strategy="most_frequent"), resample=False)

# ----------------------- threshold from OUT-OF-FOLD TRAIN predictions only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = cross_val_predict(rf, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
threshold = float(thr[np.argmax(f1s)])
print(f"\nthreshold chosen on out-of-fold train predictions: {threshold:.3f}")
print(f"5-fold OOF train F1 (churn class): {f1s.max():.3f}")

# ------------------------------------------------ final fit, ONE test look
for name, model in [("Majority baseline", majority),
                    ("Logistic regression + SMOTE", lr),
                    ("Random forest + SMOTE", rf)]:
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    cut = threshold if model is rf else 0.5
    pred = (proba >= cut).astype(int)
    print(f"\n=== {name} (test set, real class balance, n={len(y_test)}) ===")
    print(f"threshold={cut:.3f}  churn-class F1={f1_score(y_test, pred, zero_division=0):.3f}")
    if name != "Majority baseline":
        print(f"ROC-AUC={roc_auc_score(y_test, proba):.3f}  "
              f"PR-AUC={average_precision_score(y_test, proba):.3f}")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"],
                                zero_division=0))
