"""Customer Churn Prediction — training pipeline (real Telco data).

Dataset: IBM Telco Customer Churn (7,043 customers, 21 columns).
Source: https://github.com/IBM/telco-customer-churn-on-icp4d
Local copy: data/Telco-Customer-Churn.csv

Pipeline:
    load -> clean -> train/test split -> 3-model comparison
    -> 5-fold cross-validation -> holdout evaluation -> save best model.

Run:
    python train.py
"""

import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_utils import (
    MODEL_PATH,
    RESULTS_PATH,
    build_preprocessor,
    clean,
    get_features,
    load_raw,
    train_test_frames,
)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

df = clean(load_raw())
print(f"Loaded {len(df):,} customers x {df.shape[1]} columns after cleaning.")
print(f"Churn rate: {df['churn'].mean():.1%} ({int(df['churn'].sum()):,} churners)")

X, y = get_features(df)
preprocessor = build_preprocessor(X)

# Churn is imbalanced (~27% positive), so the linear/tree baselines use
# class_weight="balanced" and every split is stratified.
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

X_train, X_test, y_train, y_test = train_test_frames(X, y)
print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,} (stratified)")

results = []
best_name, best_model, best_f1 = None, None, -1.0

for name, estimator in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    row = {
        "model": name,
        "cv_f1_mean": round(float(cv_f1), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }
    results.append(row)
    print(row)
    if row["f1"] > best_f1:
        best_f1, best_name, best_model = row["f1"], name, pipe

result_df = pd.DataFrame(results).sort_values("f1", ascending=False)
print("\n" + result_df.to_string(index=False))
print(f"\nBest model by holdout F1: {best_name} ({best_f1:.4f})")

joblib.dump(best_model, MODEL_PATH)
result_df.to_csv(RESULTS_PATH, index=False)
print(f"Saved {MODEL_PATH} and {RESULTS_PATH}")
