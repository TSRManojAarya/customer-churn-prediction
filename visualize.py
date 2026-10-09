"""Charts for the churn project: EDA + best-model evaluation.

Reads the real Telco data, reproduces the exact train/test split from
data_utils (same seed), scores the saved model and writes PNGs to charts/.

Run:
    python train.py        # first, so models/churn_model.joblib exists
    python visualize.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from data_utils import (
    MODEL_PATH,
    build_preprocessor,
    clean,
    get_features,
    load_raw,
    train_test_frames,
)

CHART_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHART_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150})

df = clean(load_raw())
X, y = get_features(df)
X_train, X_test, y_train, y_test = train_test_frames(X, y)

# --- 1. Churn distribution ---------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["churn"].value_counts().sort_index()
ax.bar(["Stayed (0)", "Churned (1)"], counts.values, color=["#2ca02c", "#d62728"])
ax.set_title("Customer churn distribution (n=7,043)")
ax.set_ylabel("Customers")
for i, v in enumerate(counts.values):
    ax.text(i, v + 40, f"{v:,}\n({v / len(df):.1%})", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "churn_distribution.png"))
plt.close(fig)

# --- 2. Churn rate by contract type -------------------------------------------
rate = df.groupby("Contract")["churn"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(rate.index, rate.values, color="#1f77b4")
ax.set_title("Churn rate by contract type")
ax.set_ylabel("Churn rate")
ax.set_ylim(0, max(rate.values) * 1.25)
for i, v in enumerate(rate.values):
    ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "churn_by_contract.png"))
plt.close(fig)

# --- 3. Tenure and monthly charges by churn label ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, title in zip(
    axes,
    ["tenure", "MonthlyCharges"],
    ["Tenure (months) by churn label", "Monthly charges ($) by churn label"],
):
    for label, color in [(0, "#2ca02c"), (1, "#d62728")]:
        ax.hist(
            df.loc[df["churn"] == label, col],
            bins=30,
            alpha=0.55,
            label=f"churn={label}",
            color=color,
        )
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Customers")
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "tenure_vs_charges.png"))
plt.close(fig)

# --- 4. Correlation heatmap (numeric features) ----------------------------------
num = df.select_dtypes(exclude="object")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation heatmap (numeric features + churn)")
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "correlation_heatmap.png"))
plt.close(fig)

# --- 5 & 6. Best-model confusion matrix + ROC -----------------------------------
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    model_name = "saved best model"
else:
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    model_name = "logistic-regression fallback (train.py not run yet)"

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=["Predicted stay", "Predicted churn"],
    yticklabels=["Actual stay", "Actual churn"],
)
ax.set_title(f"Confusion matrix — {model_name}")
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "confusion_matrix.png"))
plt.close(fig)

fpr, tpr, _ = roc_curve(y_test, proba)
auc = roc_auc_score(y_test, proba)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
ax.set_title(f"ROC curve — {model_name}")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "roc_curve.png"))
plt.close(fig)

print(f"Saved 6 charts to {CHART_DIR}/")
