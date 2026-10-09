"""Shared data loading / cleaning for the churn project.

All entry points (train.py, visualize.py, app.py) use these helpers so the
cleaning logic lives in exactly one place.
"""

import os

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "churn_model.joblib")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "model_results.csv")

DATASET_URL = "https://github.com/IBM/telco-customer-churn-on-icp4d"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw() -> pd.DataFrame:
    """Load the raw Telco CSV (raises a helpful error if missing)."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            f"Download 'Telco-Customer-Churn.csv' from {DATASET_URL} "
            "and place it in the data/ folder."
        )
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw Telco frame: numeric TotalCharges, binary target, no IDs."""
    df = df.copy()
    # TotalCharges ships with blank strings for brand-new customers (tenure = 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["customerID", "Churn"])


def get_features(df: pd.DataFrame):
    """Split a cleaned frame into X (features) and y (target)."""
    return df.drop(columns="churn"), df["churn"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median/scale numeric columns, impute/one-hot encode categoricals."""
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )


def train_test_frames(X: pd.DataFrame, y: pd.Series):
    """Reproducible stratified split shared by training and visualisation."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
