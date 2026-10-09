"""Customer Churn Prediction & Business Insights (Streamlit demo).

Dataset: IBM Telco Customer Churn — 7,043 real telecom customers.
Source: https://github.com/IBM/telco-customer-churn-on-icp4d

Run:
    streamlit run app.py
"""

import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from data_utils import (
    MODEL_PATH,
    build_preprocessor,
    clean,
    get_features,
    load_raw,
    train_test_frames,
)

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction & Business Insights")
st.caption(
    "Trained on the real IBM Telco Customer Churn dataset "
    "(7,043 customers). Metrics below are holdout test results."
)

# --- Data ---------------------------------------------------------------------
try:
    df = clean(load_raw())
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

X, y = get_features(df)
X_train, X_test, y_train, y_test = train_test_frames(X, y)

# --- Model (load the trained best model, train a fallback if missing) ---------
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), "saved best model (train.py)"
    preprocessor = build_preprocessor(X)
    fallback = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    fallback.fit(X_train, y_train)
    return fallback, "logistic-regression fallback (run train.py for the best model)"

with st.spinner("Loading model..."):
    model, model_note = get_model()
st.caption(f"Model in use: {model_note}")

# --- 1. Explore ----------------------------------------------------------------
st.subheader("1. Explore the data")
c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Churn rate", f"{df['churn'].mean():.1%}")
c3.metric("Input features", str(X.shape[1]))
with st.expander("Preview dataset"):
    st.dataframe(df.head(15), use_container_width=True)

left, right = st.columns(2)
with left:
    st.write("Churn distribution (0 = stay, 1 = churn)")
    st.bar_chart(df["churn"].value_counts().sort_index())
with right:
    st.write("Churn rate by contract type")
    st.bar_chart(df.groupby("Contract")["churn"].mean().sort_values(ascending=False))

# --- 2. Evaluate ---------------------------------------------------------------
st.subheader("2. Evaluate on holdout test data")
pred = model.predict(X_test)
metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred, zero_division=0),
    "Recall": recall_score(y_test, pred, zero_division=0),
    "F1-score": f1_score(y_test, pred, zero_division=0),
}
cols = st.columns(4)
for col, (label, score) in zip(cols, metrics.items()):
    col.metric(label, f"{score:.3f}")
st.caption(
    f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,} "
    "| stratified split, random_state=42"
)
cm = confusion_matrix(y_test, pred, labels=[0, 1])
st.write("Confusion matrix (rows = actual, columns = predicted)")
st.dataframe(
    pd.DataFrame(
        cm,
        index=["Actual stay (0)", "Actual churn (1)"],
        columns=["Predicted stay (0)", "Predicted churn (1)"],
    )
)

# --- 3. Predict one customer ----------------------------------------------------
st.subheader("3. Predict churn risk for one customer")
with st.form("customer_form"):
    a, b, c = st.columns(3)
    tenure = a.slider("Tenure (months)", 0, 72, 12)
    monthly = b.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, 1.0)
    contract = c.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    d, e, f = st.columns(3)
    internet = d.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    tech = e.selectbox("Tech support", ["No", "Yes", "No internet service"])
    payment = f.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    g, h = st.columns(2)
    senior = g.selectbox("Senior citizen", ["No (0)", "Yes (1)"])
    partner = h.selectbox("Has partner", ["No", "Yes"])
    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    total = round(monthly * max(tenure, 1), 2)
    example = pd.DataFrame(
        [
            {
                # Form inputs:
                "gender": "Female",
                "SeniorCitizen": 1 if senior.startswith("Yes") else 0,
                "Partner": partner,
                "Dependents": "No",
                "tenure": tenure,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": internet,
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": tech,
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": contract,
                "PaperlessBilling": "Yes",
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )
    risk = float(model.predict_proba(example)[0][1])
    st.metric("Model-estimated churn probability", f"{risk:.1%}")
    if risk >= 0.60:
        st.warning(
            "Higher predicted risk — typical drivers are month-to-month contracts, "
            "fiber-optic service without tech support, and high monthly charges. "
            "Prioritise a retention offer and a support check-in."
        )
    elif risk >= 0.35:
        st.info("Medium predicted risk — monitor usage and support contacts.")
    else:
        st.success("Lower predicted risk — continue normal monitoring.")
    st.caption(
        "Risk bands are illustrative. A prediction is a signal for investigation, "
        "not a guarantee the customer will leave."
    )

# --- 4. Business interpretation --------------------------------------------------
st.subheader("4. Business interpretation")
st.markdown(
    """
- **Recall matters most here:** every churner the model misses is lost revenue,
  while a false alarm only costs a retention offer — so we optimise F1/recall
  and use `class_weight="balanced"` for the ~27% churn rate.
- **Precision** tells us what share of flagged customers would actually churn.
- Before production: validate on fresh data, check leakage and fairness across
  segments, tune the decision threshold to the real cost of offers vs churn,
  and measure actual retention outcomes.
"""
)
