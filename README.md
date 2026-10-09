# Customer Churn Prediction (Telco)

End-to-end classification project: predict whether a telecom customer will churn,
evaluate the models honestly, and serve the best one in an interactive Streamlit app.

## Business problem

Acquiring a new customer costs far more than retaining one. This project flags
at-risk customers so a retention team can act **before** they leave — the classic
predictive-analytics use case: turn historical customer data into a ranked,
actionable signal.

## Dataset (real)

**IBM Telco Customer Churn** — 7,043 customers, 21 columns
(demographics, tenure, contract, services, charges, churn label).

- Source: https://github.com/IBM/telco-customer-churn-on-icp4d
- Local copy: `data/Telco-Customer-Churn.csv`
- Churn rate: **26.5%** (1,869 churners) — imbalanced, handled with
  stratified splits and `class_weight="balanced"`.

## Pipeline

```
load -> clean -> stratified train/test split -> 3-model comparison
      -> 5-fold cross-validation -> holdout evaluation -> save best model
```

**Cleaning:** `TotalCharges` ships with blank strings for new customers
(coerced to numeric, median-imputed inside the pipeline); `Churn` Yes/No mapped
to 1/0; `customerID` dropped. Numeric features are median-imputed + scaled,
categoricals imputed + one-hot encoded in a single `ColumnTransformer`, fitted
only on training data (no leakage).

## Results (holdout test set, n = 1,409)

| Model | CV F1 (5-fold) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Decision Tree ⭐ | 0.609 | 0.741 | 0.508 | **0.791** | **0.619** | 0.829 |
| Logistic Regression | **0.633** | 0.738 | 0.504 | 0.783 | 0.614 | **0.841** |
| Random Forest | 0.606 | **0.764** | **0.549** | 0.620 | 0.582 | 0.823 |

Best model by holdout F1 (saved to `models/churn_model.joblib`): **Decision Tree**.
Full table: `model_results.csv`.

**Reading the metrics:** recall is the business metric here — a missed churner is
lost revenue, while a false alarm only costs a retention offer. That is why the
models are tuned for F1/recall rather than raw accuracy.

## Charts (`charts/`)

`churn_distribution.png` · `churn_by_contract.png` (month-to-month churns most) ·
`tenure_vs_charges.png` · `correlation_heatmap.png` ·
`confusion_matrix.png` · `roc_curve.png`

## Project structure

```
customer-churn-prediction/
├── train.py            # training + comparison + saves best model
├── visualize.py        # EDA + evaluation charts -> charts/
├── app.py              # Streamlit demo: KPIs, metrics, 1-customer prediction
├── data_utils.py       # shared loading/cleaning/splitting (single source of truth)
├── data/               # Telco-Customer-Churn.csv
├── models/             # churn_model.joblib (created by train.py)
├── charts/             # PNGs (created by visualize.py)
└── model_results.csv   # metrics table (created by train.py)
```

## Run

```bash
pip install -r requirements.txt
python train.py        # train, compare, save best model
python visualize.py    # generate charts/
streamlit run app.py   # launch the interactive demo
```

## Limitations / next steps

1. Tune the decision threshold to the real cost of retention offers vs churn.
2. Try gradient boosting and probability calibration.
3. Validate on fresh data; check fairness across customer segments.
4. Measure actual retention outcomes before claiming business value.
