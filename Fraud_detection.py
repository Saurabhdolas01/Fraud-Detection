import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .block-container {
        max-width: 900px;
        padding: 2rem 2rem 2rem 2rem;
        background-color: #f9f9f9;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.6em 1.5em;
        border-radius: 6px;
        font-weight: bold;
    }
    .stMetric {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Fraud Detection System")

# ------------------ LOAD MODEL & SCALER ------------------
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📤 Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    orig_df = df.copy()

    # ---------- Section 1: Uploaded Data Preview ----------
    with st.expander("📋 Uploaded Data Preview & Summary", expanded=True):
        st.dataframe(df.head())
        st.write("📐 Shape:", df.shape)
        st.write("❓ Missing Values:", df.isnull().sum().sum())
        st.write("📊 Fraud Distribution:")
        st.write(df['isFraud'].value_counts(normalize=True))

    # ---------- Feature Engineering ----------
    df['deltaOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['deltaDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['isMerchantDest'] = df['nameDest'].apply(lambda x: 1 if str(x).startswith('M') else 0)
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    df.drop(columns=['nameOrig', 'nameDest'], inplace=True)

    true_labels = df['isFraud']
    df.drop(columns=['isFraud'], inplace=True)
    df = df.reindex(columns=model.get_booster().feature_names, fill_value=0)
    X_scaled = scaler.transform(df)

    # ---------- Prediction ----------
    threshold = 0.5  # Fixed
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # ---------- Section 2: Model Evaluation ----------
    with st.expander("📉 Model Evaluation", expanded=True):
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        st.write("✅ Confusion Matrix")
        st.write(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1']))

        st.write("✅ Classification Report")
        report = classification_report(true_labels, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    # ---------- Section 3: Business Cost Simulation ----------
    with st.expander("📉 Business Cost Simulation", expanded=True):
        cost_fp = 500
        cost_fn = 10000
        fp = ((predictions == 1) & (true_labels == 0)).sum()
        fn = ((predictions == 0) & (true_labels == 1)).sum()
        total_cost = fp * cost_fp + fn * cost_fn

        st.metric("💸 False Positives Cost", f"₹{fp * cost_fp:,}")
        st.metric("💥 False Negatives Cost", f"₹{fn * cost_fn:,}")
        st.metric("🧾 Total Estimated Business Cost", f"₹{total_cost:,}")
