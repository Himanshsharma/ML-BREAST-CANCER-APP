import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Breast Cancer Classification - ML Assignment 2")
st.write("Comparison of 6 Machine Learning Models")

# ===============================
# Model Folder Path
# ===============================

MODEL_PATH = "model"

model_files = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

# ===============================
# Upload CSV
# ===============================

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select Target Column
    target_column = "target"

    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # ===============================
    # Load Scaler (Always Load)
    # ===============================

    scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Ensure feature names match training
    feature_names = scaler.feature_names_in_

    try:
        X = X[feature_names]
    except Exception:
        st.error("Feature mismatch! Please upload correct test dataset.")
        st.stop()

    # ===============================
    # Select Model
    # ===============================

    selected_model_name = st.selectbox("Select Model", list(model_files.keys()))
    model_file = model_files[selected_model_name]

    model = joblib.load(os.path.join(MODEL_PATH, model_file))

    # ===============================
    # Apply Scaling if Required
    # ===============================

    if selected_model_name in ["Logistic Regression", "KNN"]:
        X_transformed = scaler.transform(X)
    else:
        X_transformed = X

    # ===============================
    # Predictions
    # ===============================

    y_pred = model.predict(X_transformed)

    # AUC calculation
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_transformed)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = None

    # ===============================
    # Evaluation Metrics
    # ===============================

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
    col2.metric("Precision", round(precision_score(y, y_pred), 4))
    col3.metric("Recall", round(recall_score(y, y_pred), 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y, y_pred), 4))
    col5.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))

    if auc is not None:
        col6.metric("AUC Score", round(auc, 4))
    else:
        col6.metric("AUC Score", "N/A")

    # ===============================
    # Confusion Matrix
    # ===============================

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)

    # ===============================
    # Classification Report
    # ===============================

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

else:
    st.info("Please upload a CSV file to begin.")
