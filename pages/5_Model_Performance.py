import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" /
    "customer_churn_engineered.csv"
)


# ==========================================
# 2. PAGE HEADER
# ==========================================

st.title("📈 Machine Learning Model Performance")

st.markdown("""
Evaluate the performance of the trained customer churn
prediction model using standard classification metrics.
""")

st.divider()


# ==========================================
# 3. LOAD MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


try:

    model = load_model()

    df = load_dataset()

except Exception as e:

    st.error(f"Unable to load model or dataset: {e}")
    st.stop()


# ==========================================
# 4. PREPARE TEST DATA
# ==========================================

try:

    # Use the exact feature columns expected by the model
    features = list(model.feature_names_in_)

    missing_columns = [
        col for col in features
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    X = df[features].copy()

    # Convert target to binary
    y = df["Churn"].replace({
        "Yes": 1,
        "No": 0
    })

    y = pd.to_numeric(y, errors="coerce")

    if y.isna().any():

        raise ValueError(
            "Churn contains missing or invalid target values."
        )

    y = y.astype(int)

    if not set(y.unique()).issubset({0, 1}):

        raise ValueError(
            "Churn must contain only 0 and 1."
        )

    # Recreate the original test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Generate predictions
    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    st.success("Model and evaluation dataset loaded successfully!")

except Exception as e:

    st.error(f"Unable to prepare model evaluation: {e}")
    st.stop()


# ==========================================
# 5. CALCULATE PERFORMANCE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# 6. DISPLAY KPI CARDS
# ==========================================

st.subheader("Model Performance Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Precision",
    f"{precision * 100:.2f}%"
)

col3.metric(
    "Recall",
    f"{recall * 100:.2f}%"
)

col4, col5 = st.columns(2)

col4.metric(
    "F1 Score",
    f"{f1 * 100:.2f}%"
)

col5.metric(
    "ROC-AUC",
    f"{roc_auc:.4f}"
)

st.divider()


# ==========================================
# 7. METRICS COMPARISON
# ==========================================

st.subheader("Performance Metrics Comparison")

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

fig = px.bar(
    metrics_df,
    x="Metric",
    y="Score",
    text=metrics_df["Score"].round(3),
    title="Classification Performance Metrics",
    range_y=[0, 1]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

fig = px.imshow(
    cm,
    text_auto=True,
    x=["Predicted Stay", "Predicted Churn"],
    y=["Actual Stay", "Actual Churn"],
    color_continuous_scale="Blues",
    title="Churn Prediction Confusion Matrix"
)

fig.update_layout(
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

tn, fp, fn, tp = cm.ravel()

col1, col2 = st.columns(2)

col1.metric("True Negatives", int(tn))
col2.metric("False Positives", int(fp))

col3, col4 = st.columns(2)

col3.metric("False Negatives", int(fn))
col4.metric("True Positives", int(tp))

st.divider()


# ==========================================
# 9. ROC CURVE
# ==========================================

st.subheader("ROC Curve")

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC Curve (AUC = {roc_auc:.4f})"
    )
)

fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(dash="dash")
    )
)

fig.update_layout(
    title="Receiver Operating Characteristic Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

st.subheader("Detailed Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Retained", "Churned"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

st.divider()


# ==========================================
# 11. MODEL PERFORMANCE INTERPRETATION
# ==========================================

st.subheader("Model Performance Interpretation")

st.write(
    f"**Accuracy:** The model correctly classified "
    f"{accuracy * 100:.2f}% of test customers."
)

st.write(
    f"**Precision:** Of customers predicted to churn, "
    f"{precision * 100:.2f}% actually churned."
)

st.write(
    f"**Recall:** The model identified "
    f"{recall * 100:.2f}% of customers who actually churned."
)

st.write(
    f"**F1 Score:** The balance between precision and recall "
    f"is {f1 * 100:.2f}%."
)

st.write(
    f"**ROC-AUC:** The model's ability to distinguish between "
    f"churned and retained customers is {roc_auc:.4f}."
)

st.info(
    "For customer churn applications, recall is particularly "
    "important because missing an actual churner can mean "
    "losing an opportunity for customer retention."
)


# ==========================================
# 12. DOWNLOAD EVALUATION RESULTS
# ==========================================

st.divider()

metrics_download = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

csv_data = metrics_download.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Model Evaluation Metrics",
    data=csv_data,
    file_name="churn_model_evaluation.csv",
    mime="text/csv"
)


# ==========================================
# 13. DATASET INFORMATION
# ==========================================

st.divider()

st.subheader("Evaluation Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Dataset Records",
    f"{len(df):,}"
)

col2.metric(
    "Training Records",
    f"{len(X_train):,}"
)

col3.metric(
    "Testing Records",
    f"{len(X_test):,}"
)

st.caption(
    "Metrics are calculated on the 20% holdout test split "
    "using random_state=42 and stratified sampling."
)