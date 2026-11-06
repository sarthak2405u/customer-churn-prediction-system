import streamlit as st
import pandas as pd
import plotly.express as px
import sys

from pathlib import Path
from sqlalchemy import text
from database.db_connection import engine


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Prediction History",
    page_icon="📋",
    layout="wide"
)


# ==========================================
# 2. PAGE HEADER
# ==========================================

st.title("📋 Customer Prediction History")

st.markdown("""
Monitor previously generated customer churn predictions
stored in your PostgreSQL database.
""")

st.divider()


# ==========================================
# 3. LOAD PREDICTION DATA
# ==========================================

@st.cache_data(ttl=60)
def load_prediction_history():

    query = """
        SELECT
            p.prediction_id,
            p.customer_id,
            p.churn_probability,
            p.predicted_churn,
            p.model_name,
            p.prediction_date
        FROM predictions p
        ORDER BY p.prediction_date DESC;
    """

    with engine.connect() as connection:

        data = pd.read_sql(
            text(query),
            connection
        )

    return data


try:

    prediction_df = load_prediction_history()

    if prediction_df.empty:

        st.info(
            "No prediction records found. "
            "Generate a prediction first."
        )

        st.stop()


    # ======================================
    # 4. DATA PREPARATION
    # ======================================

    prediction_df["churn_probability"] = pd.to_numeric(
        prediction_df["churn_probability"],
        errors="coerce"
    )

    prediction_df["prediction_date"] = pd.to_datetime(
        prediction_df["prediction_date"],
        errors="coerce"
    )

    prediction_df["predicted_churn"] = (
        prediction_df["predicted_churn"].astype(int)
    )

    prediction_df["prediction_status"] = (
        prediction_df["predicted_churn"]
        .map({
            1: "Likely to Churn",
            0: "Likely to Stay"
        })
    )

    prediction_df["risk_category"] = pd.cut(
        prediction_df["churn_probability"],
        bins=[-0.001, 0.40, 0.70, 1.001],
        labels=["Low", "Medium", "High"]
    )


    # ======================================
    # 5. DASHBOARD FILTERS
    # ======================================

    st.sidebar.header("Prediction Filters")

    status_options = [
        "All",
        "Likely to Churn",
        "Likely to Stay"
    ]

    selected_status = st.sidebar.selectbox(
        "Prediction Status",
        status_options
    )

    risk_options = [
        "All",
        "Low",
        "Medium",
        "High"
    ]

    selected_risk = st.sidebar.selectbox(
        "Risk Category",
        risk_options
    )

    filtered_df = prediction_df.copy()

    if selected_status != "All":

        filtered_df = filtered_df[
            filtered_df["prediction_status"] == selected_status
        ]

    if selected_risk != "All":

        filtered_df = filtered_df[
            filtered_df["risk_category"].astype(str) == selected_risk
        ]


    # ======================================
    # 6. CUSTOMER PREDICTION OVERVIEW
    # ======================================

    st.subheader("Prediction Overview")

    total_predictions = len(filtered_df)

    churn_predictions = int(
        (filtered_df["predicted_churn"] == 1).sum()
    )

    retained_predictions = int(
        (filtered_df["predicted_churn"] == 0).sum()
    )

    average_probability = (
        filtered_df["churn_probability"].mean() * 100
        if total_predictions > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Predictions",
        f"{total_predictions:,}"
    )

    col2.metric(
        "Likely to Churn",
        f"{churn_predictions:,}"
    )

    col3.metric(
        "Likely to Stay",
        f"{retained_predictions:,}"
    )

    col4.metric(
        "Average Churn Probability",
        f"{average_probability:.2f}%"
    )

    st.divider()


    if filtered_df.empty:

        st.warning("No records match the selected filters.")

    else:

        # ==================================
        # 7. PREDICTION DISTRIBUTION
        # ==================================

        st.subheader("Prediction Distribution")

        distribution = pd.DataFrame({
            "Prediction": [
                "Likely to Churn",
                "Likely to Stay"
            ],
            "Count": [
                churn_predictions,
                retained_predictions
            ]
        })

        fig = px.pie(
            distribution,
            names="Prediction",
            values="Count",
            hole=0.45,
            title="Churn vs Retention Predictions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()


        # ==================================
        # 8. RISK CATEGORY DISTRIBUTION
        # ==================================

        st.subheader("Customer Risk Distribution")

        risk_distribution = (
            filtered_df["risk_category"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .reset_index()
        )

        risk_distribution.columns = [
            "Risk Category",
            "Customers"
        ]

        fig = px.bar(
            risk_distribution,
            x="Risk Category",
            y="Customers",
            text="Customers",
            title="Customers by Churn Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()


        # ==================================
        # 9. PREDICTION TRENDS
        # ==================================

        st.subheader("Prediction Trends Over Time")

        trend_df = filtered_df.dropna(
            subset=["prediction_date"]
        ).copy()

        if not trend_df.empty:

            trend_df["prediction_day"] = (
                trend_df["prediction_date"].dt.date
            )

            daily_predictions = (
                trend_df.groupby("prediction_day")
                .size()
                .reset_index(name="Total Predictions")
            )

            fig = px.line(
                daily_predictions,
                x="prediction_day",
                y="Total Predictions",
                markers=True,
                title="Daily Prediction Activity"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No valid prediction dates available.")


        st.divider()


        # ==================================
        # 10. HIGH-RISK CUSTOMERS
        # ==================================

        st.subheader("High-Risk Customers")

        high_risk_df = filtered_df[
            (filtered_df["risk_category"] == "High")
            & (filtered_df["predicted_churn"] == 1)
        ].copy()

        if not high_risk_df.empty:

            st.dataframe(
                high_risk_df[
                    [
                        "customer_id",
                        "churn_probability",
                        "model_name",
                        "prediction_date"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No high-risk churn predictions found.")


        st.divider()


        # ==================================
        # 11. COMPLETE PREDICTION HISTORY
        # ==================================

        st.subheader("Complete Prediction History")

        display_df = filtered_df.copy()

        display_df["churn_probability"] = (
            display_df["churn_probability"] * 100
        ).round(2)

        display_df = display_df.rename(
            columns={
                "prediction_id": "Prediction ID",
                "customer_id": "Customer ID",
                "churn_probability": "Churn Probability (%)",
                "predicted_churn": "Predicted Churn",
                "model_name": "Model",
                "prediction_date": "Prediction Date",
                "prediction_status": "Prediction Status",
                "risk_category": "Risk Category"
            }
        )

        st.dataframe(
            display_df[
                [
                    "Prediction ID",
                    "Customer ID",
                    "Churn Probability (%)",
                    "Predicted Churn",
                    "Prediction Status",
                    "Risk Category",
                    "Model",
                    "Prediction Date"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


        # ==================================
        # 12. DOWNLOAD PREDICTION HISTORY
        # ==================================

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction History",
            data=csv_data,
            file_name="customer_prediction_history.csv",
            mime="text/csv"
        )


    # ======================================
    # 13. REFRESH DASHBOARD
    # ======================================

    st.divider()

    if st.button("Refresh Prediction History"):

        st.cache_data.clear()

        st.rerun()


except Exception as e:

    st.error(
        f"Unable to load prediction history: {e}"
    )