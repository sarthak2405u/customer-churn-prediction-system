import streamlit as st
import pandas as pd
import plotly.express as px
import sys

from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Customer Analytics",
    page_icon="📊",
    layout="wide"
)

# Import existing SQL analytics functions
from app.sql_analytics import (
    get_customer_overview,
    get_churn_by_contract,
    get_churn_by_payment,
    get_churn_by_internet,
    get_high_risk_customers,
    get_churn_by_tenure
)


# ==========================================
# 2. DASHBOARD HEADER
# ==========================================

st.title("📊 Customer Analytics Dashboard")

st.markdown("""
Explore customer behaviour, subscription patterns,
and churn trends using PostgreSQL analytics.
""")

st.divider()


# ==========================================
# 3. LOAD DATABASE ANALYTICS
# ==========================================

@st.cache_data(ttl=60)
def load_dashboard_data():

    overview = get_customer_overview()

    contract = get_churn_by_contract()

    payment = get_churn_by_payment()

    internet = get_churn_by_internet()

    high_risk = get_high_risk_customers()

    tenure = get_churn_by_tenure()

    return (
        overview,
        contract,
        payment,
        internet,
        high_risk,
        tenure
    )


try:

    (
        overview,
        contract_data,
        payment_data,
        internet_data,
        high_risk_data,
        tenure_data
    ) = load_dashboard_data()

    if overview.empty:
        st.warning("No customer data available.")
        st.stop()

    # ======================================
    # 4. CUSTOMER OVERVIEW
    # ======================================

    st.subheader("Customer Overview")

    total_customers = int(
        overview.iloc[0]["total_customers"]
    )

    churned_customers = int(
        overview.iloc[0]["churned_customers"]
    )

    retained_customers = int(
        overview.iloc[0]["retained_customers"]
    )

    churn_percentage = float(
        overview.iloc[0]["churn_percentage"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Retained Customers",
        f"{retained_customers:,}"
    )

    col4.metric(
        "Churn Percentage",
        f"{churn_percentage:.2f}%"
    )

    st.divider()


    # ======================================
    # 5. CUSTOMER CHURN DISTRIBUTION
    # ======================================

    st.subheader("Customer Churn Distribution")

    churn_distribution = pd.DataFrame({
        "Customer Status": [
            "Churned",
            "Retained"
        ],

        "Customers": [
            churned_customers,
            retained_customers
        ]
    })

    fig = px.pie(
        churn_distribution,
        names="Customer Status",
        values="Customers",
        hole=0.45,
        title="Churned vs Retained Customers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()


    # ======================================
    # 6. CHURN BY CONTRACT
    # ======================================

    st.subheader("Churn Analysis by Contract")

    if not contract_data.empty:

        fig = px.bar(
            contract_data,
            x="contract",
            y="churn_percentage",
            text="churn_percentage",
            title="Churn Percentage by Contract",
            labels={
                "contract": "Contract Type",
                "churn_percentage": "Churn Percentage"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with st.expander("View Contract Analytics Data"):
            st.dataframe(
                contract_data,
                use_container_width=True,
                hide_index=True
            )


    # ======================================
    # 7. CHURN BY PAYMENT METHOD
    # ======================================

    st.subheader("Churn Analysis by Payment Method")

    if not payment_data.empty:

        fig = px.bar(
            payment_data,
            x="payment_method",
            y="churn_percentage",
            text="churn_percentage",
            title="Churn Percentage by Payment Method",
            labels={
                "payment_method": "Payment Method",
                "churn_percentage": "Churn Percentage"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with st.expander("View Payment Analytics Data"):
            st.dataframe(
                payment_data,
                use_container_width=True,
                hide_index=True
            )


    # ======================================
    # 8. CHURN BY INTERNET SERVICE
    # ======================================

    st.subheader("Churn Analysis by Internet Service")

    if not internet_data.empty:

        fig = px.bar(
            internet_data,
            x="internet_service",
            y="churn_percentage",
            text="churn_percentage",
            title="Churn Percentage by Internet Service",
            labels={
                "internet_service": "Internet Service",
                "churn_percentage": "Churn Percentage"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with st.expander("View Internet Analytics Data"):
            st.dataframe(
                internet_data,
                use_container_width=True,
                hide_index=True
            )


    # ======================================
    # 9. CHURN BY TENURE
    # ======================================

    st.subheader("Customer Tenure Analysis")

    if not tenure_data.empty:

        fig = px.bar(
            tenure_data,
            x="tenure_group",
            y="churn_percentage",
            text="churn_percentage",
            title="Churn Percentage by Customer Tenure",
            labels={
                "tenure_group": "Customer Tenure",
                "churn_percentage": "Churn Percentage"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with st.expander("View Tenure Analytics Data"):
            st.dataframe(
                tenure_data,
                use_container_width=True,
                hide_index=True
            )


    # ======================================
    # 10. HIGH-RISK CUSTOMERS
    # ======================================

    st.subheader("High-Risk Customers")

    if not high_risk_data.empty:

        st.dataframe(
            high_risk_data,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No high-risk customers found.")


    # ======================================
    # 11. REFRESH DASHBOARD
    # ======================================

    st.divider()

    if st.button("Refresh Dashboard"):

        st.cache_data.clear()

        st.rerun()


except Exception as e:

    st.error(
        f"Unable to load customer analytics: {e}"
    )