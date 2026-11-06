import streamlit as st
import pandas as pd
import plotly.express as px
import sys

from pathlib import Path
from sqlalchemy import text

# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_connection import engine


# Page configuration
st.set_page_config(
    page_title="Customer Sentiment Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Sentiment Analytics")
st.markdown("### Machine Learning Based Customer Feedback Analysis")


# Load sentiment summary
@st.cache_data(ttl=60)
def load_sentiment_summary():

    query = """
    SELECT
        COUNT(*) AS total_reviews,
        COUNT(*) FILTER (
            WHERE ml_sentiment = 'Positive'
        ) AS positive_reviews,
        COUNT(*) FILTER (
            WHERE ml_sentiment = 'Negative'
        ) AS negative_reviews,
        ROUND(AVG(ml_confidence), 4) AS average_confidence
    FROM customer_reviews
    WHERE ml_sentiment IS NOT NULL;
    """

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


# Load churn analytics
@st.cache_data(ttl=60)
def load_churn_sentiment():

    query = """
    SELECT
        r.ml_sentiment,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        SUM(c.churn) AS churned_customers,
        ROUND(
            AVG(c.churn::NUMERIC) * 100, 2
        ) AS churn_percentage
    FROM customers c
    INNER JOIN customer_reviews r
        ON c.customer_id = r.customer_id
    WHERE r.ml_sentiment IS NOT NULL
    GROUP BY r.ml_sentiment;
    """

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


# Load negative reviews
@st.cache_data(ttl=60)
def load_negative_reviews():

    query = """
    SELECT
        c.customer_id,
        c.contract,
        c.tenure,
        c.monthly_charges,
        c.churn,
        r.review_text,
        r.ml_sentiment,
        r.ml_confidence
    FROM customers c
    INNER JOIN customer_reviews r
        ON c.customer_id = r.customer_id
    WHERE r.ml_sentiment = 'Negative'
    ORDER BY r.ml_confidence DESC;
    """

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


# Display KPIs
try:

    summary = load_sentiment_summary()
    churn_data = load_churn_sentiment()
    negative_reviews = load_negative_reviews()

    total_reviews = int(summary.iloc[0]["total_reviews"])
    positive_reviews = int(summary.iloc[0]["positive_reviews"])
    negative_count = int(summary.iloc[0]["negative_reviews"])

    avg_confidence = summary.iloc[0]["average_confidence"]

    avg_confidence = (
        float(avg_confidence) * 100
        if pd.notna(avg_confidence)
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", total_reviews)
    col2.metric("Positive Reviews", positive_reviews)
    col3.metric("Negative Reviews", negative_count)
    col4.metric("Average Confidence", f"{avg_confidence:.2f}%")

    st.divider()

    # Sentiment distribution
    st.subheader("Sentiment Distribution")

    sentiment_data = pd.DataFrame({
        "Sentiment": ["Positive", "Negative"],
        "Reviews": [positive_reviews, negative_count]
    })

    fig = px.pie(
        sentiment_data,
        names="Sentiment",
        values="Reviews",
        hole=0.45,
        title="Positive vs Negative Reviews"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Churn analytics
    st.subheader("Customer Churn by Sentiment")

    if not churn_data.empty:

        fig = px.bar(
            churn_data,
            x="ml_sentiment",
            y="churn_percentage",
            text="churn_percentage",
            title="Churn Percentage by Sentiment",
            labels={
                "ml_sentiment": "Sentiment",
                "churn_percentage": "Churn Percentage"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(churn_data, use_container_width=True)

    # Negative customer feedback
    st.subheader("Customers with Negative Feedback")

    st.dataframe(
        negative_reviews,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()

except Exception as e:

    st.error(f"Unable to load dashboard data: {e}")