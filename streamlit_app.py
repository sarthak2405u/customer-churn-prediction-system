import streamlit as st

st.set_page_config(
    page_title="Customer Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 17px;
    color: #808080;
    margin-bottom: 30px;
}

div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.2);
}

</style>
""", unsafe_allow_html=True)


# Homepage
st.markdown(
    '<div class="main-title">📊 Customer Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An Integrated Machine Learning, NLP and Business Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.divider()

# Introduction
st.header("Project Overview")

st.write("""
The Customer Intelligence System is an end-to-end analytics
application designed to understand customer behaviour,
predict customer churn and analyse customer feedback.

It integrates Machine Learning, Natural Language Processing,
PostgreSQL and interactive data visualization.
""")

# Dashboard modules
st.subheader("Platform Modules")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **📊 Customer Analytics**

    - Customer segmentation insights
    - Churn analysis
    - Contract and payment analysis
    - Customer behaviour visualization
    """)

    st.info("""
    **📉 Churn Prediction**

    - Individual customer predictions
    - Churn probability
    - Risk classification
    - SHAP explainability
    """)

    st.info("""
    **📈 Model Performance**

    - Accuracy, Precision and Recall
    - F1 Score and ROC-AUC
    - Confusion Matrix
    - ROC Curve
    """)

with col2:
    st.info("""
    **💬 Customer Sentiment**

    - VADER sentiment analysis
    - TF-IDF vectorization
    - Naive Bayes classification
    - Customer feedback analytics
    """)

    st.info("""
    **📋 Prediction History**

    - PostgreSQL integration
    - Prediction records
    - High-risk customer monitoring
    - Prediction trends
    """)

st.divider()

# Technology stack
st.subheader("Technology Stack")

technologies = [
    "Python",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "XGBoost",
    "NLTK",
    "SHAP",
    "PostgreSQL",
    "SQLAlchemy",
    "Streamlit",
    "Plotly"
]

st.write(" | ".join(technologies))

st.divider()

st.success(
    "Select a dashboard module from the sidebar to get started."
)

st.caption("Customer Intelligence & Churn Prediction System | Portfolio Project")