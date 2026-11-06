import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
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
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)


# ==========================================
# 2. MODEL PATHS
# ==========================================

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"

BACKGROUND_PATH = PROJECT_ROOT / "models" / "shap_background.pkl"

FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "shap_feature_names.pkl"


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load churn model: {e}")
    st.stop()


# ==========================================
# 4. LOAD SHAP EXPLAINER
# ==========================================

@st.cache_resource
def load_shap_explainer():

    background_data = joblib.load(BACKGROUND_PATH)

    feature_names = joblib.load(FEATURE_NAMES_PATH)

    if hasattr(background_data, "toarray"):
        background_data = background_data.toarray()

    classifier = model.named_steps["classifier"]

    explainer = shap.LinearExplainer(
        classifier,
        background_data
    )

    return explainer, feature_names


# ==========================================
# 5. PAGE HEADER
# ==========================================

st.title("📉 Customer Churn Prediction")

st.markdown("""
Predict customer churn using Machine Learning,
understand prediction results using SHAP,
and store prediction history in PostgreSQL.
""")

st.divider()


# ==========================================
# 6. CUSTOMER INFORMATION FORM
# ==========================================

st.subheader("Enter Customer Information")

with st.form("churn_prediction_form"):

    st.markdown("### Customer Identification")

    customer_id = st.text_input(
        "Customer ID",
        placeholder="CUST-00001"
    )

    col1, col2 = st.columns(2)

    # --------------------------------------
    # CUSTOMER DETAILS
    # --------------------------------------

    with col1:

        st.markdown("### Personal Information")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        st.markdown("### Internet Services")

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    # --------------------------------------
    # BILLING AND SUBSCRIPTION DETAILS
    # --------------------------------------

    with col2:

        st.markdown("### Phone Services")

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        st.markdown("### Billing Information")

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submit = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )


# ==========================================
# 7. FEATURE ENGINEERING
# ==========================================

if submit:

    if not customer_id.strip():

        st.error("Please enter a Customer ID.")

        st.stop()

    # Tenure Group
    if tenure <= 12:
        tenure_group = "New Customer"

    elif tenure <= 24:
        tenure_group = "Regular Customer"

    elif tenure <= 48:
        tenure_group = "Established Customer"

    elif tenure <= 60:
        tenure_group = "Loyal Customer"

    else:
        tenure_group = "Long-Term Customer"

    # Total Services
    service_values = [
        phone_service,
        multiple_lines,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]

    total_services = (
        sum(value == "Yes" for value in service_values)
        + int(internet_service != "No")
    )

    # Automatic Payment
    auto_payment = int(
        payment_method in [
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    # Monthly Contract
    monthly_contract = int(
        contract == "Month-to-month"
    )

    # Technical Support
    has_tech_support = int(
        tech_support == "Yes"
    )


    # ======================================
    # 8. CREATE CUSTOMER DATAFRAME
    # ======================================

    customer_data = pd.DataFrame([{

        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "TenureGroup": tenure_group,
        "TotalServices": total_services,
        "AutoPayment": auto_payment,
        "MonthlyContract": monthly_contract,
        "HasTechSupport": has_tech_support

    }])


    # ======================================
    # 9. CHURN PREDICTION
    # ======================================

    try:

        expected_features = list(model.feature_names_in_)

        missing_features = [
            col for col in expected_features
            if col not in customer_data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        customer_data = customer_data[expected_features]

        prediction = model.predict(customer_data)[0]

        probability = model.predict_proba(
            customer_data
        )[0][1]

        churn_percentage = probability * 100


        # ==================================
        # 10. DISPLAY PREDICTION RESULTS
        # ==================================

        st.divider()

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Churn Probability",
            f"{churn_percentage:.2f}%"
        )

        col2.metric(
            "Prediction",
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )

        if churn_percentage >= 70:
            risk = "High"

        elif churn_percentage >= 40:
            risk = "Medium"

        else:
            risk = "Low"

        col3.metric(
            "Risk Category",
            risk
        )

        st.progress(float(probability))

        if prediction == 1:
            st.error("This customer is predicted to churn.")

        else:
            st.success("This customer is predicted to stay.")


        # ==================================
        # 11. RETENTION RECOMMENDATIONS
        # ==================================

        st.divider()

        st.subheader("Customer Retention Recommendations")

        if prediction == 1:

            st.warning(
                "Recommended actions to reduce churn risk:"
            )

            st.write(
                "- Contact the customer with a personalized retention offer."
            )

            if contract == "Month-to-month":
                st.write(
                    "- Offer an annual contract with suitable incentives."
                )

            if tech_support == "No":
                st.write(
                    "- Consider offering technical support."
                )

            if monthly_charges > 80:
                st.write(
                    "- Review pricing and consider suitable discounts."
                )

        else:

            st.success(
                "The customer currently has a lower predicted churn risk."
            )

            st.write("- Maintain service quality.")
            st.write("- Continue customer engagement.")
            st.write("- Monitor customer satisfaction.")


        # ==================================
        # 12. SHAP EXPLAINABILITY
        # ==================================

        st.divider()

        st.subheader("🔍 Why This Customer May Churn")

        try:

            explainer, shap_feature_names = load_shap_explainer()

            preprocessor = model.named_steps["preprocessor"]

            transformed_customer = preprocessor.transform(
                customer_data
            )

            if hasattr(transformed_customer, "toarray"):
                transformed_customer = transformed_customer.toarray()

            shap_explanation = explainer(
                transformed_customer
            )

            shap_values = np.asarray(
                shap_explanation.values
            )

            if shap_values.ndim == 3:
                shap_values = shap_values[0, :, 1]

            else:
                shap_values = shap_values[0]

            base_values = np.asarray(
                shap_explanation.base_values
            )

            if base_values.ndim == 2:
                base_value = base_values[0, 1]

            else:
                base_value = base_values.reshape(-1)[0]

            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=transformed_customer[0],
                feature_names=list(shap_feature_names)
            )

            fig = plt.figure(figsize=(11, 7))

            shap.plots.waterfall(
                explanation,
                max_display=15,
                show=False
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True
            )

            plt.close("all")

            st.caption(
                "SHAP values explain contributions to the model's "
                "output in log-odds, not direct percentage-point "
                "changes in churn probability."
            )

        except Exception as shap_error:

            st.warning(
                f"SHAP explanation could not be generated: {shap_error}"
            )


        # ==================================
        # 13. SAVE PREDICTION TO POSTGRESQL
        # ==================================

        st.divider()

        st.subheader("PostgreSQL Prediction Storage")

        try:

            with engine.begin() as connection:

                # Verify customer ID
                customer_exists = connection.execute(
                    text("""
                        SELECT EXISTS (
                            SELECT 1
                            FROM customers
                            WHERE customer_id = :customer_id
                        );
                    """),
                    {
                        "customer_id": customer_id.strip()
                    }
                ).scalar()

                if customer_exists:

                    # Insert prediction record
                    result = connection.execute(
                        text("""
                            INSERT INTO predictions (
                                customer_id,
                                churn_probability,
                                predicted_churn,
                                model_name
                            )
                            VALUES (
                                :customer_id,
                                :probability,
                                :prediction,
                                :model_name
                            )
                            RETURNING prediction_id;
                        """),
                        {
                            "customer_id": customer_id.strip(),
                            "probability": float(probability),
                            "prediction": int(prediction),
                            "model_name": "Logistic Regression"
                        }
                    )

                    prediction_id = result.scalar_one()

                    st.success(
                        "Prediction successfully saved to PostgreSQL!"
                    )

                    st.write(
                        f"**Prediction ID:** {prediction_id}"
                    )

                else:

                    st.warning(
                        f"Customer ID '{customer_id}' was not found "
                        "in the customers table. Prediction was not saved."
                    )

        except Exception as db_error:

            st.error(
                f"Database insertion failed: {db_error}"
            )


        # ==================================
        # 14. CUSTOMER INPUT DETAILS
        # ==================================

        with st.expander("View Customer Input Details"):

            st.dataframe(
                customer_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(f"Prediction failed: {e}")