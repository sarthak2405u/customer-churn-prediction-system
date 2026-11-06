import pandas as pd

from database.db_connection import engine
from sqlalchemy import text

def get_customer_overview():

    query = """
    SELECT
        COUNT(*) AS total_customers,

        SUM(churn) AS churned_customers,

        COUNT(*) - SUM(churn) AS retained_customers,

        ROUND(AVG(churn) * 100, 2) AS churn_percentage

    FROM customers;
    """

    return pd.read_sql(query, engine)

def get_churn_by_contract():

    query = """
    SELECT
        contract,

        COUNT(*) AS total_customers,

        SUM(churn) AS churned_customers,

        ROUND(AVG(churn) * 100, 2) AS churn_percentage

    FROM customers

    GROUP BY contract

    ORDER BY churn_percentage DESC;
    """

    return pd.read_sql(query, engine)

def get_churn_by_payment():

    query = """
    SELECT
        payment_method,

        COUNT(*) AS total_customers,

        SUM(churn) AS churned_customers,

        ROUND(AVG(churn) * 100, 2) AS churn_percentage

    FROM customers

    GROUP BY payment_method

    ORDER BY churn_percentage DESC;
    """

    return pd.read_sql(query, engine)

def get_churn_by_internet():

    query = """
    SELECT
        internet_service,

        COUNT(*) AS total_customers,

        SUM(churn) AS churned_customers,

        ROUND(AVG(churn) * 100, 2) AS churn_percentage

    FROM customers

    GROUP BY internet_service

    ORDER BY churn_percentage DESC;
    """

    return pd.read_sql(query, engine)

def get_high_risk_customers():

    query = """
    SELECT
        customer_id,
        tenure,
        contract,
        monthly_charges,
        total_charges,
        internet_service

    FROM customers

    WHERE churn = 1

    AND monthly_charges > 80

    ORDER BY monthly_charges DESC

    LIMIT 20;
    """

    return pd.read_sql(query, engine)

def get_churn_by_tenure():

    query = """
    SELECT

        CASE

            WHEN tenure <= 12 THEN '0-12 Months'

            WHEN tenure <= 24 THEN '13-24 Months'

            WHEN tenure <= 48 THEN '25-48 Months'

            ELSE '49-72 Months'

        END AS tenure_group,

        COUNT(*) AS total_customers,

        SUM(churn) AS churned_customers,

        ROUND(AVG(churn) * 100, 2) AS churn_percentage

    FROM customers

    GROUP BY tenure_group

    ORDER BY MIN(tenure);

    """

    return pd.read_sql(query, engine)