# Customer Intelligence & Churn Prediction System

An end-to-end customer analytics and machine learning application designed to analyze customer behavior, predict customer churn, and identify customer sentiment through Natural Language Processing.

## Project Overview

The Customer Intelligence & Churn Prediction System combines machine learning, NLP, data visualization, and PostgreSQL to help businesses understand customer behavior and identify customers who are likely to discontinue their services.

The application provides interactive dashboards for customer analytics, sentiment analysis, churn prediction, prediction history, and model performance evaluation.

## Key Features

### 1. Customer Analytics Dashboard
- Analyze customer demographics and subscription patterns.
- Visualize churn distribution.
- Analyze churn across contract types, payment methods, and internet services.
- Identify high-risk customer segments using SQL analytics.

### 2. Customer Sentiment Analysis
- Analyze customer reviews using VADER sentiment analysis.
- Classify reviews using machine learning.
- Compare sentiment predictions from different approaches.
- Visualize positive and negative customer feedback.

### 3. Churn Prediction
- Predict whether a customer is likely to leave.
- Generate churn probabilities using a trained Logistic Regression pipeline.
- Categorize customers into risk levels.
- Explain predictions using SHAP.

### 4. Prediction History
- Store customer churn predictions in PostgreSQL.
- Track previous predictions.
- Analyze prediction trends.
- Export prediction history.

### 5. Model Performance
- Evaluate the churn prediction model.
- Display accuracy, precision, recall, and F1-score.
- Visualize the confusion matrix and ROC curve.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| NLP | NLTK, VADER, TF-IDF |
| Explainable AI | SHAP |
| Database | PostgreSQL |
| Database Integration | SQLAlchemy, Psycopg2 |
| Web Framework | Streamlit |
| Model Storage | Joblib / Pickle |

## Machine Learning Models

### Customer Churn Prediction
The project explores multiple classification algorithms, including:

- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Machine
- XGBoost

The final saved churn model is a Logistic Regression pipeline with preprocessing.

### Customer Sentiment Classification

The sentiment analysis component evaluates:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine

TF-IDF is used for text feature extraction.

## Explainable AI

SHAP (SHapley Additive exPlanations) is integrated to explain individual churn predictions and identify the features contributing to the model's output.

## Database Integration

PostgreSQL is used to store and manage:

- Customer information
- Customer reviews
- Sentiment predictions
- Churn prediction history

SQL queries are used to generate analytical insights for the dashboard.

## Project Structure

```text
Customer_Intelligence_System/
│
├── app/
│   ├── dashboard.py
│   ├── prediction.py
│   ├── sentiment.py
│   └── sql_analytics.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── database/
│   ├── db_connection.py
│   ├── queries.sql
│   └── schema.sql
│
├── models/
│   ├── churn_model.pkl
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── shap_background.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 07_postgresql_integration.ipynb
│
├── pages/
│   ├── 1_Customer_Analytics.py
│   ├── 2_Customer_Sentiment.py
│   ├── 3_Churn_Prediction.py
│   ├── 4_Prediction_History.py
│   └── 5_Model_Performance.py
│
├── src/
│   ├── model_training.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── sentiment_analysis.py
│   └── visualization.py
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sarthak2405u/customer-churn-prediction-system.git
```

### 2. Navigate to the Project Directory

```bash
cd Customer_Intelligence_System
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure PostgreSQL

Create a PostgreSQL database named:

```sql
CREATE DATABASE customer_intelligence;
```

Configure the database credentials in a `.env` file:

```env
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=customer_intelligence
```

Run the SQL scripts in the database folder to create the required tables.

### 6. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser.

## Datasets

The project uses customer churn data and customer review data.

- Customer churn dataset: Used for customer behavior analysis and churn prediction.
- Customer reviews dataset: Used for sentiment analysis and NLP-based classification.

The sentiment classifier is trained on general review datasets. The sample customer reviews included in the application are illustrative and should not be treated as representative business feedback.

## Future Improvements

- Deploy the application to a cloud platform.
- Integrate real-time customer feedback.
- Implement automated customer retention recommendations.
- Improve model performance with additional customer data.
- Add user authentication and role-based access.
- Integrate automated email alerts for high-risk customers.

## Author

Developed as an end-to-end machine learning and customer analytics portfolio project.

## License

This project is intended for educational and portfolio purposes.