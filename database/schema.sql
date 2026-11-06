CREATE TABLE IF NOT EXISTS customers (

    customer_id VARCHAR(20) PRIMARY KEY,

    gender VARCHAR(10),

    senior_citizen INTEGER,

    partner VARCHAR(10),

    dependents VARCHAR(10),

    tenure INTEGER,

    phone_service VARCHAR(10),

    multiple_lines VARCHAR(30),

    internet_service VARCHAR(30),

    online_security VARCHAR(30),

    online_backup VARCHAR(30),

    device_protection VARCHAR(30),

    tech_support VARCHAR(30),

    streaming_tv VARCHAR(30),

    streaming_movies VARCHAR(30),

    contract VARCHAR(30),

    paperless_billing VARCHAR(10),

    payment_method VARCHAR(50),

    monthly_charges NUMERIC(10,2),

    total_charges NUMERIC(10,2),

    churn INTEGER

);

CREATE TABLE IF NOT EXISTS predictions (

    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    customer_id VARCHAR(20),

    churn_probability NUMERIC(10,6),

    predicted_churn INTEGER,

    model_name VARCHAR(100),

    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)

);

CREATE TABLE IF NOT EXISTS customer_reviews (

    review_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    customer_id VARCHAR(20),

    review_text TEXT,

    sentiment VARCHAR(20),

    sentiment_score NUMERIC(10,6),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)

);