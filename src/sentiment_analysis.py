import re
import nltk
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Initialize VADER Sentiment Analyzer

sia = SentimentIntensityAnalyzer()

print("Sentiment Analyzer Initialized Successfully!")

def preprocess_text(text):

    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords but preserve negations
    stop_words = set(stopwords.words('english'))

    negations = {'not', 'no', 'nor', 'never'}

    stop_words = stop_words - negations

    filtered_words = [
        word for word in tokens
        if word not in stop_words
    ]

    return ' '.join(filtered_words)

def analyze_sentiment(text):

    scores = sia.polarity_scores(str(text))

    compound_score = scores['compound']

    if compound_score >= 0.05:

        sentiment = 'Positive'

    elif compound_score <= -0.05:

        sentiment = 'Negative'

    else:

        sentiment = 'Neutral'

    return sentiment, compound_score

def analyze_reviews(df):

    df = df.copy()

    # Text preprocessing
    df['cleaned_review'] = df['review_text'].apply(
        preprocess_text
    )

    # Sentiment analysis
    results = df['review_text'].apply(
        analyze_sentiment
    )

    df['sentiment'] = results.apply(
        lambda x: x[0]
    )

    df['sentiment_score'] = results.apply(
        lambda x: x[1]
    )

    return df

import joblib
from pathlib import Path

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"


# Load trained model
sentiment_model = joblib.load(MODEL_PATH)

# Load TF-IDF vectorizer
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment_ml(review_text):

    # Convert review into numerical features
    review_vector = tfidf_vectorizer.transform([str(review_text)])

    # Predict sentiment
    prediction = sentiment_model.predict(review_vector)[0]

    # Get prediction probability
    probabilities = sentiment_model.predict_proba(review_vector)[0]

    confidence = max(probabilities)

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "sentiment": sentiment,
        "confidence": round(float(confidence), 4)
    }


def analyze_reviews_ml(df):

    df = df.copy()

    results = df["review_text"].apply(predict_sentiment_ml)

    df["ml_sentiment"] = results.apply(lambda x: x["sentiment"])

    df["ml_confidence"] = results.apply(lambda x: x["confidence"])

    return df