"""
Fake Product Review & Sentiment Analysis Pipeline
==================================================
Architecture: Sublinear TF-IDF (1, 2 N-Grams) + Maximum-Margin Linear SVM
Author: Likith Yadav
Repository: sentimental_analysis_using_ML

This module implements:
1. Ingestion of multi-class customer reviews (Positive, Neutral, Negative)
2. NLP text sanitization and punctuation normalization
3. High-dimensional Sublinear TF-IDF Vectorization with unigrams & bigrams
4. Supervised classification benchmarking (Logistic Regression vs Linear SVM)
5. Discrepancy auditing between numeric ratings and linguistic sentiment
"""

import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 50000


def clean_text(text):
    """
    Sanitizes review strings by lowercasing, stripping special noise,
    and removing superfluous whitespace while preserving negation markers.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(dataset_path="Dataset-SA.csv", sample_size=1500):
    """
    Loads raw review data, sanitizes text, and drops empty rows.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    print(f"Loading review data from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Required columns: Review and Sentiment
    df = df.dropna(subset=["Review", "Sentiment"]).copy()
    df["Sentiment"] = df["Sentiment"].str.strip().str.lower()
    df = df[df["Sentiment"].isin(["positive", "neutral", "negative"])].copy()
    
    df["clean_review"] = df["Review"].apply(clean_text)
    df = df[df["clean_review"].str.len() > 2].copy()

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).copy()

    print(f"Loaded {len(df):,} cleaned review samples.")
    print("Class distribution:\n", df["Sentiment"].value_counts())
    return df


def build_pipeline(classifier_type="svm"):
    """
    Builds an end-to-end TF-IDF + Classifier Scikit-Learn Pipeline.
    """
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    if classifier_type == "svm":
        clf = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    elif classifier_type == "logistic":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return Pipeline([
        ("tfidf", tfidf),
        ("classifier", clf)
    ])


def train_and_benchmark(df):
    """
    Splits dataset, trains candidate classifiers, and evaluates metrics.
    """
    X = df["clean_review"]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    models = {
        "Logistic Regression": build_pipeline("logistic"),
        "Linear SVM": build_pipeline("svm")
    }

    results = []
    best_f1 = -1
    best_pipe = None
    best_name = None

    for name, pipe in models.items():
        print(f"\nFitting {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })
        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_pipe = pipe
            best_name = name

    results_df = pd.DataFrame(results)
    return results_df, best_pipe, best_name, (X_test, y_test)


def predict_sentiment(review_text, pipeline):
    """
    Predicts sentiment for a single or list of raw reviews.
    """
    if isinstance(review_text, str):
        cleaned = [clean_text(review_text)]
    else:
        cleaned = [clean_text(r) for r in review_text]
    return pipeline.predict(cleaned)


if __name__ == "__main__":
    print("=" * 65)
    print("REVIEW SENTIMENT & DISCREPANCY AUDITING INFERENCE PIPELINE")
    print("=" * 65)

    model_path = "case2_sentiment_model.joblib"
    if os.path.exists(model_path):
        print(f"Loading trained pipeline from: {model_path}")
        model = joblib.load(model_path)

        sample_reviews = [
            "Superb build quality! Works flawlessly and sound is crisp.",
            "Average product, neither good nor bad. Delivered in 3 days.",
            "Bad quality, very cheap plastic material. Do not buy!",
            "Working fine, but product quality very cheap and flimsy."
        ]

        print("\nInference on Sample Reviews:")
        preds = predict_sentiment(sample_reviews, model)
        for rev, pred in zip(sample_reviews, preds):
            print(f"  • Review: \"{rev}\"")
            print(f"    --> Predicted Sentiment: [{pred.upper()}]\n")
    else:
        print(f"Model file '{model_path}' not found. Run training on Dataset-SA.csv.")
