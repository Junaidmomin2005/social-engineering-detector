"""
Model Training Module:
Trains the baseline NLP + ML classifier for Conversational Authenticity.

Pipeline:
- Feature extraction: TF-IDF on `clean_conversation_text`
- Model: LogisticRegression
- Artifacts: Saved to backend/ml/artifacts/
"""

import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


# Project root and artifact paths
BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"
VAL_DATA_PATH = BASE_DIR / "data" / "processed" / "val.csv"
ARTIFACTS_DIR = BASE_DIR / "backend" / "ml" / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"

TARGET_COLUMN = "Conversational Authenticity"
TEXT_COLUMN = "clean_conversation_text"


def train_pipeline():
    print(f"Loading training data from: {TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Loading validation data from: {VAL_DATA_PATH}")
    val_df = pd.read_csv(VAL_DATA_PATH)

    print(f"Training set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")

    # Prepare text and targets
    X_train_text = train_df[TEXT_COLUMN].fillna("").astype(str)
    y_train = train_df[TARGET_COLUMN].astype(str)

    X_val_text = val_df[TEXT_COLUMN].fillna("").astype(str)
    y_val = val_df[TARGET_COLUMN].astype(str)

    print(f"\nTarget distribution (Train):\n{y_train.value_counts()}")
    print(f"\nTarget distribution (Validation):\n{y_val.value_counts()}")

    # TF-IDF Vectorizer configuration
    # word n-grams: 1–2, max_features: 5000, lowercase: True, English stop words, sublinear_tf: True
    print("\nInitializing TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        lowercase=True,
        stop_words="english",
        sublinear_tf=True,
    )

    print("Fitting TF-IDF Vectorizer ONLY on training text...")
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    print(f"Transformed train matrix shape: {X_train_tfidf.shape}")

    print("Transforming validation text using fitted vectorizer...")
    X_val_tfidf = vectorizer.transform(X_val_text)
    print(f"Transformed validation matrix shape: {X_val_tfidf.shape}")

    # Logistic Regression Classifier
    print("\nTraining LogisticRegression classifier...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)

    # Validation evaluation
    y_val_pred = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
    val_f1_weighted = f1_score(y_val, y_val_pred, average="weighted")

    print("\n--- Validation Performance ---")
    print(f"Validation Accuracy:    {val_accuracy:.4f}")
    print(f"Validation Macro F1:    {val_f1_macro:.4f}")
    print(f"Validation Weighted F1: {val_f1_weighted:.4f}")
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, y_val_pred, digits=4))

    # Serialize artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving vectorizer to {VECTORIZER_PATH}...")
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    print("Training pipeline completed successfully.")
    return model, vectorizer


if __name__ == "__main__":
    train_pipeline()
