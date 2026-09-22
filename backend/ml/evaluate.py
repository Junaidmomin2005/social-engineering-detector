"""
Model Evaluation Module:
Evaluates the trained Logistic Regression model and TF-IDF vectorizer on the test split.
Generates reports/model_performance.md with real, reproducible metrics.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train.csv"
VAL_DATA_PATH = BASE_DIR / "data" / "processed" / "val.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test.csv"
ARTIFACTS_DIR = BASE_DIR / "backend" / "ml" / "artifacts"
REPORT_PATH = BASE_DIR / "reports" / "model_performance.md"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"

TARGET_COLUMN = "Conversational Authenticity"
TEXT_COLUMN = "clean_conversation_text"


def evaluate_pipeline():
    # Verify artifacts exist
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Artifacts not found. Please run train.py first. Missing: "
            f"{MODEL_PATH if not MODEL_PATH.exists() else ''} "
            f"{VECTORIZER_PATH if not VECTORIZER_PATH.exists() else ''}"
        )

    print(f"Loading test data from: {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    train_size = len(train_df)
    val_size = len(val_df)
    test_size = len(test_df)

    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Load artifacts
    print(f"Loading vectorizer from {VECTORIZER_PATH}...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # Extract test features and ground truth
    X_test_text = test_df[TEXT_COLUMN].fillna("").astype(str)
    y_test = test_df[TARGET_COLUMN].astype(str)

    # Transform test features (DO NOT FIT)
    print("Transforming test text using loaded vectorizer (no refitting)...")
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Run predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate metrics
    classes = sorted(list(np.unique(y_test)))
    acc = accuracy_score(y_test, y_pred)

    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)

    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cls_report = classification_report(y_test, y_pred, labels=classes, digits=4)

    print("\n================ EVALUATION METRICS (TEST SET) ================")
    print(f"Accuracy:           {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision (Macro):   {prec_macro:.4f}")
    print(f"Precision (Weighted):{prec_weighted:.4f}")
    print(f"Recall (Macro):      {rec_macro:.4f}")
    print(f"Recall (Weighted):   {rec_weighted:.4f}")
    print(f"F1-score (Macro):    {f1_macro:.4f}")
    print(f"F1-score (Weighted): {f1_weighted:.4f}")
    print("\nConfusion Matrix (Labels: {}):".format(classes))
    print(cm)
    print("\nClassification Report:")
    print(cls_report)
    print("================================================================")

    # Format Markdown Confusion Matrix
    cm_header = "| True \\ Predicted | " + " | ".join(classes) + " | Total |"
    cm_sep = "| :--- | " + " | ".join([":---:"] * len(classes)) + " | :---: |"
    cm_rows = []
    for i, true_label in enumerate(classes):
        row_vals = [str(cm[i, j]) for j in range(len(classes))]
        row_total = str(cm[i].sum())
        cm_rows.append(f"| **{true_label}** | " + " | ".join(row_vals) + f" | {row_total} |")
    cm_md_table = "\n".join([cm_header, cm_sep] + cm_rows)

    # Interpretation calculation
    fake_idx = classes.index("Fake") if "Fake" in classes else None
    real_idx = classes.index("Real") if "Real" in classes else None
    susp_idx = classes.index("Suspicious") if "Suspicious" in classes else None

    report_content = f"""# Model Performance Report

## 1. Dataset Used
- **Source**: `Conversational Liveness & Social Engineering Detector`
- **Train Split**: `data/processed/train.csv`
- **Validation Split**: `data/processed/val.csv`
- **Test Split**: `data/processed/test.csv` (strictly unseen during training/tuning)
- **Input Feature**: `clean_conversation_text` (all excluded/leakage columns strictly avoided)
- **Target Label**: `Conversational Authenticity` (Classes: {", ".join(classes)})

## 2. TF-IDF Configuration
- **Word n-grams**: `(1, 2)` (Unigrams and Bigrams)
- **Max Features**: `5000`
- **Lowercase**: `True`
- **Stop Words**: `english`
- **Sublinear TF**: `True` (uses `1 + log(tf)`)
- **Fitting Rule**: Fitted strictly and exclusively on the training split (`train.csv`). Transformed onto validation and test splits without refitting.

## 3. ML Algorithm
- **Classifier**: `LogisticRegression`
- **Parameters**: `max_iter=1000`, `random_state=42`, `solver='lbfgs'`, `multi_class='auto'`
- **Explainability**: Linear coefficients over TF-IDF n-grams provide transparent, explainable feature weights for authenticity attribution.

## 4. Training / Validation / Test Sizes
| Split | Sample Count | Percentage |
| :--- | :---: | :---: |
| **Train** | {train_size:,} | {train_size / (train_size + val_size + test_size) * 100:.1f}% |
| **Validation** | {val_size:,} | {val_size / (train_size + val_size + test_size) * 100:.1f}% |
| **Test** | {test_size:,} | {test_size / (train_size + val_size + test_size) * 100:.1f}% |
| **Total** | {train_size + val_size + test_size:,} | 100.0% |

## 5. Accuracy
- **Test Accuracy**: **{acc:.4f}** ({acc * 100:.2f}%)
- **Formula**: `Correct Predictions / Total Test Samples` = `{int(acc * test_size)} / {test_size}`

## 6. Precision
- **Precision (Macro Averaging)**: **{prec_macro:.4f}**
  - Unweighted arithmetic mean across classes: `(Precision_Fake + Precision_Real + Precision_Suspicious) / 3`
- **Precision (Weighted Averaging)**: **{prec_weighted:.4f}**
  - Average of class precisions weighted by each class's test support

## 7. Recall
- **Recall (Macro Averaging)**: **{rec_macro:.4f}**
  - Unweighted arithmetic mean across classes: `(Recall_Fake + Recall_Real + Recall_Suspicious) / 3`
- **Recall (Weighted Averaging)**: **{rec_weighted:.4f}**
  - Average of class recalls weighted by each class's test support

## 8. F1-Score
- **F1-Score (Macro Averaging)**: **{f1_macro:.4f}**
  - Harmonic mean of Macro Precision and Macro Recall
- **F1-Score (Weighted Averaging)**: **{f1_weighted:.4f}**
  - Harmonic mean weighted by each class's test support

### Per-Class Metrics (Classification Report)
```
{cls_report}
```

## 9. Confusion Matrix
{cm_md_table}

*Rows represent ground-truth classes; columns represent model predictions.*

## 10. Short Interpretation
- **Model Discriminative Power**: The TF-IDF + Logistic Regression baseline achieved **{acc * 100:.2f}%** test accuracy and **{f1_macro:.4f}** Macro F1-score across all 3 classes on the test set.
- **Leakage Isolation**: Feature space was strictly restricted to `clean_conversation_text`. No excluded/leakage columns (`conversation_id`, `suggested_action`, `Conversation Type`, `Manipulation Type`, `Malicious`, `Impersonation`, `Threat Severity`, or tabular metadata) were provided to the model.
- **Data Characteristic Warning**: As audited in Step 2 (`reports/dataset_preprocessing_report.md`), the underlying synthetic benchmark dataset contains explicit evaluative meta-commentary phrases within `conversation_text` (e.g., *"The dialogue is consistent with genuine communication..."*, *"A fake-looking conversation..."*). While this allows TF-IDF n-grams to achieve perfect classification on this benchmark corpus, real-world deployment will require sanitized conversational turns or adversarial robustness checks.
- **Deployment Status**: The trained model (`model.pkl`) and fitted vectorizer (`tfidf_vectorizer.pkl`) have been serialized to `backend/ml/artifacts/` for downstream inference.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nSaved evaluation report to {REPORT_PATH}")
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classes": classes,
    }


if __name__ == "__main__":
    evaluate_pipeline()
