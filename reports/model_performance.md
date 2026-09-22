# Model Performance Report

## 1. Dataset Used
- **Source**: `Conversational Liveness & Social Engineering Detector`
- **Train Split**: `data/processed/train.csv`
- **Validation Split**: `data/processed/val.csv`
- **Test Split**: `data/processed/test.csv` (strictly unseen during training/tuning)
- **Input Feature**: `clean_conversation_text` (all excluded/leakage columns strictly avoided)
- **Target Label**: `Conversational Authenticity` (Classes: Fake, Real, Suspicious)

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
| **Train** | 14,000 | 70.0% |
| **Validation** | 3,000 | 15.0% |
| **Test** | 3,000 | 15.0% |
| **Total** | 20,000 | 100.0% |

## 5. Accuracy
- **Test Accuracy**: **1.0000** (100.00%)
- **Formula**: `Correct Predictions / Total Test Samples` = `3000 / 3000`

## 6. Precision
- **Precision (Macro Averaging)**: **1.0000**
  - Unweighted arithmetic mean across classes: `(Precision_Fake + Precision_Real + Precision_Suspicious) / 3`
- **Precision (Weighted Averaging)**: **1.0000**
  - Average of class precisions weighted by each class's test support

## 7. Recall
- **Recall (Macro Averaging)**: **1.0000**
  - Unweighted arithmetic mean across classes: `(Recall_Fake + Recall_Real + Recall_Suspicious) / 3`
- **Recall (Weighted Averaging)**: **1.0000**
  - Average of class recalls weighted by each class's test support

## 8. F1-Score
- **F1-Score (Macro Averaging)**: **1.0000**
  - Harmonic mean of Macro Precision and Macro Recall
- **F1-Score (Weighted Averaging)**: **1.0000**
  - Harmonic mean weighted by each class's test support

### Per-Class Metrics (Classification Report)
```
              precision    recall  f1-score   support

        Fake     1.0000    1.0000    1.0000      1839
        Real     1.0000    1.0000    1.0000       793
  Suspicious     1.0000    1.0000    1.0000       368

    accuracy                         1.0000      3000
   macro avg     1.0000    1.0000    1.0000      3000
weighted avg     1.0000    1.0000    1.0000      3000

```

## 9. Confusion Matrix
| True \ Predicted | Fake | Real | Suspicious | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Fake** | 1839 | 0 | 0 | 1839 |
| **Real** | 0 | 793 | 0 | 793 |
| **Suspicious** | 0 | 0 | 368 | 368 |

*Rows represent ground-truth classes; columns represent model predictions.*

## 10. Short Interpretation
- **Model Discriminative Power**: The TF-IDF + Logistic Regression baseline achieved **100.00%** test accuracy and **1.0000** Macro F1-score across all 3 classes on the test set.
- **Leakage Isolation**: Feature space was strictly restricted to `clean_conversation_text`. No excluded/leakage columns (`conversation_id`, `suggested_action`, `Conversation Type`, `Manipulation Type`, `Malicious`, `Impersonation`, `Threat Severity`, or tabular metadata) were provided to the model.
- **Data Characteristic Warning**: As audited in Step 2 (`reports/dataset_preprocessing_report.md`), the underlying synthetic benchmark dataset contains explicit evaluative meta-commentary phrases within `conversation_text` (e.g., *"The dialogue is consistent with genuine communication..."*, *"A fake-looking conversation..."*). While this allows TF-IDF n-grams to achieve perfect classification on this benchmark corpus, real-world deployment will require sanitized conversational turns or adversarial robustness checks.
- **Deployment Status**: The trained model (`model.pkl`) and fitted vectorizer (`tfidf_vectorizer.pkl`) have been serialized to `backend/ml/artifacts/` for downstream inference.
