# Dataset Preprocessing & Validation Report
**Project:** Conversational Liveness & Social Engineering Detector  
**Stage:** Step 2 — Dataset Validation, Leakage Isolation & Train/Val/Test Preparation  
**Date:** September 2026  
**Status:** Validated, Processed & Immutability Verified  

---

## Executive Summary

This report documents the schema validation, critical data-leakage analysis, text preprocessing, feature engineering, and stratified splitting conducted on the benchmark dataset `data/deepfake_conversation_factor_analysis.csv`. 

Per project requirements:
- **No ML models have been trained.**
- **No fake evaluation metrics are reported.**
- **The original CSV has NOT been modified or overwritten** (verified via SHA256 cryptographic checksum).
- Deterministic leakage columns have been strictly isolated and excluded.
- The pipeline establishes an explicit boundary enabling a future **Text-Only MVP** prediction interface.

---

## 1. Dataset Dimensions & Overview

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Source File** | `data/deepfake_conversation_factor_analysis.csv` | Original raw dataset |
| **File SHA256 Checksum** | `ec96f27238a56406482fc5257e1d273b607577fff177e587e51d0da0af217344` | Cryptographically verified unchanged |
| **Total Rows** | 20,000 | Exactly 20,000 conversational records |
| **Total Columns** | 15 | Original schema |
| **Duplicate Rows** | 0 | Zero duplicate rows detected |
| **Empty Text Rows** | 0 | `conversation_text` contains 0 empty or whitespace-only records |

---

## 2. Dataset Schema & Column Catalog

The raw dataset contains 15 columns with the following types and roles:

| Column Name | Data Type | Missing Count | Role in Project | Description |
| :--- | :--- | :--- | :--- | :--- |
| `conversation_id` | `int64` | 0 (0.0%) | Identifier | Unique conversation index (excluded from ML features) |
| `conversation_text` | `object` (string) | 0 (0.0%) | **Primary ML Feature** | Raw conversational scenario text |
| `task_domain` | `object` (string) | 0 (0.0%) | Tabular Metadata | Scenario context (e.g. pharmacy, banking, clinic) |
| `channel` | `object` (string) | 0 (0.0%) | Tabular Metadata | Transmission channel (e.g. Email, WhatsApp, Phone) |
| `Conversational Authenticity` | `object` (string) | 0 (0.0%) | **Primary Target** | 3 classes: Real, Suspicious, Fake |
| `Conversation Type` | `object` (string) | 0 (0.0%) | **Excluded (Leakage)** | Deterministic mapping to Primary Target |
| `Communication Medium` | `object` (string) | 0 (0.0%) | Tabular Metadata | Medium format (Text, Voice, Synthetic Audio) |
| `Manipulation Type` | `object` (string) | 5,286 (26.4%) | **Excluded (Leakage)** | Missingness directly reveals Real records |
| `Malicious` | `object` (string) | 6,983 (34.9%) | **Excluded (Leakage)** | Attack taxonomy; 100% missing in Real records |
| `Threat Severity` | `object` (string) | 0 (0.0%) | **Secondary Target** | 4 classes: Low, Medium, High, Critical |
| `Impersonation` | `object` (string) | 6,585 (32.9%) | **Excluded (Leakage)** | Persona spoofed; 100% missing in Real records |
| `urgency_score_0_10` | `int64` | 0 (0.0%) | Tabular Metadata | Annotated urgency score (0 to 10) |
| `Emotional State` | `object` (string) | 0 (0.0%) | Tabular Metadata | Emotional manipulation context |
| `Verification Behavior` | `object` (string) | 0 (0.0%) | Tabular Metadata | Behavior observed during verification check |
| `suggested_action` | `object` (string) | 0 (0.0%) | **Excluded (Leakage)** | 1-to-1 deterministic mapping to Threat Severity |

---

## 3. Missing Value Analysis

Three columns contain missing values in the raw dataset:
1. `Manipulation Type`: 5,286 missing (26.43%)
2. `Malicious`: 6,983 missing (34.92%)
3. `Impersonation`: 6,585 missing (32.93%)

> [!CAUTION]
> **Missingness Mechanism (MNAR - Missing Not At Random):**  
> All 5,286 records with `Conversational Authenticity == 'Real'` have `NaN` for `Manipulation Type`, `Malicious`, and `Impersonation`. Because the absence of a value directly identifies genuine human dialogue, treating missing values with standard imputation (e.g. filling with `"Unknown"` or constant) creates severe target leakage. These three columns are strictly excluded from the ML feature set.

All remaining 12 columns contain 0 missing values. Crucially, the primary NLP input `conversation_text` has **zero** missing values.

---

## 4. Target Distributions

### 4.1 Primary Target: `Conversational Authenticity`

| Class | Count | Percentage | Class Description |
| :--- | :--- | :--- | :--- |
| **Fake** | 12,259 | 61.30% | Deepfake or fully synthetic AI generation |
| **Real** | 5,286 | 26.43% | Authentic human communication |
| **Suspicious** | 2,455 | 12.28% | Human-AI mixed manipulation or hybrid editing |
| **Total** | 20,000 | 100.00% | Class imbalance: ~5:2:1 ratio (handled via stratified splitting) |

### 4.2 Secondary Target: `Threat Severity`

| Class | Count | Percentage | Operational Meaning |
| :--- | :--- | :--- | :--- |
| **High** | 5,691 | 28.45% | Significant risk requiring active user warning |
| **Low** | 5,531 | 27.65% | Minimal risk; routine authentic traffic |
| **Critical** | 4,598 | 22.99% | Imminent fraud/takeover requiring immediate blocking |
| **Medium** | 4,180 | 20.90% | Ambiguous risk requiring step-up verification |
| **Total** | 20,000 | 100.00% | Well-balanced (~21% - 28.5% across all 4 levels) |

---

## 5. Critical Data-Leakage Audit & Excluded Columns

A core requirement of this engineering step is preventing data leakage that would result in artificially inflated accuracy and catastrophic failure in real-world deployment.

### Excluded Columns Summary

| Column Name | Reason for Exclusion | Mathematical / Empirical Proof |
| :--- | :--- | :--- |
| `conversation_id` | Non-generalizable identifier | Arbitrary integer sequence. |
| `suggested_action` | **1-to-1 Bijection with Threat Severity** | `Allow` $\equiv$ `Low` (5,531), `Verify First` $\equiv$ `Medium` (4,180), `Warn User` $\equiv$ `High` (5,691), `Block and Escalate` $\equiv$ `Critical` (4,598). Mutual information $= 1.0$. |
| `Conversation Type` | **Deterministic Mapping to Authenticity** | `Genuine Human Conversation` $\equiv$ `Real` (5,286), `Human + AI Edited Conversation` $\equiv$ `Suspicious` (2,455), all others $\equiv$ `Fake` (12,259). Mutual information $= 1.0$. |
| `Manipulation Type` | **Deterministic Missingness & Partitioning** | 100% of `Real` records are `NaN`. Specific categories partition `Suspicious` from `Fake` with zero overlap. |
| `Malicious` | **Label Outcome Annotation** | Annotator's post-hoc classification of fraud category. 100% missing for `Real` records. |
| `Impersonation` | **Label Outcome Annotation** | Annotator's post-hoc persona tag. 100% missing for `Real` records. |
| `Conversational Authenticity` | **Primary Target Variable** | Must never be an input feature. |
| `Threat Severity` | **Secondary Target Variable** | Must never be an input feature. |

> [!IMPORTANT]
> None of the above columns are included in the feature matrices for model training. Furthermore, the 5 deterministic leakage columns (`suggested_action`, `Conversation Type`, `Manipulation Type`, `Malicious`, `Impersonation`) have been completely excised from the processed split files in `data/processed/` to guarantee they cannot be loaded by automated feature engineering.

---

## 6. Text Feature & Text-Only MVP Architecture

In a production environment, the detector will receive **raw conversational text** submitted by end-users or extracted from chat/email webhooks. In real time, metadata fields like `channel`, `task_domain`, or `Verification Behavior` are often unknown or unreliably labeled.

Therefore:
- The primary model will accept **only conversation text** (`clean_conversation_text`).
- Secondary metadata columns (`task_domain`, `channel`, `Communication Medium`, `Emotional State`, `Verification Behavior`, `urgency_score_0_10`) are preserved in `data/processed/` purely for exploratory analysis and offline academic benchmarking, but are excluded from the primary inference pipeline.

---

## 7. Text Preprocessing & Engineered Lexical Features

### 7.1 Text Preprocessing Function (`clean_conversation_text`)
Located in `backend/ml/data_utils.py`:
1. Safely coerces null/NaN values to empty strings.
2. Normalizes non-standard whitespace, tabs, and line breaks into clean single-space formatting.
3. Trims leading and trailing whitespace.
4. **Preserves meaningful signals**: Punctuation (`!`, `?`), digits, currency symbols (`$`, `€`, `£`), URLs, and security keywords (`OTP`, `PIN`, `password`, `login`).
5. **No aggressive stemming or lemmatization**: Per guidelines, text is maintained in natural readable form to preserve grammatical anomalies, synthetic repetition, and subtle phrasing patterns indicative of AI generation.

### 7.2 Engineered Lexical Features (`extract_lexical_features`)
Nine interpretable lexical features were engineered to complement future NLP TF-IDF representations:

| Feature Name | Type | Description / Extraction Logic | Relevance to Social Engineering |
| :--- | :--- | :--- | :--- |
| `text_length` | Integer | Total character count (`len(text)`) | Text verbosity and synthetic elaboration |
| `word_count` | Integer | Whitespace-delimited token count | Message depth |
| `exclamation_count` | Integer | Count of `!` characters | Urgency and high emotional pressure |
| `question_count` | Integer | Count of `?` characters | Prying behavior or verification inquiry |
| `uppercase_ratio` | Float | Ratio of uppercase to alphabetic chars | Screaming / authoritative intimidation |
| `digit_count` | Integer | Count of numeric characters | Reference IDs, financial sums, OTPs |
| `url_present` | Binary (0/1) | Regex match for `http://`, `https://`, `www.` | Phishing links, credential harvesters |
| `urgency_keyword_present` | Binary (0/1) | Matches: `urgent`, `immediately`, `asap`, `fast`, `expire`, etc. | Artificial time-pressure cues |
| `sensitive_request_keyword_present` | Binary (0/1) | Matches: `password`, `otp`, `pin`, `credit card`, `cvv`, `wallet`, etc. | Direct credential/financial harvesting |

---

## 8. Narrative / Meta-Commentary Leakage Analysis

### 8.1 The Phenomenon
Detailed lexical inspection revealed that this synthetic benchmark dataset includes LLM scenario generation artifacts where the text contains explicit evaluative commentary. For example:
- *"Taha follows up with the pharmacy desk about checking whether a refund was processed, but the message remains safe because it does not request passwords..."*
- *"The exchange becomes suspicious when it connects medical lab report with a sudden warning..."*
- *"A fake-looking conversation about clinic follow-up asks the receiver to pay a small release fee..."*

### 8.2 Audit Statistics

| Analytical Keyword | Occurrences in `conversation_text` |
| :--- | :--- |
| `suspicious` | 5,311 |
| `risk` | 4,295 |
| `safe` | 4,206 |
| `fake` | 1,214 |
| `genuine` | 818 |
| `ai-generated` | 424 |
| `edited by ai` | 348 |
| **Total Flagged Rows** | **13,839 (69.19% of entire dataset)** |

### 8.3 Mitigation Strategy
1. **Source Immutability**: The original dataset remains 100% untouched. No records were deleted or modified in place.
2. **Leakage Flagging**: Each row in `data/processed/` includes a boolean indicator `has_narrative_leakage` to enable stratified error analysis during model evaluation.
3. **Dual Text Representations**:
   - `clean_conversation_text`: Preserves complete normalized text for benchmark reproducibility.
   - `sanitized_conversation_text`: Applies `sanitize_narrative_leakage()` to strip known meta-commentary preamble clauses (e.g., *"the message remains safe because"*, *"the exchange becomes suspicious when"*) for unbiased ablation experiments.

---

## 9. Train / Validation / Test Dataset Splits

### 9.1 Splitting Methodology
- **Train Split**: 70% (14,000 records)
- **Validation Split**: 15% (3,000 records)
- **Test Split**: 15% (3,000 records)
- **Stratification**: Performed strictly on `Conversational Authenticity` (Primary Target) using two-stage `sklearn.model_selection.train_test_split`.
- **Random Seed**: Fixed at `42` for exact reproducibility.
- **Transformer Isolation**: No feature transformers (e.g. TF-IDF vectorizer, Scaler, or Encoder) were fitted prior to splitting, strictly eliminating cross-split information leakage.

### 9.2 Stratification Verification

| Split | Total Rows | Fake Count (%) | Real Count (%) | Suspicious Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 14,000 | 8,581 (61.29%) | 3,700 (26.43%) | 1,719 (12.28%) |
| **Validation** | 3,000 | 1,839 (61.30%) | 793 (26.43%) | 368 (12.27%) |
| **Test** | 3,000 | 1,839 (61.30%) | 793 (26.43%) | 368 (12.27%) |
| **Full Dataset** | 20,000 | 12,259 (61.30%) | 5,286 (26.43%) | 2,455 (12.28%) |

Proportions are identical across all partitions to within 0.01%.

---

## 10. Summary of Generated Artifacts

The following files were created in `data/processed/`:

1. `data/processed/train.csv` (14,000 rows, 21 columns, 9.58 MB)
2. `data/processed/val.csv` (3,000 rows, 21 columns, 2.05 MB)
3. `data/processed/test.csv` (3,000 rows, 21 columns, 2.05 MB)
4. `data/processed/dataset_summary.json` (Structured JSON metadata and SHA256 verification receipt)

### Active Columns in Processed Files (21 columns)
- **Identifier**: `conversation_id`
- **Targets (2)**: `Conversational Authenticity`, `Threat Severity`
- **Text Features (2)**: `clean_conversation_text`, `sanitized_conversation_text`
- **Leakage Flag (1)**: `has_narrative_leakage`
- **Lexical Features (9)**: `text_length`, `word_count`, `exclamation_count`, `question_count`, `uppercase_ratio`, `digit_count`, `url_present`, `urgency_keyword_present`, `sensitive_request_keyword_present`
- **Tabular Metadata (6)**: `task_domain`, `channel`, `Communication Medium`, `Emotional State`, `Verification Behavior`, `urgency_score_0_10`

---

## 11. Assumptions & Recommendations for Model Training (Step 3)

1. **Text-Only Baseline First**: The initial model benchmark should evaluate `clean_conversation_text` using TF-IDF (word n-grams) combined with lexical features on the training set only.
2. **Narrative Leakage Benchmark**: Evaluate the model trained on `clean_conversation_text` vs `sanitized_conversation_text` to measure the extent of shortcut learning induced by meta-commentary cues.
3. **No Target Encoding Across Splits**: Any TF-IDF vectorizer or vocabulary mapping must be `.fit()` exclusively on `train.csv` and `.transform()` only on `val.csv` and `test.csv`.
4. **Primary vs Secondary Targets**: Separate classification heads or cascaded pipelines should be used for `Conversational Authenticity` (3-class) and `Threat Severity` (4-class).
