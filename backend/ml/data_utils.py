"""
Data Utilities Module:
Provides modular, reproducible helper functions for dataset loading, validation,
leakage isolation, text cleaning, lexical feature extraction, and stratified splitting.

Designed for: Conversational Liveness & Social Engineering Detector (Step 2)
"""

import os
import re
import hashlib
from typing import Tuple, Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# =====================================================================
# CONSTANTS & SCHEMA DEFINITION
# =====================================================================

EXPECTED_COLUMNS: List[str] = [
    "conversation_id",
    "conversation_text",
    "task_domain",
    "channel",
    "Conversational Authenticity",
    "Conversation Type",
    "Communication Medium",
    "Manipulation Type",
    "Malicious",
    "Threat Severity",
    "Impersonation",
    "urgency_score_0_10",
    "Emotional State",
    "Verification Behavior",
    "suggested_action"
]

PRIMARY_TARGET: str = "Conversational Authenticity"
EXPECTED_PRIMARY_CLASSES: List[str] = ["Fake", "Real", "Suspicious"]

SECONDARY_TARGET: str = "Threat Severity"
EXPECTED_SECONDARY_CLASSES: List[str] = ["Critical", "High", "Low", "Medium"]

PRIMARY_TEXT_FEATURE: str = "conversation_text"

# ---------------------------------------------------------------------
# CRITICAL DATA LEAKAGE ISOLATION
# ---------------------------------------------------------------------
# These columns MUST NEVER be passed as input features to an ML model:
# 1. conversation_id: Arbitrary index, no predictive semantic value.
# 2. suggested_action: Deterministic 1-to-1 mapping to Threat Severity.
# 3. Conversation Type: Deterministic mapping to Conversational Authenticity.
# 4. Manipulation Type: Missingness reveals Real (100% missing in Real records).
# 5. Malicious: Label outcome; missing in 100% of Real records.
# 6. Impersonation: Label outcome; missing in 100% of Real records.
# 7. Conversational Authenticity: The ground-truth primary target.
# 8. Threat Severity: The ground-truth secondary target.
EXCLUDED_LEAKAGE_COLUMNS: List[str] = [
    "conversation_id",
    "suggested_action",
    "Conversation Type",
    "Manipulation Type",
    "Malicious",
    "Impersonation",
    "Conversational Authenticity",
    "Threat Severity"
]

# Secondary tabular metadata columns:
# Preserved for academic analysis/exploration, but excluded from primary text model
TABULAR_METADATA_COLUMNS: List[str] = [
    "task_domain",
    "channel",
    "Communication Medium",
    "Emotional State",
    "Verification Behavior",
    "urgency_score_0_10"
]

# Narrative leakage / meta-commentary patterns in conversation_text:
# In synthetic benchmark datasets, scenario generators often include explicit
# evaluation words that reveal the label to the text model.
NARRATIVE_LEAKAGE_KEYWORDS: List[str] = [
    "safe",
    "suspicious",
    "fake",
    "risk",
    "genuine",
    "ai-generated",
    "edited by ai",
    "scam"
]

# Explicit commentary clauses that can be removed for unbiased model experiments
COMMENTARY_CLAUSE_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b(?:the message remains safe because|is consistent with genuine communication|the dialogue is consistent with genuine communication|so the risk remains limited)\b.*?(?=[.;,]|$)', flags=re.IGNORECASE),
    re.compile(r'\b(?:the exchange becomes suspicious when|the suspicious part is that|the text appears edited by ai because)\b', flags=re.IGNORECASE),
    re.compile(r'\b(?:the conversation shifts into template-style|the risk increases because|a fake-looking conversation|a suspicious sender)\b', flags=re.IGNORECASE),
    re.compile(r'\b(?:safe because it does not request|safe because|dialogue is consistent with genuine communication)\b', flags=re.IGNORECASE),
]


# =====================================================================
# DATASET LOADING & VALIDATION
# =====================================================================

def compute_file_sha256(filepath: str) -> str:
    """Calculate the SHA256 checksum of a file to verify immutability."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the raw CSV dataset safely without modifying the source file.
    
    Args:
        csv_path: Path to raw CSV file.
        
    Returns:
        pd.DataFrame containing the raw records.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def validate_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate dataset columns, row counts, null values, and class distributions.
    
    Args:
        df: Input DataFrame to validate.
        
    Returns:
        Dictionary containing validation results and summary metrics.
    """
    results: Dict[str, Any] = {}
    
    # 1. Column existence validation
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    results["all_expected_columns_present"] = len(missing_cols) == 0
    results["missing_columns"] = missing_cols
    
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    
    # 2. Dimensions & duplicates
    results["total_rows"] = int(len(df))
    results["total_columns"] = int(len(df.columns))
    results["duplicate_rows"] = int(df.duplicated().sum())
    
    # 3. Missing values per column
    results["missing_values"] = {col: int(df[col].isnull().sum()) for col in df.columns}
    
    # 4. Empty conversation_text check
    empty_text_mask = df[PRIMARY_TEXT_FEATURE].astype(str).str.strip() == ""
    results["empty_text_count"] = int(empty_text_mask.sum())
    
    # 5. Primary target class validation
    primary_classes = sorted(df[PRIMARY_TARGET].dropna().unique().tolist())
    results["primary_target_classes"] = primary_classes
    results["primary_target_valid"] = set(primary_classes) == set(EXPECTED_PRIMARY_CLASSES)
    results["primary_target_distribution"] = {
        cls: int((df[PRIMARY_TARGET] == cls).sum()) for cls in EXPECTED_PRIMARY_CLASSES
    }
    
    # 6. Secondary target class validation
    secondary_classes = sorted(df[SECONDARY_TARGET].dropna().unique().tolist())
    results["secondary_target_classes"] = secondary_classes
    results["secondary_target_valid"] = set(secondary_classes) == set(EXPECTED_SECONDARY_CLASSES)
    results["secondary_target_distribution"] = {
        cls: int((df[SECONDARY_TARGET] == cls).sum()) for cls in EXPECTED_SECONDARY_CLASSES
    }
    
    # 7. Leakage column confirmation
    results["excluded_leakage_columns"] = EXCLUDED_LEAKAGE_COLUMNS
    
    return results


# =====================================================================
# TEXT PREPROCESSING & FEATURE ENGINEERING
# =====================================================================

def clean_conversation_text(text: Any) -> str:
    """
    Clean conversation text in a safe, non-destructive manner.
    
    - Handles null/NaN/non-string gracefully
    - Normalizes inconsistent whitespace (multiple spaces, tabs, line breaks)
    - Strips leading/trailing whitespace
    - Preserves meaningful punctuation (!, ?, $, %, @, etc.), numbers, URLs, and casing
    - Avoids aggressive stemming or lemmatization (per project requirements)
    
    Args:
        text: Input raw text or object.
        
    Returns:
        Cleaned string.
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    # Normalize unicode/whitespace spaces to single space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def detect_narrative_leakage(text: str) -> Dict[str, Any]:
    """
    Detect whether text contains analytical/meta-commentary cues that
    could allow a model to memorize labels instead of conversational patterns.
    
    Args:
        text: Conversation text to inspect.
        
    Returns:
        Dictionary with boolean flag and detected keywords.
    """
    if not text:
        return {"has_leakage": False, "matched_keywords": []}
    
    text_lower = text.lower()
    matched = []
    for kw in NARRATIVE_LEAKAGE_KEYWORDS:
        # Use word-boundary regex to prevent partial word matches
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text_lower):
            matched.append(kw)
            
    return {
        "has_leakage": len(matched) > 0,
        "matched_keywords": matched
    }


def sanitize_narrative_leakage(text: str) -> str:
    """
    Optional experimental sanitizer:
    Removes obvious meta-commentary indicator clauses and analytical cue words
    for fair model experimentation without narrative shortcutting.
    
    This does NOT modify the original CSV file.
    
    Args:
        text: Input conversation text.
        
    Returns:
        Sanitized conversation text.
    """
    if not text:
        return ""
    
    sanitized = text
    # Remove prominent meta-commentary clauses
    for pattern in COMMENTARY_CLAUSE_PATTERNS:
        sanitized = pattern.sub('', sanitized)
    
    # Normalize resulting whitespace
    sanitized = re.sub(r'\s{2,}', ' ', sanitized)
    sanitized = re.sub(r'\s+([.,!?])', r'\1', sanitized)
    return sanitized.strip()


def extract_lexical_features(df: pd.DataFrame, text_col: str = PRIMARY_TEXT_FEATURE) -> pd.DataFrame:
    """
    Extract engineered lexical features from conversation text.
    
    These features represent interpretable statistical and behavioral indicators
    of social engineering and conversational liveness.
    
    Features generated:
    - text_length: Character count
    - word_count: Number of whitespace-separated tokens
    - exclamation_count: Count of '!'
    - question_count: Count of '?'
    - uppercase_ratio: Proportion of uppercase alphabetic characters
    - digit_count: Count of numeric digits
    - url_present: 1 if HTTP/HTTPS/WWW pattern is present, 0 otherwise
    - urgency_keyword_present: 1 if urgency words (urgent, immediately, asap, etc.) present
    - sensitive_request_keyword_present: 1 if credential/financial words (otp, pin, password, etc.) present
    
    Args:
        df: DataFrame containing the text column.
        text_col: Name of the conversation text column.
        
    Returns:
        DataFrame with the engineered lexical feature columns.
    """
    # Regex patterns for social engineering indicators
    url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    urgency_pattern = re.compile(
        r'\b(urgent|urgently|immediate|immediately|asap|now|fast|hurry|expire|deadline|within\s+\d+\s+hours?)\b',
        re.IGNORECASE
    )
    sensitive_pattern = re.compile(
        r'\b(password|passwords|otp|pin|credentials?|credit\s*card|cvv|account\s*details?|wallet\s*pin|transfer\s*money|release\s*fee|bank\s*details?)\b',
        re.IGNORECASE
    )
    
    features = pd.DataFrame(index=df.index)
    series = df[text_col].fillna("").astype(str)
    
    features["text_length"] = series.str.len()
    features["word_count"] = series.apply(lambda s: len(s.split()))
    features["exclamation_count"] = series.apply(lambda s: s.count('!'))
    features["question_count"] = series.apply(lambda s: s.count('?'))
    
    def calc_upper_ratio(s: str) -> float:
        alphas = [c for c in s if c.isalpha()]
        if not alphas:
            return 0.0
        return sum(1 for c in alphas if c.isupper()) / len(alphas)
    
    features["uppercase_ratio"] = series.apply(calc_upper_ratio).round(4)
    features["digit_count"] = series.apply(lambda s: sum(1 for c in s if c.isdigit()))
    features["url_present"] = series.apply(lambda s: 1 if url_pattern.search(s) else 0)
    features["urgency_keyword_present"] = series.apply(lambda s: 1 if urgency_pattern.search(s) else 0)
    features["sensitive_request_keyword_present"] = series.apply(lambda s: 1 if sensitive_pattern.search(s) else 0)
    
    return features


# =====================================================================
# DATASET SPLITTING (STRATIFIED, LEAKAGE-FREE)
# =====================================================================

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into reproducible, stratified train/validation/test partitions.
    
    Stratification is performed on the primary target: 'Conversational Authenticity'.
    
    Requirements:
    - Train: 70% (14,000 rows for 20,000 dataset)
    - Val: 15% (3,000 rows for 20,000 dataset)
    - Test: 15% (3,000 rows for 20,000 dataset)
    - Stratified by Conversational Authenticity
    - Fixed random seed
    - No data leakage between splits
    
    Args:
        df: DataFrame to split.
        train_ratio: Proportion for training set (default 0.70).
        val_ratio: Proportion for validation set (default 0.15).
        test_ratio: Proportion for test set (default 0.15).
        random_seed: Random state for deterministic reproduction.
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0; got {total_ratio}")
        
    if PRIMARY_TARGET not in df.columns:
        raise ValueError(f"Primary target '{PRIMARY_TARGET}' required for stratified split.")
    
    # Step 1: Split into Train (70%) and Temp (30%)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=df[PRIMARY_TARGET]
    )
    
    # Step 2: Split Temp into Val (15%) and Test (15%)
    # val proportion of temp: val_ratio / (val_ratio + test_ratio) = 0.15 / 0.30 = 0.50
    relative_val_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val_ratio),
        random_state=random_seed,
        stratify=temp_df[PRIMARY_TARGET]
    )
    
    # Reset indices cleanly
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, val_df, test_df


def separate_features_and_targets(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Strictly separate valid input features from target columns and leakage columns.
    
    Returns:
        Dictionary containing:
        - 'text_feature': Series of cleaned conversation text
        - 'primary_target': Series of Conversational Authenticity
        - 'secondary_target': Series of Threat Severity
        - 'metadata': DataFrame of non-leaky tabular context (for analysis only)
        - 'excluded_columns': List of columns strictly banned from ML feature input
    """
    return {
        "text_feature": df[PRIMARY_TEXT_FEATURE],
        "primary_target": df[PRIMARY_TARGET],
        "secondary_target": df[SECONDARY_TARGET],
        "metadata": df[[c for c in TABULAR_METADATA_COLUMNS if c in df.columns]],
        "excluded_columns": EXCLUDED_LEAKAGE_COLUMNS
    }
