"""
Dataset Audit & Preprocessing Script:
Loads the raw dataset, conducts comprehensive schema and leakage audits,
cleans text, extracts lexical features, partitions stratified train/val/test splits,
and persists training-ready datasets to data/processed/.

Ensures zero modification to data/deepfake_conversation_factor_analysis.csv.
"""

import os
import sys
import json
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ml.data_utils import (
    EXPECTED_COLUMNS,
    PRIMARY_TARGET,
    SECONDARY_TARGET,
    PRIMARY_TEXT_FEATURE,
    EXCLUDED_LEAKAGE_COLUMNS,
    TABULAR_METADATA_COLUMNS,
    compute_file_sha256,
    load_raw_dataset,
    validate_dataset_schema,
    clean_conversation_text,
    detect_narrative_leakage,
    sanitize_narrative_leakage,
    extract_lexical_features,
    split_dataset
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("preprocess")


def run_dataset_audit_and_preprocessing(
    raw_csv_path: str = os.path.join(PROJECT_ROOT, "data", "deepfake_conversation_factor_analysis.csv"),
    processed_dir: str = os.path.join(PROJECT_ROOT, "data", "processed")
) -> Dict[str, Any]:
    """
    Execute full dataset audit, validation, feature preparation, and stratified splitting.
    
    Args:
        raw_csv_path: Absolute or relative path to the raw dataset CSV.
        processed_dir: Directory where processed CSV splits will be saved.
        
    Returns:
        Dictionary containing comprehensive audit results and summary statistics.
    """
    logger.info("=" * 70)
    logger.info("CONVERSATIONAL LIVENESS & SOCIAL ENGINEERING DETECTOR")
    logger.info("Step 2: Dataset Preprocessing & Data-Leakage Audit Pipeline")
    logger.info("=" * 70)
    
    # -----------------------------------------------------------------
    # 1. VERIFY SOURCE CSV EXISTENCE & RECORD INITIAL SHA256 HASH
    # -----------------------------------------------------------------
    logger.info(f"Checking raw dataset at: {raw_csv_path}")
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Source dataset does not exist: {raw_csv_path}")
        
    initial_sha256 = compute_file_sha256(raw_csv_path)
    initial_mtime = os.path.getmtime(raw_csv_path)
    logger.info(f"Original CSV SHA256: {initial_sha256}")
    logger.info(f"Original CSV Modification Time: {initial_mtime}")
    
    # -----------------------------------------------------------------
    # 2. LOAD RAW DATASET & VALIDATE SCHEMA
    # -----------------------------------------------------------------
    df_raw = load_raw_dataset(raw_csv_path)
    audit_report = validate_dataset_schema(df_raw)
    
    logger.info(f"Total records loaded: {audit_report['total_rows']}")
    logger.info(f"Total columns loaded: {audit_report['total_columns']}")
    logger.info(f"Duplicate rows detected: {audit_report['duplicate_rows']}")
    logger.info(f"Empty conversation_text count: {audit_report['empty_text_count']}")
    
    # Print Missing Values Breakdown
    logger.info("\n--- Missing Values Breakdown ---")
    for col, count in audit_report["missing_values"].items():
        pct = (count / audit_report["total_rows"]) * 100
        logger.info(f"  - {col:<30}: {count:>6} missing ({pct:>5.1f}%)")
        
    # Print Target Distributions
    logger.info("\n--- Target Distributions ---")
    logger.info(f"Primary Target: '{PRIMARY_TARGET}'")
    for cls_name, count in audit_report["primary_target_distribution"].items():
        pct = (count / audit_report["total_rows"]) * 100
        logger.info(f"  - {cls_name:<15}: {count:>6} ({pct:>5.1f}%)")
        
    logger.info(f"Secondary Target: '{SECONDARY_TARGET}'")
    for cls_name, count in audit_report["secondary_target_distribution"].items():
        pct = (count / audit_report["total_rows"]) * 100
        logger.info(f"  - {cls_name:<15}: {count:>6} ({pct:>5.1f}%)")
        
    # -----------------------------------------------------------------
    # 3. DETECT POTENTIAL NARRATIVE / META-COMMENTARY LEAKAGE IN TEXT
    # -----------------------------------------------------------------
    logger.info("\n--- Narrative / Meta-Commentary Leakage Audit ---")
    leakage_detections = [detect_narrative_leakage(t) for t in df_raw[PRIMARY_TEXT_FEATURE]]
    leakage_flags = [d["has_leakage"] for d in leakage_detections]
    total_leakage_flagged = sum(leakage_flags)
    pct_leakage = (total_leakage_flagged / len(df_raw)) * 100
    
    keyword_freq: Dict[str, int] = {}
    for d in leakage_detections:
        for kw in d["matched_keywords"]:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
            
    logger.info(f"Rows flagged with potential meta-commentary cues: {total_leakage_flagged} ({pct_leakage:.2f}%)")
    logger.info("Keyword frequency in conversation_text:")
    for kw, cnt in sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - '{kw}': {cnt} occurrences")
        
    audit_report["narrative_leakage"] = {
        "flagged_rows_count": total_leakage_flagged,
        "flagged_rows_percentage": round(pct_leakage, 2),
        "keyword_frequencies": keyword_freq
    }
    
    # -----------------------------------------------------------------
    # 4. PREPARE CLEANED ML DATAFRAME & ENGINEERED LEXICAL FEATURES
    # -----------------------------------------------------------------
    logger.info("\n--- Applying Non-Destructive Text Preprocessing & Feature Extraction ---")
    df_clean = df_raw.copy()
    
    # Non-destructive text cleaning
    df_clean["clean_conversation_text"] = df_clean[PRIMARY_TEXT_FEATURE].apply(clean_conversation_text)
    
    # Optional experimental sanitized text (for ablation studies without analytical cues)
    df_clean["sanitized_conversation_text"] = df_clean["clean_conversation_text"].apply(sanitize_narrative_leakage)
    
    # Narrative leakage flag
    df_clean["has_narrative_leakage"] = leakage_flags
    
    # Extract interpretable lexical features
    df_lexical = extract_lexical_features(df_clean, text_col="clean_conversation_text")
    logger.info(f"Engineered {df_lexical.shape[1]} lexical features:")
    for feat in df_lexical.columns:
        logger.info(f"  - {feat}")
        
    # Combine cleaned data with lexical features
    df_ml = pd.concat([df_clean, df_lexical], axis=1)
    
    # -----------------------------------------------------------------
    # 5. STRICT FEATURE & LEAKAGE SEPARATION
    # -----------------------------------------------------------------
    logger.info("\n--- Critical Data-Leakage Column Isolation ---")
    logger.info("The following columns are strictly EXCLUDED from ML features:")
    leakage_reasons = {
        "conversation_id": "Arbitrary record identifier; lacks semantic generalization.",
        "suggested_action": "Deterministic 1-to-1 bijection with Threat Severity.",
        "Conversation Type": "Deterministic mapping to Conversational Authenticity.",
        "Manipulation Type": "Missingness directly reveals Real records (100% missing in Real).",
        "Malicious": "Outcome label; missing for 100% of Real records.",
        "Impersonation": "Outcome label; missing for 100% of Real records.",
        "Conversational Authenticity": "Primary target variable.",
        "Threat Severity": "Secondary target variable."
    }
    for col, reason in leakage_reasons.items():
        logger.info(f"  [EXCLUDED] {col:<30}: {reason}")
        
    # Define valid columns to persist in processed training sets:
    # 1. Identifier for traceability: conversation_id
    # 2. Text columns: clean_conversation_text, sanitized_conversation_text, conversation_text
    # 3. Targets: Conversational Authenticity, Threat Severity
    # 4. Lexical features: 9 columns
    # 5. Tabular metadata: for academic/exploratory analysis
    # 6. Narrative leakage flag: has_narrative_leakage
    # NOTICE: The 5 deterministic leakage columns (suggested_action, Conversation Type,
    # Manipulation Type, Malicious, Impersonation) are EXCLUDED from processed ML splits.
    
    processed_columns = [
        "conversation_id",
        PRIMARY_TARGET,
        SECONDARY_TARGET,
        "clean_conversation_text",
        "sanitized_conversation_text",
        "has_narrative_leakage",
        "text_length",
        "word_count",
        "exclamation_count",
        "question_count",
        "uppercase_ratio",
        "digit_count",
        "url_present",
        "urgency_keyword_present",
        "sensitive_request_keyword_present"
    ] + [c for c in TABULAR_METADATA_COLUMNS if c in df_ml.columns]
    
    df_processed = df_ml[processed_columns].copy()
    
    # -----------------------------------------------------------------
    # 6. REPRODUCIBLE STRATIFIED TRAIN / VAL / TEST SPLIT
    # -----------------------------------------------------------------
    logger.info("\n--- Stratified Train/Val/Test Dataset Splitting ---")
    logger.info("Split configuration: Train=70%, Validation=15%, Test=15%, Seed=42, Stratify='Conversational Authenticity'")
    
    train_df, val_df, test_df = split_dataset(
        df_processed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    logger.info(f"Train split size: {len(train_df):>6} rows ({(len(train_df)/len(df_processed))*100:.1f}%)")
    logger.info(f"Val split size:   {len(val_df):>6} rows ({(len(val_df)/len(df_processed))*100:.1f}%)")
    logger.info(f"Test split size:  {len(test_df):>6} rows ({(len(test_df)/len(df_processed))*100:.1f}%)")
    
    # Verify split distributions
    split_stats = {}
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split[PRIMARY_TARGET].value_counts().to_dict()
        dist_pct = {k: round((v / len(split)) * 100, 2) for k, v in dist.items()}
        split_stats[name] = {"counts": dist, "percentages": dist_pct}
        logger.info(f"{name:<5} Target Distribution: {dist_pct}")
        
    audit_report["split_statistics"] = split_stats
    audit_report["split_row_counts"] = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df)
    }
    
    # -----------------------------------------------------------------
    # 7. EXPORT PROCESSED DATASETS
    # -----------------------------------------------------------------
    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    summary_path = os.path.join(processed_dir, "dataset_summary.json")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # -----------------------------------------------------------------
    # 8. VERIFY ORIGINAL CSV REMAINS UNMODIFIED
    # -----------------------------------------------------------------
    final_sha256 = compute_file_sha256(raw_csv_path)
    final_mtime = os.path.getmtime(raw_csv_path)
    is_unchanged = (initial_sha256 == final_sha256) and (initial_mtime == final_mtime)
    
    if not is_unchanged:
        raise RuntimeError("CRITICAL ERROR: Original raw dataset was modified during preprocessing!")
        
    logger.info(f"\nSource CSV immutability check PASSED: SHA256 verified unchanged ({final_sha256})")
    
    # -----------------------------------------------------------------
    # 9. SAVE METADATA SUMMARY JSON
    # -----------------------------------------------------------------
    summary_data = {
        "source_dataset_sha256": final_sha256,
        "source_dataset_rows": len(df_raw),
        "source_dataset_columns": len(df_raw.columns),
        "primary_target": PRIMARY_TARGET,
        "secondary_target": SECONDARY_TARGET,
        "primary_feature": PRIMARY_TEXT_FEATURE,
        "lexical_features": list(df_lexical.columns),
        "excluded_leakage_columns": EXCLUDED_LEAKAGE_COLUMNS,
        "tabular_metadata_columns": TABULAR_METADATA_COLUMNS,
        "splits": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "random_seed": 42,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "stratified_by": PRIMARY_TARGET
        },
        "source_unmodified": True
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
        
    logger.info(f"Processed splits saved successfully to: {processed_dir}")
    logger.info(f"  - {train_path} ({os.path.getsize(train_path)} bytes)")
    logger.info(f"  - {val_path} ({os.path.getsize(val_path)} bytes)")
    logger.info(f"  - {test_path} ({os.path.getsize(test_path)} bytes)")
    logger.info(f"  - {summary_path}")
    logger.info("=" * 70)
    logger.info("Preprocessing & Validation completed successfully!")
    logger.info("=" * 70)
    
    return audit_report


if __name__ == "__main__":
    run_dataset_audit_and_preprocessing()
