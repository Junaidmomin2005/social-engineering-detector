"""
Unit and integration tests for Data Preprocessing and Validation Pipeline.
Supports both unittest (standard library) and pytest.
"""

import os
import unittest
import pandas as pd

from backend.ml.data_utils import (
    clean_conversation_text,
    extract_lexical_features,
    detect_narrative_leakage,
    sanitize_narrative_leakage,
    split_dataset,
    compute_file_sha256,
    PRIMARY_TARGET,
    SECONDARY_TARGET
)


class TestPreprocessingPipeline(unittest.TestCase):

    def test_clean_conversation_text_null_and_empty(self):
        self.assertEqual(clean_conversation_text(None), "")
        self.assertEqual(clean_conversation_text(""), "")
        self.assertEqual(clean_conversation_text("   \n\t  "), "")

    def test_clean_conversation_text_normalization(self):
        raw = "Hello   world!\nThis is a\ttest with $500 and OTP 123456."
        cleaned = clean_conversation_text(raw)
        self.assertEqual(cleaned, "Hello world! This is a test with $500 and OTP 123456.")
        self.assertIn("$500", cleaned)
        self.assertIn("OTP 123456", cleaned)

    def test_extract_lexical_features(self):
        df = pd.DataFrame({
            "conversation_text": [
                "URGENT: Please verify your password at http://example.com immediately! Are you there?",
                "Normal conversation about dinner tonight."
            ]
        })
        lexical = extract_lexical_features(df, "conversation_text")
        self.assertEqual(len(lexical), 2)
        self.assertGreaterEqual(lexical.loc[0, "exclamation_count"], 1)
        self.assertGreaterEqual(lexical.loc[0, "question_count"], 1)
        self.assertEqual(lexical.loc[0, "url_present"], 1)
        self.assertEqual(lexical.loc[0, "urgency_keyword_present"], 1)
        self.assertEqual(lexical.loc[0, "sensitive_request_keyword_present"], 1)
        self.assertGreater(lexical.loc[0, "uppercase_ratio"], 0.0)

        self.assertEqual(lexical.loc[1, "url_present"], 0)
        self.assertEqual(lexical.loc[1, "urgency_keyword_present"], 0)
        self.assertEqual(lexical.loc[1, "sensitive_request_keyword_present"], 0)

    def test_detect_narrative_leakage(self):
        leakage_text = "The exchange becomes suspicious when the sender asks for card details."
        res = detect_narrative_leakage(leakage_text)
        self.assertTrue(res["has_leakage"])
        self.assertIn("suspicious", res["matched_keywords"])

        clean_text = "Can you send the project meeting notes from yesterday?"
        res_clean = detect_narrative_leakage(clean_text)
        self.assertFalse(res_clean["has_leakage"])
        self.assertEqual(len(res_clean["matched_keywords"]), 0)

    def test_sanitize_narrative_leakage(self):
        leakage_text = "The exchange becomes suspicious when it connects medical lab report with a sudden warning."
        sanitized = sanitize_narrative_leakage(leakage_text)
        self.assertNotIn("The exchange becomes suspicious when", sanitized)

    def test_split_dataset_stratification(self):
        # Using 1,000 samples so 70%/15%/15% (700, 150, 150) yields exact integer class distributions
        df = pd.DataFrame({
            "conversation_text": [f"text {i}" for i in range(1000)],
            PRIMARY_TARGET: ["Fake"] * 600 + ["Real"] * 300 + ["Suspicious"] * 100
        })
        train, val, test = split_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

        self.assertEqual(len(train), 700)
        self.assertEqual(len(val), 150)
        self.assertEqual(len(test), 150)

        for split in [train, val, test]:
            counts = split[PRIMARY_TARGET].value_counts().to_dict()
            self.assertEqual(counts["Fake"], int(len(split) * 0.60))
            self.assertEqual(counts["Real"], int(len(split) * 0.30))
            self.assertEqual(counts["Suspicious"], int(len(split) * 0.10))

    def test_original_dataset_integrity(self):
        raw_path = os.path.join("data", "deepfake_conversation_factor_analysis.csv")
        expected_sha256 = "ec96f27238a56406482fc5257e1d273b607577fff177e587e51d0da0af217344"
        actual_sha256 = compute_file_sha256(raw_path)
        self.assertEqual(actual_sha256, expected_sha256, "Original raw CSV was altered!")

    def test_processed_splits_exist_and_exclude_leakage(self):
        for split_name in ["train", "val", "test"]:
            split_path = os.path.join("data", "processed", f"{split_name}.csv")
            self.assertTrue(os.path.exists(split_path), f"Processed split {split_name}.csv does not exist")

            df = pd.read_csv(split_path)
            # Check that deterministic leakage columns are excluded from processed splits
            banned = ["suggested_action", "Conversation Type", "Manipulation Type", "Malicious", "Impersonation"]
            for b in banned:
                self.assertNotIn(b, df.columns, f"Leaky column '{b}' found in {split_name}.csv!")

            # Check target columns exist
            self.assertIn(PRIMARY_TARGET, df.columns)
            self.assertIn(SECONDARY_TARGET, df.columns)
            self.assertIn("clean_conversation_text", df.columns)
            self.assertIn("text_length", df.columns)


if __name__ == "__main__":
    unittest.main()
