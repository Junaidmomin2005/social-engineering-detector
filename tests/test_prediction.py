"""
Comprehensive Unit Tests for Prediction Service:
Tests diverse categories of custom input to verify that:
1. Normal conversations stay LOW risk.
2. Security-awareness statements stay LOW risk.
3. Suspicious requests produce elevated risk (MEDIUM/HIGH).
4. Strong social engineering combinations reach HIGH or CRITICAL.
5. Synonymous phrasings generalize appropriately.
6. Schema and bounds are strictly enforced.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.predict import analyze_conversation, load_artifacts


def run_comprehensive_tests():
    print("==================================================================")
    print("Running Comprehensive Evaluation of Prediction & Risk Service")
    print("==================================================================")

    model, vectorizer = load_artifacts()
    print(f"Loaded model: {type(model).__name__}")
    print(f"Loaded vectorizer: {type(vectorizer).__name__}")
    print("Model loaded from disk, NOT retrained.\n")

    test_suite = [
        # Category A: Normal conversations
        {
            "category": "A - Normal Conversation",
            "text": "How are you?",
            "expected_level": "LOW",
        },
        {
            "category": "A - Normal Conversation",
            "text": "What time is the meeting?",
            "expected_level": "LOW",
        },
        {
            "category": "A - Normal Conversation",
            "text": "What is your name?",
            "expected_level": "LOW",
        },
        {
            "category": "A - Normal Conversation",
            "text": "Can you send me the project report?",
            "expected_level": "LOW",
        },
        {
            "category": "A - Normal Conversation",
            "text": "Can you help me with Java?",
            "expected_level": "LOW",
        },
        {
            "category": "A - Normal Conversation",
            "text": "I will reach the office at 10.",
            "expected_level": "LOW",
        },
        {
            "category": "A - Normal Conversation",
            "text": "Where are you going?",
            "expected_level": "LOW",
        },
        {
            "category": "A - Normal Conversation",
            "text": "Can you send the assignment?",
            "expected_level": "LOW",
        },

        # Category B: Security-awareness statements
        {
            "category": "B - Security Awareness",
            "text": "Never share your OTP with anyone.",
            "expected_level": "LOW",
        },
        {
            "category": "B - Security Awareness",
            "text": "Your password should always remain private.",
            "expected_level": "LOW",
        },
        {
            "category": "B - Security Awareness",
            "text": "Do not click suspicious links.",
            "expected_level": "LOW",
        },
        {
            "category": "B - Security Awareness",
            "text": "Remember to never give your credentials to unknown callers.",
            "expected_level": "LOW",
        },
        {
            "category": "B - Security Awareness",
            "text": "Beware of fake messages asking for your bank account details.",
            "expected_level": "LOW",
        },

        # Category C: Suspicious requests
        {
            "category": "C - Suspicious Request",
            "text": "Please click this link to verify your account.",
            "expected_level": ["MEDIUM", "HIGH"],
            "must_detect_request": "Link/URL",
        },
        {
            "category": "C - Suspicious Request",
            "text": "Send me your login credentials.",
            "expected_level": ["MEDIUM", "HIGH"],
            "must_detect_request": "Login/Credentials",
        },
        {
            "category": "C - Suspicious Request",
            "text": "Please provide your personal details for verification.",
            "expected_level": ["MEDIUM", "HIGH"],
            "must_detect_request": "Personal Information",
        },
        {
            "category": "C - Suspicious Request",
            "text": "Download and install this remote access software for support.",
            "expected_level": ["MEDIUM", "HIGH"],
            "must_detect_request": "Software Installation",
        },
        {
            "category": "C - Suspicious Request",
            "text": "Please transfer 250 dollars to confirm your booking.",
            "expected_level": ["MEDIUM", "HIGH"],
            "must_detect_request": "Money/Payment",
        },
        {
            "category": "C - Suspicious Request",
            "text": "Verify your identity by providing your card details.",
            "expected_level": ["MEDIUM", "HIGH"],
            "must_detect_request": "Bank/Card Details",
        },

        # Category D: Strong social engineering
        {
            "category": "D - Strong Social Engineering",
            "text": "Your account will be blocked unless you send the OTP immediately.",
            "expected_level": ["HIGH", "CRITICAL"],
            "must_detect_request": "OTP",
            "must_detect_manip": "Threat/Fear",
        },
        {
            "category": "D - Strong Social Engineering",
            "text": "This is the bank security team. Send your OTP right now.",
            "expected_level": ["HIGH", "CRITICAL"],
            "must_detect_request": "OTP",
            "must_detect_manip": "Authority/Impersonation",
        },
        {
            "category": "D - Strong Social Engineering",
            "text": "Your payment failed. Click this link immediately and enter your card details.",
            "expected_level": ["HIGH", "CRITICAL"],
            "must_detect_request": "Bank/Card Details",
            "must_detect_manip": "Urgency",
        },
        {
            "category": "D - Strong Social Engineering",
            "text": "URGENT: This is IT support. Your account is suspended. Send your password right now.",
            "expected_level": ["HIGH", "CRITICAL"],
            "must_detect_request": "Password",
            "must_detect_manip": "Authority/Impersonation",
        },
        {
            "category": "D - Strong Social Engineering",
            "text": "This is police fraud department. You must transfer funds immediately or legal action will be taken.",
            "expected_level": ["HIGH", "CRITICAL"],
            "must_detect_request": "Money/Payment",
            "must_detect_manip": "Threat/Fear",
        },

        # Category E: Different wording for same underlying request (OTP/Verification Code)
        {
            "category": "E - Wording Variation (OTP)",
            "text": "Send me your OTP.",
            "must_detect_request": "OTP",
        },
        {
            "category": "E - Wording Variation (OTP)",
            "text": "Please provide the verification code.",
            "must_detect_request": "OTP",
        },
        {
            "category": "E - Wording Variation (OTP)",
            "text": "Tell me the one-time code you received.",
            "must_detect_request": "OTP",
        },
        {
            "category": "E - Wording Variation (OTP)",
            "text": "Share the code sent to your phone.",
            "must_detect_request": "OTP",
        },
        {
            "category": "E - Wording Variation (OTP)",
            "text": "What is the one-time password on your screen?",
            "must_detect_request": "OTP",
        },
        {
            "category": "E - Wording Variation (OTP)",
            "text": "Give me the 2FA code you just got.",
            "must_detect_request": "OTP",
        },

        # Category F: Completely custom / arbitrary sentences
        {
            "category": "F - Arbitrary Custom Message",
            "text": "The delivery package was delayed due to heavy rain. Will check tracking tomorrow.",
            "expected_level": "LOW",
        },
        {
            "category": "F - Arbitrary Custom Message",
            "text": "Can we reschedule our sync to 3 PM? I have an unexpected customer call.",
            "expected_level": "LOW",
        },
        {
            "category": "F - Arbitrary Custom Message",
            "text": "The recipe calls for two tablespoons of olive oil and minced garlic.",
            "expected_level": "LOW",
        },
        {
            "category": "F - Arbitrary Custom Message",
            "text": "Please review the pull request on GitHub when you have free time.",
            "expected_level": "LOW",
        },
        {
            "category": "F - Arbitrary Custom Message",
            "text": "Let's organize a team lunch this Friday to celebrate the milestone.",
            "expected_level": "LOW",
        },
    ]

    passed_count = 0
    total_count = len(test_suite)

    for i, tc in enumerate(test_suite, 1):
        text = tc["text"]
        result = analyze_conversation(text)

        # 1. Structural assertions
        assert "classification" in result
        assert "confidence" in result
        assert "risk_score" in result
        assert "risk_level" in result
        assert "detected_requests" in result
        assert "manipulation_indicators" in result
        assert "recommendation" in result

        # 2. Type & range bounds
        assert result["classification"] in ["Real", "Suspicious", "Fake"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0 <= result["risk_score"] <= 100
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert isinstance(result["detected_requests"], list)
        assert isinstance(result["manipulation_indicators"], list)
        assert isinstance(result["recommendation"], str) and len(result["recommendation"]) > 0

        # 3. Expected level checks if specified
        if "expected_level" in tc:
            expected = tc["expected_level"]
            if isinstance(expected, list):
                assert result["risk_level"] in expected, (
                    f"Test #{i} [{tc['category']}] failed: Text='{text}' got risk_level='{result['risk_level']}', "
                    f"expected one of {expected} (score={result['risk_score']})"
                )
            else:
                assert result["risk_level"] == expected, (
                    f"Test #{i} [{tc['category']}] failed: Text='{text}' got risk_level='{result['risk_level']}', "
                    f"expected {expected} (score={result['risk_score']})"
                )

        # 4. Request detection checks if specified
        if "must_detect_request" in tc:
            req = tc["must_detect_request"]
            assert req in result["detected_requests"], (
                f"Test #{i} [{tc['category']}] failed: Expected request '{req}' not found in {result['detected_requests']}"
            )

        # 5. Manipulation detection checks if specified
        if "must_detect_manip" in tc:
            manip = tc["must_detect_manip"]
            assert manip in result["manipulation_indicators"], (
                f"Test #{i} [{tc['category']}] failed: Expected manipulation '{manip}' not found in {result['manipulation_indicators']}"
            )

        passed_count += 1
        print(f"[{i:02d}/{total_count}] PASS ({tc['category']}): \"{text[:45]}...\" -> {result['risk_level']} (Score: {result['risk_score']})")

    print(f"\nSUCCESS: All {passed_count}/{total_count} diverse test cases passed successfully!")


if __name__ == "__main__":
    run_comprehensive_tests()
