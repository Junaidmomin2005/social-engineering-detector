"""
API Tests for FastAPI application:
1. Verify GET / serves frontend index.html successfully (200 OK).
2. Verify POST /analyze processes valid conversation text and returns required schema.
3. Verify POST /analyze rejects empty or whitespace-only text with 400 Bad Request.
4. Verify response fields: classification, risk_score, risk_level, detected_requests, manipulation_indicators, recommendation.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app

client = TestClient(app)


def test_get_root_serves_frontend():
    """Test that GET / returns the frontend HTML page with status code 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Conversational Liveness & Social Engineering Detector" in response.text
    print("PASS: test_get_root_serves_frontend")


def test_analyze_valid_text():
    """Test that POST /analyze accepts valid text and returns all required fields."""
    payload = {
        "conversation_text": (
            "URGENT: This is your bank. Your account has been temporarily suspended. "
            "Please send your OTP immediately to verify your account."
        )
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Check required fields
    required_keys = [
        "classification",
        "risk_score",
        "risk_level",
        "detected_requests",
        "manipulation_indicators",
        "recommendation",
    ]
    for key in required_keys:
        assert key in data, f"Missing required key in response: {key}"

    assert data["classification"] in ["Real", "Suspicious", "Fake"]
    assert isinstance(data["risk_score"], int)
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert isinstance(data["detected_requests"], list)
    assert isinstance(data["manipulation_indicators"], list)
    assert isinstance(data["recommendation"], str) and len(data["recommendation"]) > 0

    # Rule-based detector checks
    assert "OTP" in data["detected_requests"]
    assert "Urgency" in data["manipulation_indicators"]

    print("PASS: test_analyze_valid_text")
    print(f"Sample response: {data}")


def test_analyze_rejects_empty_text():
    """Test that POST /analyze rejects empty or whitespace-only text with 400."""
    # Test completely empty text
    res_empty = client.post("/analyze", json={"conversation_text": ""})
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()

    # Test whitespace text
    res_whitespace = client.post("/analyze", json={"conversation_text": "   \n\t  "})
    assert res_whitespace.status_code == 400
    assert "empty" in res_whitespace.json()["detail"].lower()

    print("PASS: test_analyze_rejects_empty_text")


def test_analyze_rejects_missing_field():
    """Test that POST /analyze validates payload schema."""
    response = client.post("/analyze", json={})
    assert response.status_code == 422
    print("PASS: test_analyze_rejects_missing_field")


if __name__ == "__main__":
    test_get_root_serves_frontend()
    test_analyze_valid_text()
    test_analyze_rejects_empty_text()
    test_analyze_rejects_missing_field()
    print("\nAll API tests passed successfully!")
