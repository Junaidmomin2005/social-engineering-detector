"""
Prediction Service Module:
Provides an explainable, content-driven NLP conversational risk assessment system.

Architecture:
1. Machine Learning Classification: LogisticRegression + TF-IDF (loaded from saved artifacts).
   - Reports ML classification ('Real', 'Suspicious', 'Fake') and prediction confidence.
   - Operates as secondary supporting evidence; does NOT unilaterally dictate risk.
2. Context-Aware Content Analysis:
   - Evaluates active sensitive requests (OTP, Credentials, Banking, Personal Info, Links, etc.).
   - Filters out defensive/educational security-awareness statements (e.g., 'Never share your password').
   - Detects psychological manipulation indicators (Urgency, Threat/Fear, Authority, Secrecy).
3. Evidence-Based Risk Scoring:
   - Derives a 0-100 Risk Score strictly from the detected evidence and combined indicators.
   - Maps to Risk Level (LOW, MEDIUM, HIGH, CRITICAL).
   - Generates actionable security recommendations.
"""

from pathlib import Path
import re
from typing import Dict, List, Any, Tuple
import joblib


# Path definitions
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "backend" / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"

# Global lazy-loaded artifact caches
_MODEL = None
_VECTORIZER = None


def load_artifacts():
    """
    Loads serialized model and vectorizer from backend/ml/artifacts/.
    Ensures model is loaded from disk and never retrained at inference time.
    """
    global _MODEL, _VECTORIZER

    if _MODEL is None or _VECTORIZER is None:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"Required artifacts missing in {ARTIFACTS_DIR}. "
                "Ensure train.py has been executed successfully."
            )
        _VECTORIZER = joblib.load(VECTORIZER_PATH)
        _MODEL = joblib.load(MODEL_PATH)

    return _MODEL, _VECTORIZER


# ---------------------------------------------------------------------
# DEFENSIVE / EDUCATIONAL STATEMENT FILTER
# ---------------------------------------------------------------------
DEFENSIVE_PATTERNS = [
    r"\b(?:never|do\s+not|don'?t|should\s+never)\s+(?:share|give|send|provide|disclose|reveal|enter|click|transfer)\b",
    r"\b(?:should\s+always\s+remain\s+(?:private|confidential|secret)|keep\s+(?:your\s+)?\w+\s+(?:private|safe|secret|confidential))\b",
    r"\b(?:beware\s+of|watch\s+out\s+for|be\s+careful\s+(?:of|with|about))\b",
    r"\b(?:security\s+(?:tip|reminder|awareness|advisory)|fraud\s+warning|warning:\s*never)\b",
    r"\b(?:protect\s+your\s+(?:password|pin|account|credentials))\b",
    r"\b(?:remember\s+to\s+never|always\s+verify|no\s+one\s+will\s+ask\s+for\s+your)\b",
    r"\b(?:we\s+will\s+never\s+ask|banks?\s+never\s+ask|never\s+ask\s+for\s+your)\b",
    r"\b(?:scam\s+alert|phishing\s+awareness|educational\s+purpose)\b",
]


def is_sentence_defensive(sentence: str) -> bool:
    """Checks whether a sentence/clause is an educational or defensive warning."""
    s_lower = sentence.lower()
    for pattern in DEFENSIVE_PATTERNS:
        if re.search(pattern, s_lower):
            return True
    return False


# ---------------------------------------------------------------------
# SENSITIVE REQUEST PATTERNS (ACTIVE EXTRACTION)
# ---------------------------------------------------------------------
REQUEST_PATTERNS = {
    "OTP": [
        r"\botp\b",
        r"\bone[-\s]?time\s+(?:password|code|pin|token)\b",
        r"\bverification\s+code\b",
        r"\bsecurity\s+code\b",
        r"\b(?:2fa|mfa)\s+code\b",
        r"\bauth(?:entication)?\s+code\b",
        r"\bcode\s+(?:sent|texted|emailed)\s+to\s+your\b",
        r"\b(?:code|pin)\s+(?:you\s+)?received\b",
        r"\b(?:share|send|provide|give|tell)\s+(?:me\s+)?(?:the\s+|your\s+)?(?:one[-\s]?time\s+)?(?:verification\s+)?(?:code|pin|otp)\b",
        r"\bshare\s+the\s+code\b",
    ],
    "Password": [
        r"\bpassword\b",
        r"\bpasscode\b",
        r"\bsecret\s+pin\b",
        r"\batm\s+pin\b",
        r"\bwallet\s+pin\b",
        r"\bmpin\b",
        r"\bsecurity\s+pin\b",
        r"\bmaster\s+key\b",
        r"\bseed\s+phrase\b",
        r"\brecovery\s+phrase\b",
    ],
    "Bank/Card Details": [
        r"\bbank\s+account(?:\s+details|\s+number)?\b",
        r"\bcard\s+(?:number|details|info|credentials)\b",
        r"\b(?:credit|debit)\s+card(?:\s+(?:number|details|info))?\b",
        r"\bcvv2?\b",
        r"\bcvc2?\b",
        r"\bexpiry\s+(?:date|month|year)\b",
        r"\bexp\s+date\b",
        r"\brouting\s+number\b",
        r"\biban\b",
        r"\baccount\s+number\b",
        r"\bbanking\s+details\b",
        r"\bpayment\s+details\b",
        r"\bcardholder(?:\s+name)?\b",
    ],
    "Money/Payment": [
        r"\btransfer\s+(?:the\s+)?(?:money|funds|amount|payment|cash|balance)\b",
        r"\bsend\s+(?:the\s+)?(?:money|cash|funds|payment|crypto|bitcoin|usdt)\b",
        r"\bwire\s+(?:transfer|the\s+funds?)\b",
        r"\bmake\s+(?:a\s+)?payment\b",
        r"\bpay\s+(?:the\s+)?(?:fee|amount|charges?|penalty|fine|release\s+fee|due|ransom|tax|me)\b",
        r"\bdeposit\s+(?:money|cash|funds)\b",
        r"\bgift\s+cards?\b",
        r"[₹$€£]\s*\d+",
        r"\b\d+\s*(?:rupees?|dollars?|inr|usd|crypto|bitcoins?|btc|eth|usdt)\b",
    ],
    "Personal Information": [
        r"\bpersonal\s+(?:details|information|data)\b",
        r"\baadhaar(?:\s+card|\s+number)?\b",
        r"\bssn\b",
        r"\bsocial\s+security(?:\s+number)?\b",
        r"\bnational\s+id\b",
        r"\bpassport(?:\s+details|\s+number)?\b",
        r"\bdate\s+of\s+birth\b",
        r"\bdob\b",
        r"\bidentity\s+documents?\b",
        r"\bhome\s+address\b",
        r"\bresidential\s+address\b",
        r"\bmother'?s\s+maiden\s+name\b",
        r"\bgovernment\s+id\b",
    ],
    "Login/Credentials": [
        r"\blogin\s+credentials\b",
        r"\busername\s+and\s+password\b",
        r"\bsign[-\s]?in\s+credentials\b",
        r"\baccount\s+credentials\b",
        r"\blogin\s+(?:details|info|information)\b",
        r"\buser\s*id\s+and\s+password\b",
        r"\baccess\s+credentials\b",
    ],
    "Identity Verification": [
        r"\bverify\s+(?:your\s+)?(?:identity|account|details|profile|wallet|card)\b",
        r"\bconfirm\s+(?:your\s+)?(?:identity|account|details|profile|wallet|card)\b",
        r"\bidentity\s+verification\b",
        r"\baccount\s+verification\b",
        r"\bkyc\s+verification\b",
        r"\bcomplete\s+(?:your\s+)?kyc\b",
        r"\bvalidate\s+(?:your\s+)?(?:account|identity)\b",
        r"\breactivate\s+(?:your\s+)?account\b",
    ],
    "Link/URL": [
        r"https?://\S+",
        r"www\.\S+",
        r"\bbit\.ly/\S+",
        r"\btinyurl\.com/\S+",
        r"\bclick\s+(?:on\s+)?(?:this\s+|the\s+)?(?:link|url|button)\b",
        r"\bopen\s+(?:the\s+)?(?:link|url|attachment)\b",
        r"\bvisit\s+(?:this\s+)?(?:link|url|website|portal)\b",
        r"\bfollowing\s+link\b",
        r"\blink\s+(?:provided|below|here)\b",
    ],
    "Software Installation": [
        r"\binstall\s+(?:this\s+)?(?:app|software|file|apk|tool|extension|plugin|certificate)\b",
        r"\bdownload\s+(?:and\s+install|app|file|software|apk|tool)\b",
        r"\banydesk\b",
        r"\bteamviewer\b",
        r"\bquicksupport\b",
        r"\bremote\s+access(?:\s+software|\s+tool)?\b",
        r"\bremote\s+desktop\b",
        r"\brun\s+(?:this\s+)?(?:file|program|executable|\.exe)\b",
    ],
}


# ---------------------------------------------------------------------
# MANIPULATION INDICATOR PATTERNS
# ---------------------------------------------------------------------
MANIPULATION_PATTERNS = {
    "Urgency": [
        r"\bimmediately\b",
        r"\burgent(?:ly)?\b",
        r"\bright\s+now\b",
        r"\basap\b",
        r"\bhurry\b",
        r"\bwithin\s+\d+\s+(?:minutes?|hours?|seconds?)\b",
        r"\bact\s+(?:fast|now)\b",
        r"\bexpires?\s+(?:soon|in\s+\d+|today)\b",
        r"\bwithout\s+delay\b",
        r"\binstant(?:ly)?\b",
        r"\btime\s+is\s+running\s+out\b",
        r"\blast\s+chance\b",
        r"\bat\s+once\b",
    ],
    "Threat/Fear": [
        r"\b(?:(?:will\s+be|is|are|has\s+been)\s+(?:blocked|suspended|terminated|deactivated|closed|frozen|locked|cancelled))\b",
        r"\blose\s+access\b",
        r"\blegal\s+action\b",
        r"\barrest\b",
        r"\blawsuit\b",
        r"\bpenalty\b",
        r"\bfine\s+of\b",
        r"\b(?:payment|transaction)\s+failed\b",
        r"\b(?:account|security)\s+(?:compromised|alert|breach|incident)\b",
        r"\bunauthorized\s+(?:access|transaction|activity|charge)\b",
        r"\baction\s+will\s+be\s+taken\b",
        r"\bface\s+(?:severe\s+)?consequences\b",
        r"\breport(?:ed)?\s+to\s+(?:police|authorities)\b",
    ],
    "Authority/Impersonation": [
        r"\b(?:(?:this\s+is|i\s+am|speaking\s+from)\s+(?:the\s+)?(?:bank|it\s+department|it\s+support|security\s+team|police|fraud\s+department|customs|tax\s+department|customer\s+(?:care|service|support)|administrator|compliance\s+officer|cyber\s+crime|helpdesk))\b",
        r"\bbank\s+(?:manager|official|security\s+team|representative)\b",
        r"\bit\s+support(?:\s+team)?\b",
        r"\bfraud\s+department\b",
        r"\bofficer\s+from\b",
        r"\bcalling\s+from\s+(?:your\s+)?bank\b",
        r"\bofficial\s+support\b",
    ],
    "Secrecy": [
        r"\bdon'?t\s+tell\s+(?:anyone|anybody)\b",
        r"\bkeep\s+(?:this\s+)?(?:confidential|secret|between\s+us|to\s+yourself)\b",
        r"\bdo\s+not\s+(?:disclose|share\s+with\s+anyone|tell\s+anyone|discuss)\b",
        r"\bconfidential\s+matter\b",
        r"\bkeep\s+it\s+quiet\b",
    ],
}


def split_into_clauses(text: str) -> List[str]:
    """Splits conversational text into sentences or major clauses."""
    raw_clauses = re.split(r"[.!?;\n]+", text)
    return [c.strip() for c in raw_clauses if c.strip()]


def detect_requests(text: str) -> List[str]:
    """
    Detects sensitive request categories made in the conversation.
    Filters out defensive or security-awareness statements.
    """
    clauses = split_into_clauses(text)
    detected = set()

    for clause in clauses:
        # If this clause is explicitly educational/defensive, skip request extraction for it
        if is_sentence_defensive(clause):
            continue

        clause_lower = clause.lower()
        for category, patterns in REQUEST_PATTERNS.items():
            if category in detected:
                continue
            for pattern in patterns:
                if re.search(pattern, clause_lower, re.IGNORECASE):
                    detected.add(category)
                    break

    return sorted(list(detected))


def detect_manipulations(text: str, detected_requests: List[str]) -> List[str]:
    """
    Detects psychological manipulation indicators present in the text.
    Filters out indicators inside defensive/advisory statements.
    """
    clauses = split_into_clauses(text)
    indicators = set()

    for clause in clauses:
        if is_sentence_defensive(clause):
            continue

        clause_lower = clause.lower()
        for category, patterns in MANIPULATION_PATTERNS.items():
            if category in indicators:
                continue
            for pattern in patterns:
                if re.search(pattern, clause_lower, re.IGNORECASE):
                    indicators.add(category)
                    break

    # Sensitive Information Request indicator triggers if active sensitive requests exist
    sensitive_categories = {
        "OTP",
        "Password",
        "Bank/Card Details",
        "Login/Credentials",
        "Personal Information",
    }
    if any(req in sensitive_categories for req in detected_requests):
        indicators.add("Sensitive Information Request")

    return sorted(list(indicators))


def compute_risk_score(
    prediction: str,
    class_probs: Dict[str, float],
    detected_requests: List[str],
    manipulation_indicators: List[str],
    is_defensive_statement: bool,
) -> int:
    """
    Computes an explainable, evidence-driven Risk Score from 0 to 100.

    Key Principles:
    1. If the text is purely an educational/defensive statement (e.g. 'Never share your password'),
       the risk remains minimal (LOW).
    2. If NO sensitive requests and NO manipulation indicators are found, the message
       is benign, and the score remains strictly LOW (< 30) regardless of ML classification.
    3. The ML model acts as a secondary supporting signal (+5 for Suspicious, +10 for Fake),
       never independently pushing benign text to HIGH or CRITICAL.
    4. Strong social-engineering evidence (e.g., OTP + Urgency + Authority) dynamically scales
       points to HIGH (60-84) or CRITICAL (85-100).
    """
    # 1. Pure defensive / awareness statement check
    if is_defensive_statement and not detected_requests and not manipulation_indicators:
        return 5

    # 2. Benign conversation check (zero requests, zero manipulation cues)
    if not detected_requests and not manipulation_indicators:
        # Base benign score with minor secondary ML signal
        if prediction == "Fake":
            return 15
        elif prediction == "Suspicious":
            return 10
        else:
            return 5

    # 3. Evidence-based scoring when actual indicators are detected
    score = 0

    # Request points
    for req in detected_requests:
        if req in ["OTP", "Password", "Bank/Card Details"]:
            score += 35
        elif req in ["Software Installation"]:
            score += 30
        elif req in ["Login/Credentials", "Personal Information"]:
            score += 25
        elif req in ["Money/Payment", "Identity Verification"]:
            score += 20
        elif req == "Link/URL":
            score += 15

    # Manipulation indicator points
    if "Threat/Fear" in manipulation_indicators:
        score += 20
    if "Urgency" in manipulation_indicators:
        score += 15
    if "Authority/Impersonation" in manipulation_indicators:
        score += 15
    if "Secrecy" in manipulation_indicators:
        score += 10
    if "Sensitive Information Request" in manipulation_indicators:
        score += 10

    # Combined synergy bonus for high-risk combinations
    active_manip_count = len([m for m in manipulation_indicators if m != "Sensitive Information Request"])
    if active_manip_count >= 2:
        score += 10
    if active_manip_count >= 3:
        score += 5

    has_critical_req = any(r in ["OTP", "Password", "Bank/Card Details"] for r in detected_requests)
    has_high_pressure = any(m in ["Threat/Fear", "Urgency", "Authority/Impersonation"] for m in manipulation_indicators)
    if has_critical_req and has_high_pressure:
        score += 15

    # Secondary ML signal: provides modest calibration if evidence exists
    if prediction == "Fake":
        score += 10
    elif prediction == "Suspicious":
        score += 5

    return max(0, min(100, score))


def get_risk_level(risk_score: int) -> str:
    """
    Maps 0-100 risk score to standard tier:
    - 0-29: LOW
    - 30-59: MEDIUM
    - 60-84: HIGH
    - 85-100: CRITICAL
    """
    if risk_score < 30:
        return "LOW"
    elif risk_score < 60:
        return "MEDIUM"
    elif risk_score < 85:
        return "HIGH"
    else:
        return "CRITICAL"


def get_recommendation(risk_level: str) -> str:
    """Returns clear, actionable security advice based on risk level."""
    recommendations = {
        "LOW": "No significant social-engineering indicators detected. Conversation appears normal.",
        "MEDIUM": "Exercise caution and verify the request before sharing any information.",
        "HIGH": "Do not share sensitive information. Independently verify the sender through an official channel.",
        "CRITICAL": "Do not provide OTP, credentials, banking information, or payment. Verify the request through an official channel.",
    }
    return recommendations.get(risk_level, recommendations["LOW"])


def analyze_conversation(conversation_text: str) -> Dict[str, Any]:
    """
    Analyzes a conversation string and returns an explainable security analysis.

    Parameters:
        conversation_text (str): Input text transcript, chat message, or email.

    Returns:
        dict: Standardized security analysis dictionary containing:
            - classification (str): 'Real', 'Suspicious', or 'Fake'
            - confidence (float): Model confidence for predicted class
            - risk_score (int): 0-100 evidence-based score
            - risk_level (str): 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
            - detected_requests (List[str]): List of detected items requested
            - manipulation_indicators (List[str]): List of psychological manipulation cues
            - recommendation (str): Actionable advice
    """
    if not isinstance(conversation_text, str) or not conversation_text.strip():
        return {
            "classification": "Real",
            "confidence": 0.0,
            "risk_score": 0,
            "risk_level": "LOW",
            "detected_requests": [],
            "manipulation_indicators": [],
            "recommendation": get_recommendation("LOW"),
        }

    # 1. Load pre-trained model and vectorizer from disk
    model, vectorizer = load_artifacts()

    # 2. Transform text using pre-fitted TF-IDF
    X_tfidf = vectorizer.transform([conversation_text])

    # 3. Model prediction
    pred_class = model.predict(X_tfidf)[0]

    # 4. Probabilities & confidence
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_tfidf)[0]
        class_probs = {cls_name: float(prob) for cls_name, prob in zip(model.classes_, probs)}
        confidence = float(class_probs.get(pred_class, 1.0))
    else:
        class_probs = {pred_class: 1.0}
        confidence = 1.0

    # 5. Content analysis: requests and manipulation cues
    detected_reqs = detect_requests(conversation_text)
    manipulation_cues = detect_manipulations(conversation_text, detected_reqs)

    # 6. Check if text is overall defensive / educational
    clauses = split_into_clauses(conversation_text)
    is_defensive = any(is_sentence_defensive(c) for c in clauses)

    # 7. Compute evidence-based risk score & level
    risk_score = compute_risk_score(
        prediction=pred_class,
        class_probs=class_probs,
        detected_requests=detected_reqs,
        manipulation_indicators=manipulation_cues,
        is_defensive_statement=is_defensive,
    )
    risk_level = get_risk_level(risk_score)

    # 8. Actionable recommendation
    recommendation = get_recommendation(risk_level)

    return {
        "classification": pred_class,
        "confidence": round(confidence, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "detected_requests": detected_reqs,
        "manipulation_indicators": manipulation_cues,
        "recommendation": recommendation,
    }


# Convenience alias for predict function
predict = analyze_conversation


if __name__ == "__main__":
    sample_text = (
        "This is bank customer support. Your account is suspended. "
        "Send me your OTP immediately or legal action will be taken."
    )
    result = analyze_conversation(sample_text)
    import json
    print(json.dumps(result, indent=2))
