# Conversational Liveness & Social Engineering Detector

An academic machine learning and natural language processing project designed to detect synthetic/manipulated communications (conversational liveness) and identify social engineering risks in multi-channel textual interactions.

---

## 1. Project Overview & Objectives

In modern digital communication, social engineering attacks frequently leverage synthetic conversational agents, voice cloning transcripts, and generative text to manipulate targets. This project implements a lightweight, explainable, and production-ready ML/NLP system tailored for local execution on standard consumer hardware.

### Core Objectives:
1. **Primary Prediction Target — Conversational Liveness / Authenticity:**
   - Classify conversations into `Real`, `Suspicious`, or `Fake`.
2. **Secondary Prediction Target — Threat Severity Assessment:**
   - Gauge social engineering risk into `Low`, `Medium`, `High`, or `Critical`.
3. **Explainable Indicators:**
   - Extract psychological triggers (urgency, emotional manipulation, verification evasion) and linguistic cues.

---

## 2. System Architecture

```text
Social-engineering-detector/
│
├── data/                                 # Raw dataset store
│   └── deepfake_conversation_factor_analysis.csv
│
├── backend/
│   ├── app/                              # FastAPI application layer
│   │   ├── main.py                       # API routes & entrypoint
│   │   ├── schemas.py                    # Pydantic request/response models
│   │   ├── services/                     # Business logic & inference services
│   │   │   ├── nlp_service.py            # Text normalization & vectorization
│   │   │   ├── prediction_service.py     # Model inference orchestration
│   │   │   ├── risk_service.py           # Heuristic & ML threat scoring
│   │   │   └── explanation_service.py    # Feature attributions & trigger cues
│   │   └── models/                       # Active serialized model artifacts
│   │
│   ├── ml/                               # ML experimentation & pipeline scripts
│   │   ├── preprocess.py                 # Data hygiene, split, leak-prevention
│   │   ├── features.py                   # TF-IDF & tabular feature engineering
│   │   ├── train.py                      # Multi-model training pipeline
│   │   ├── evaluate.py                   # Classification metrics & confusion matrix
│   │   └── artifacts/                    # Exported pipelines & evaluation summaries
│   │
│   └── requirements.txt                  # Minimal Python dependencies
│
├── frontend/                             # Lightweight web UI
│
├── reports/                              # Academic documentation, EDA & evaluation reports
│
├── tests/                                # Unit and integration test suite
│
├── README.md                             # Project documentation
└── .gitignore                            # VCS ignore rules
```

---

## 3. Technology Stack & Key Decisions

- **Language:** Python (3.10+)
- **ML & NLP Engine:** `scikit-learn`, `pandas`, `numpy` (TF-IDF N-grams, linear classifiers, ensemble trees).
- **Backend API:** `FastAPI` + `Uvicorn` + `Pydantic`.
- **Hardware Profile:** Fully lightweight; runs locally on CPU with zero GPU/CUDA dependencies.
- **Architectural Constraints:**
  - No external cloud LLM APIs required.
  - No heavy database setup required (stateless inference).
  - Explicit data leakage prevention safeguards applied across preprocessing.

---

## 4. Setup & Installation

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 5. Development Roadmap

- [x] **Phase 1: Dataset Inspection & EDA** (Complete)
- [x] **Phase 2: Project Scaffolding & Directory Setup** (Complete)
- [ ] **Phase 3: Data Preprocessing & Leakage Mitigation**
- [ ] **Phase 4: ML/NLP Pipeline Training & Evaluation**
- [ ] **Phase 5: FastAPI Service Layer Implementation**
- [ ] **Phase 6: Frontend Development & End-to-End Verification**
