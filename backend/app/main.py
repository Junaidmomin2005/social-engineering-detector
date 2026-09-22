"""
FastAPI application entrypoint for Conversational Liveness & Social Engineering Detector.

Endpoints:
- GET /: Serves the web interface (frontend/index.html).
- POST /analyze: Accepts conversation_text JSON, performs security analysis, and returns JSON.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.app.predict import analyze_conversation, load_artifacts


BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload ML model and TF-IDF vectorizer at startup for instant inference
    load_artifacts()
    yield


app = FastAPI(
    title="Conversational Liveness & Social Engineering Detector",
    description="Analyzes conversational text for social engineering cues, manipulation indicators, and liveness.",
    version="1.0.0",
    lifespan=lifespan,
)


class AnalyzeRequest(BaseModel):
    conversation_text: str = Field(
        ...,
        description="The conversational text transcript, email, or chat message to analyze.",
    )


@app.get("/", response_class=FileResponse)
def serve_index():
    """Serves the plain HTML/CSS/JS frontend interface."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return FileResponse(INDEX_HTML)


@app.post("/analyze")
def analyze_endpoint(request: AnalyzeRequest):
    """
    Analyzes conversation text using the trained ML model and rule-based extractors.
    Validates input and returns structured security analysis.
    """
    text = request.conversation_text.strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Conversation text cannot be empty or contain only whitespace.",
        )

    try:
        result = analyze_conversation(text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing security analysis: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
