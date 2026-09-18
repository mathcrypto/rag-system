# FastAPI app — API only (no HTML/CSS). Frontend lives in /frontend.
#
# Clients: browser (CORS) or curl → POST /api/ask → RagService → ask().
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AskRequest, AskResponse, HealthResponse
from api.service import RagService

app = FastAPI(
    title="RAG System API",
    description="Backend only. Chat UI is in the frontend/ package.",
    version="0.1.0",
)

_cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_service = RagService()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return _service.health()


@app.post("/api/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return _service.ask(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
