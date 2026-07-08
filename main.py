import os
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from persona import SYSTEM_PROMPT, SYSTEM_PROMPT_JD

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://avinashpothireddy.github.io",
        "http://localhost:8080",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── rate limiting ────────────────────────────────────────────────────────
# ponytail: in-memory dict, resets on restart, not shared across workers.
# Fine at portfolio-site traffic scale with --workers 1. Revisit with
# redis/slowapi only if this ever needs to survive restarts or scale out.
_WINDOW_SECONDS = 60
_LIMITS = {"chat": 8, "jd-fit": 3}
_hits: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    return (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )


def _enforce_rate_limit(request: Request, bucket: str) -> None:
    key = f"{bucket}:{_client_ip(request)}"
    now = time.time()
    hits = _hits[key]
    hits[:] = [t for t in hits if now - t < _WINDOW_SECONDS]
    if len(hits) >= _LIMITS[bucket]:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a bit.")
    hits.append(now)


# ── schemas ──────────────────────────────────────────────────────────────
class HistoryItem(BaseModel):
    role: str
    content: str = Field(..., max_length=800)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=800)
    history: list[HistoryItem] = Field(default_factory=list, max_length=6)


class ChatResponse(BaseModel):
    reply: str


class JdFitRequest(BaseModel):
    jd_text: str = Field(..., min_length=20, max_length=6000)


class JdFitResult(BaseModel):
    fit_score: int
    summary: str
    strengths: list[str]
    gaps: list[str]


# ── endpoints ────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    _enforce_rate_limit(request, "chat")

    contents = []
    for item in req.history:
        role = "model" if item.role == "agent" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=item.content)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=req.message)]))

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=700,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream model error")

    return ChatResponse(reply=resp.text or "")


@app.post("/api/jd-fit", response_model=JdFitResult)
def jd_fit(req: JdFitRequest, request: Request):
    _enforce_rate_limit(request, "jd-fit")

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=req.jd_text)])],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_JD,
                response_mime_type="application/json",
                response_schema=JdFitResult,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return JdFitResult.model_validate_json(resp.text)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream model error")
