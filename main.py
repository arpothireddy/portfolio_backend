import os
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

from persona import SYSTEM_PROMPT, SYSTEM_PROMPT_JD

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://arpothireddy.github.io",
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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in req.history:
        role = "assistant" if item.role == "agent" else "user"
        messages.append({"role": role, "content": item.content})
    messages.append({"role": "user", "content": req.message})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.4,
            max_completion_tokens=700,
            reasoning_effort="low",
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream model error")

    return ChatResponse(reply=resp.choices[0].message.content or "")


_JD_FIT_SCHEMA = JdFitResult.model_json_schema()
_JD_FIT_SCHEMA["additionalProperties"] = False


@app.post("/api/jd-fit", response_model=JdFitResult)
def jd_fit(req: JdFitRequest, request: Request):
    _enforce_rate_limit(request, "jd-fit")

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_JD},
                {"role": "user", "content": req.jd_text},
            ],
            max_completion_tokens=1536,
            reasoning_effort="low",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "jd_fit_result", "strict": True, "schema": _JD_FIT_SCHEMA},
            },
        )
        return JdFitResult.model_validate_json(resp.choices[0].message.content)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream model error")
