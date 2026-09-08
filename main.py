import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-backend")

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel, Field

from persona import SYSTEM_PROMPT, SYSTEM_PROMPT_JD

client = Groq(api_key=os.environ["GROQ_API_KEY"])
# Swappable via env so a newer free Groq model can be pointed at without a code
# change. Defaults to the model currently in production if GROQ_MODEL is unset.
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Shared secret gating /api/stats. Read from env; if unset, /api/stats stays
# closed (returns 401) rather than exposing counters. Never hardcode a token.
STATS_TOKEN = os.environ.get("STATS_TOKEN", "")

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


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    logger.error("422 on %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ── rate limiting ────────────────────────────────────────────────────────
# ponytail: in-memory dict, resets on restart, not shared across workers.
# Fine at portfolio-site traffic scale with --workers 1. Revisit with
# redis/slowapi only if this ever needs to survive restarts or scale out.
_WINDOW_SECONDS = 60
_LIMITS = {"chat": 8, "jd-fit": 3, "track": 30}
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


# ── event tracking (anonymous, free-tier only) ─────────────────────────────
# Counters live in memory only — no DB, bucket, or file. They reset on cold
# start (fine for a lightweight dashboard) and are consistent because the
# server runs --workers 1. The durable record is the structured logger.info
# "EVENT ..." line below, captured by Cloud Logging (well within its free
# tier). We never store IPs, personal data, or full JD text — only a count,
# and for jd_fit an optional fit_score plus a short role label.
ALLOWED_EVENTS = {"visit", "chat", "jd_fit", "book_click"}
_counts_all: dict[str, int] = defaultdict(int)
_counts_today: dict[str, int] = defaultdict(int)
_today: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _roll_day() -> None:
    """Reset the per-day counters when the UTC date rolls over."""
    global _today, _counts_today
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if now != _today:
        _today = now
        _counts_today = defaultdict(int)


# ── schemas ──────────────────────────────────────────────────────────────
class HistoryItem(BaseModel):
    role: str
    content: str = Field(..., max_length=3000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=6000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    reply: str


class TrackRequest(BaseModel):
    event: str = Field(..., max_length=40)
    meta: dict | None = None


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


@app.post("/api/track")
def track(req: TrackRequest, request: Request):
    _enforce_rate_limit(request, "track")

    if req.event not in ALLOWED_EVENTS:
        raise HTTPException(status_code=400, detail="Unknown event")

    _roll_day()
    _counts_all[req.event] += 1
    _counts_today[req.event] += 1

    # Only jd_fit carries optional, non-identifying detail: a numeric score and
    # a short role label. Anything else in meta is ignored — never logged.
    extra = ""
    if req.event == "jd_fit" and isinstance(req.meta, dict):
        parts = []
        score = req.meta.get("fit_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            parts.append("fit_score=%d" % max(0, min(100, int(score))))
        role = req.meta.get("role")
        if isinstance(role, str) and role.strip():
            parts.append("role=%r" % role.strip()[:60])
        if parts:
            extra = " " + " ".join(parts)

    # Durable, anonymous record via Cloud Logging (no IP, no personal data).
    logger.info("EVENT event=%s%s", req.event, extra)
    return {"ok": True}


@app.get("/api/stats")
def stats(request: Request):
    token = request.headers.get("X-Stats-Token", "")
    if not STATS_TOKEN or token != STATS_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    _roll_day()
    return {
        "date": _today,
        "today": {e: _counts_today.get(e, 0) for e in sorted(ALLOWED_EVENTS)},
        "all_time": {e: _counts_all.get(e, 0) for e in sorted(ALLOWED_EVENTS)},
    }


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
            max_completion_tokens=1200,
            reasoning_effort="low",
        )
    except Exception as e:
        logger.error("chat: Groq call failed: %s: %s", type(e).__name__, e)
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
            max_completion_tokens=2048,
            reasoning_effort="low",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "jd_fit_result", "strict": True, "schema": _JD_FIT_SCHEMA},
            },
        )
    except Exception as e:
        logger.error("jd_fit: Groq call failed: %s: %s", type(e).__name__, e)
        raise HTTPException(status_code=502, detail="Upstream model error")

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        logger.error("jd_fit: empty content from model")
        raise HTTPException(status_code=502, detail="Empty model response")

    # Primary path: strict validation.
    try:
        return JdFitResult.model_validate_json(raw)
    except Exception as first_err:
        # Salvage path: extract the outermost JSON object and coerce fields,
        # so a stray prefix or a mildly malformed payload doesn't 502.
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            payload = json.loads(match.group(0)) if match else json.loads(raw)
            return JdFitResult(
                fit_score=int(payload.get("fit_score", 0)),
                summary=str(payload.get("summary", "")).strip(),
                strengths=[str(s).strip() for s in payload.get("strengths", []) if str(s).strip()],
                gaps=[str(g).strip() for g in payload.get("gaps", []) if str(g).strip()],
            )
        except Exception as second_err:
            logger.error(
                "jd_fit: parse failed. strict=%s salvage=%s raw[:300]=%r",
                first_err, second_err, raw[:300],
            )
            raise HTTPException(status_code=502, detail="Could not parse fit analysis")