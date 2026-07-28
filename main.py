import json
import logging
import os
import re
import time
from collections import defaultdict

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


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    logger.error("422 on %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

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
    content: str = Field(..., max_length=3000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=6000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=12)


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