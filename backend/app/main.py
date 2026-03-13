"""
PitchPal v2 - FastAPI Backend
REST API + WebSocket streaming for real-time ReAct agent evaluation

Phase 1 Security additions:
  - IP-based rate limiting (startup: 3/24h, investor: 5/24h)
  - Input sanitization (HTML stripping, length enforcement)
  - Prompt injection detection
  - Fixed logger import (was missing, caused NameError on deck upload errors)
"""

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime
from typing import List, Dict
import threading

from app.logger import setup_json_logging
from app.metrics import metrics

# ── Switch to structured JSON logging immediately ──────────────
setup_json_logging()
logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.agent.react_agent import ReActAgent
from app.agent.semantic_cache import semantic_cache
from app.agent.evaluation_cache import evaluation_cache
from app.agent.rate_limiter import rate_limiter, RateLimitExceeded
from app.agent.share_store import share_store
from app.security import sanitize_inputs
from app.models.schemas import (
    PitchInput,
    EvaluationResponse,
    HealthResponse,
    AgentStep,
    DeckUploadResponse,
    DeckQuality,
    SAMPLE_PITCHES,
)


# =============================
# FastAPI App
# =============================

app = FastAPI(
    title="PitchPal v2 API",
    description="AI-powered startup pitch evaluator using a real ReAct agent with web search",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Track stats
start_time = datetime.now()
evaluation_count = 0

# Investor session tokens: {token: expiry_timestamp}
investor_tokens: Dict[str, float] = {}
_token_lock = threading.Lock()
TOKEN_TTL = 6 * 3600  # 6 hours


# ── IP Extraction helper ──────────────────────────────────────

def get_client_ip(request: Request) -> str:
    """
    Extract the real client IP, respecting X-Forwarded-For for reverse proxies.
    Falls back to direct connection IP.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can be a comma-separated list; first entry is the client
        ip = forwarded_for.split(",")[0].strip()
        return ip
    if request.client:
        return request.client.host
    return "unknown"


# =============================
# REST Endpoints
# =============================


@app.get("/", response_model=HealthResponse)
async def root():
    agent = ReActAgent()
    return HealthResponse(
        status="healthy", version="2.0.0",
        llm_provider=agent.llm.get_provider_name(), timestamp=datetime.now(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    agent = ReActAgent()
    return HealthResponse(
        status="healthy", version="2.0.0",
        llm_provider=agent.llm.get_provider_name(), timestamp=datetime.now(),
    )


@app.get("/sample-pitches")
async def get_sample_pitches():
    return {"pitches": SAMPLE_PITCHES}


# ── Rate Limit Status ──────────────────────────────────────────

@app.get("/rate-limit/status")
async def get_rate_limit_status(request: Request, role: str = "startup"):
    """
    Return the current rate limit status for the requesting IP.
    Safe to poll on page load — read-only, no side effects.
    """
    if role not in ("startup", "investor"):
        role = "startup"
    ip = get_client_ip(request)
    return rate_limiter.get_status(ip, role)


# ── Investor Access Code Verification ──────────────────────────

class VerifyCodeRequest(BaseModel):
    code: str

@app.post("/verify-code")
async def verify_investor_code(req: VerifyCodeRequest):
    if req.code != settings.INVESTOR_ACCESS_CODE:
        raise HTTPException(status_code=403, detail="Invalid access code")
    token = secrets.token_urlsafe(32)
    with _token_lock:
        investor_tokens[token] = time.time() + TOKEN_TTL
        now = time.time()
        expired = [t for t, exp in investor_tokens.items() if exp < now]
        for t in expired:
            del investor_tokens[t]
    return {"status": "ok", "token": token, "expires_in": TOKEN_TTL}


def validate_investor_token(token: str) -> bool:
    with _token_lock:
        if not token or token not in investor_tokens:
            return False
        if time.time() > investor_tokens[token]:
            del investor_tokens[token]
            return False
        return True


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pitch(pitch_input: PitchInput, request: Request):
    global evaluation_count

    # ── Input sanitization ──────────────────────────────────
    clean_pitch, clean_name, error_msg = sanitize_inputs(
        pitch_input.pitch_text, pitch_input.startup_name
    )
    if error_msg:
        raise HTTPException(status_code=422, detail=error_msg)

    # ── Check evaluation cache first (cache hits bypass rate limit) ──
    cached = evaluation_cache.get(clean_pitch, "startup")
    if cached:
        evaluation_count += 1
        from app.models.schemas import PitchEvaluation
        return EvaluationResponse(
            startup_name=clean_name,
            evaluation=PitchEvaluation(**cached["evaluation"]),
            agent_steps=[AgentStep(**s) for s in cached.get("steps", [])],
            processing_time=0.0,
            timestamp=datetime.now(),
            llm_provider=cached.get("llm_provider", "cached"),
        )

    # ── Rate limit check (only for fresh evaluations) ──────
    ip = get_client_ip(request)
    try:
        count, limit = rate_limiter.check_and_increment(ip, "startup")
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": str(e.retry_after_seconds)},
        )

    start = time.time()
    agent = ReActAgent()

    try:
        evaluation, steps = await agent.evaluate_pitch(
            pitch_text=clean_pitch,
            startup_name=clean_name,
        )

        evaluation_count += 1
        processing_time = time.time() - start

        # Cache for future
        evaluation_cache.set(
            pitch_text=clean_pitch,
            role="startup",
            startup_name=clean_name,
            evaluation_data=evaluation.model_dump(),
            steps_data=[s.model_dump() for s in steps],
            processing_time=round(processing_time, 2),
            llm_provider=agent.llm.get_provider_name(),
        )

        return EvaluationResponse(
            startup_name=clean_name,
            evaluation=evaluation,
            agent_steps=steps,
            processing_time=round(processing_time, 2),
            timestamp=datetime.now(),
            llm_provider=agent.llm.get_provider_name(),
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Evaluation failed. Please try again.")


@app.get("/metrics")
async def get_metrics():
    """Live metrics: latency percentiles, cache hit rate, agent stats, error rate."""
    return metrics.get_snapshot()


@app.get("/stats")
async def get_stats():
    uptime = datetime.now() - start_time
    return {
        "total_evaluations": evaluation_count,
        "uptime_seconds": round(uptime.total_seconds()),
        "uptime_formatted": str(uptime).split(".")[0],
        "status": "operational",
        "semantic_cache": semantic_cache.get_stats(),
        "evaluation_cache": evaluation_cache.get_stats(),
        "rate_limiter": rate_limiter.get_global_stats(),
        "share_store": share_store.get_stats(),
    }


# ── Shareable Evaluation Links ────────────────────────────────

class ShareRequest(BaseModel):
    evaluation: dict
    startup_name: str
    role: str = "startup"
    processing_time: float = 0.0
    llm_provider: str = "unknown"


@app.post("/share")
async def create_share_link(req: ShareRequest, request: Request):
    """
    Create a shareable link for an evaluation result.
    Returns a share_id that maps to /eval/{share_id} on the frontend.
    """
    share_id = share_store.create(
        evaluation=req.evaluation,
        startup_name=req.startup_name,
        role=req.role,
        processing_time=req.processing_time,
        llm_provider=req.llm_provider,
    )
    frontend_url = settings.FRONTEND_URL.rstrip("/")
    return {
        "share_id": share_id,
        "url": f"{frontend_url}/eval/{share_id}",
        "expires_in_days": 7,
    }


@app.get("/eval/{share_id}")
async def get_shared_evaluation(share_id: str):
    """
    Retrieve a shared evaluation by its ID.
    This is the data source for the public /eval/[id] frontend page.
    """
    entry = share_store.get(share_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="Shared evaluation not found or has expired (links expire after 7 days)."
        )
    return entry


@app.delete("/cache/clear")
async def clear_cache():
    semantic_cache.clear()
    evaluation_cache.clear()
    return {"status": "ok", "message": "All caches cleared (semantic + evaluation)"}


class DeleteCacheEntryRequest(BaseModel):
    pitch_text: str
    role: str = "startup"

@app.delete("/cache/entry")
async def delete_cache_entry(req: DeleteCacheEntryRequest):
    """Delete a single evaluation cache entry by pitch_text + role."""
    deleted = evaluation_cache.delete_entry(req.pitch_text, req.role)
    return {
        "status": "ok",
        "deleted": deleted,
        "message": "Entry removed from cache" if deleted else "Entry not found in cache",
    }


# ──────────────────────────────────────────────────────────────
# Deck Upload
# ──────────────────────────────────────────────────────────────

@app.post("/upload-deck", response_model=DeckUploadResponse)
async def upload_deck(file: UploadFile = File(...)):
    """
    Accept a PDF or PPTX pitch deck upload.
    Extracts text for the ReAct agent + analyzes deck quality with Gemini Vision.
    Returns: extracted_text, startup_name, slide_count, deck_quality scores.
    """
    from app.agent.deck_analyzer import DeckAnalyzer, DeckAnalysisError

    # Validate file type
    filename = file.filename or "upload.pdf"
    allowed = (".pdf", ".pptx", ".ppt")
    if not any(filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF or PPTX file."
        )

    # Early size check before reading into memory (prevents DoS)
    if file.size and file.size > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20 MB.")

    # Read file bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    # Run deck analysis
    try:
        analyzer = DeckAnalyzer(
            gemini_api_key=settings.GEMINI_API_KEY,
            gemini_model=settings.GEMINI_MODEL,
        )
        result = await analyzer.analyze(file_bytes, filename)
    except DeckAnalysisError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Deck analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Deck analysis failed. Please try again.")

    return DeckUploadResponse(
        startup_name=result["startup_name"],
        extracted_text=result["extracted_text"],
        slide_count=result["slide_count"],
        file_format=result["file_format"],
        deck_quality=DeckQuality(**result["deck_quality"]),
    )


# =============================
# WebSocket — Live Streaming
# =============================


@app.websocket("/ws/evaluate")
async def websocket_evaluate(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ReAct agent streaming.

    Client sends: {"startup_name": "...", "pitch_text": "...", "role": "startup|investor", "token": "..."}
    Server streams: Each AgentStep as JSON, then final evaluation.
    """
    await websocket.accept()
    global evaluation_count

    # Extract IP from WebSocket connection
    client_ip = "unknown"
    if websocket.client:
        client_ip = websocket.client.host
    # Also check forwarded header
    forwarded = websocket.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    try:
        data = await websocket.receive_json()
        startup_name = data.get("startup_name", "").strip()
        pitch_text = data.get("pitch_text", "").strip()
        role = data.get("role", "startup").strip()
        token = data.get("token", "")

        if not startup_name or not pitch_text:
            await websocket.send_json({"type": "error", "message": "Missing startup_name or pitch_text"})
            await websocket.close()
            return

        # Validate role
        if role not in ("startup", "investor"):
            role = "startup"

        # If investor role, validate token
        if role == "investor" and not validate_investor_token(token):
            await websocket.send_json({
                "type": "error",
                "message": "Invalid or expired investor token. Please re-enter access code."
            })
            await websocket.close()
            return

        # ── Input sanitization ──────────────────────────────
        clean_pitch, clean_name, error_msg = sanitize_inputs(pitch_text, startup_name)
        if error_msg:
            await websocket.send_json({"type": "error", "message": error_msg})
            await websocket.close()
            return

        # ── Check evaluation cache first (cache hits bypass rate limit) ──
        cached = evaluation_cache.get(clean_pitch, role)
        if cached:
            logger.info(
                "Evaluation cache hit",
                extra={
                    "event": "eval_cache_hit",
                    "startup": clean_name,
                    "role": role,
                    "client_ip_hash": client_ip[-4:] if client_ip else "unknown",
                }
            )
            await websocket.send_json({
                "type": "start",
                "message": f"Found cached evaluation for {clean_name}. Returning consistent results...",
            })

            # Send a single step indicating cache hit
            cache_step = AgentStep(
                step_number=1,
                step_type="thought",
                content=f"This exact pitch was evaluated before. Returning cached results to ensure consistency. Original evaluation took {cached['processing_time']}s.",
            )
            await websocket.send_json({"type": "step", "step": cache_step.model_dump()})

            await websocket.send_json({
                "type": "complete",
                "evaluation": cached["evaluation"],
                "processing_time": 0.0,
                "total_steps": len(cached.get("steps", [])),
                "llm_provider": cached.get("llm_provider", "cached"),
                "cache_hits": cached.get("cache_hits", 0),
                "from_cache": True,
            })

            metrics.record_evaluation(latency_s=0.0, from_cache=True)
            evaluation_count += 1

        else:
            # ── Rate limit check (only for fresh evaluations) ──────
            try:
                count, limit = rate_limiter.check_and_increment(client_ip, role)
            except RateLimitExceeded as e:
                metrics.record_rate_limit()
                logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "event": "rate_limit_blocked",
                        "startup": clean_name,
                        "role": role,
                        "client_ip_hash": client_ip[-4:] if client_ip else "unknown",
                        "retry_after_seconds": e.retry_after_seconds,
                    }
                )
                await websocket.send_json({
                    "type": "rate_limit",
                    "message": str(e),
                    "retry_after_seconds": e.retry_after_seconds,
                    "limit": e.limit,
                    "window_hours": e.window_hours,
                })
                await websocket.close()
                return

            # ── Similarity check (before running agent) ───────────
            similar_match = None
            pitch_embedding = None
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                embed_result = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=clean_pitch,
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=768,
                )
                pitch_embedding = embed_result["embedding"]
                similar_match = evaluation_cache.find_similar(pitch_embedding, role, threshold=0.87)
            except Exception as e:
                logger.warning(f"Similarity check failed (non-critical): {e}")

            # ── Fresh evaluation ──────────────────────────────
            start_msg: dict = {
                "type": "start",
                "message": f"Starting {'investment analysis' if role == 'investor' else 'evaluation'} of {clean_name}...",
                "rate_limit": {"used": count, "limit": limit},
            }
            if similar_match:
                start_msg["similar_pitch"] = {
                    "startup_name": similar_match["startup_name"],
                    "similarity_pct": round(similar_match["similarity"] * 100),
                }
            await websocket.send_json(start_msg)

            start = time.time()
            agent = ReActAgent()
            hits_before = semantic_cache._hits

            async def on_step(step: AgentStep):
                await websocket.send_json({"type": "step", "step": step.model_dump()})

            evaluation, steps = await agent.evaluate_pitch(
                pitch_text=clean_pitch,
                startup_name=clean_name,
                role=role,
                on_step=on_step,
            )

            processing_time = time.time() - start
            cache_hits = semantic_cache._hits - hits_before

            # Cache the evaluation for future identical pitches
            eval_data = evaluation.model_dump()
            steps_data = [s.model_dump() for s in steps]
            evaluation_cache.set(
                pitch_text=clean_pitch,
                role=role,
                startup_name=clean_name,
                evaluation_data=eval_data,
                steps_data=steps_data,
                processing_time=round(processing_time, 2),
                llm_provider=agent.llm.get_provider_name(),
                cache_hits=cache_hits,
                embedding=pitch_embedding,  # store for future similarity checks
            )

            await websocket.send_json({
                "type": "complete",
                "evaluation": eval_data,
                "processing_time": round(processing_time, 2),
                "total_steps": len(steps),
                "llm_provider": agent.llm.get_provider_name(),
                "cache_hits": cache_hits,
                "rate_limit": {"used": count, "limit": limit},
            })

            # ── Record metrics + structured log ─────────────────────
            contradictions_count = len(evaluation.contradictions)
            tool_call_count = sum(1 for s in steps if s.step_type == "observation")

            metrics.record_evaluation(
                latency_s=round(processing_time, 2),
                from_cache=False,
                tool_calls=tool_call_count,
                contradictions=contradictions_count,
            )

            logger.info(
                "Evaluation complete",
                extra={
                    "event": "evaluation_complete",
                    "startup": clean_name,
                    "role": role,
                    "processing_time_s": round(processing_time, 2),
                    "overall_score": evaluation.overall_score,
                    "recommendation": evaluation.investment_recommendation,
                    "tool_calls": tool_call_count,
                    "semantic_cache_hits": cache_hits,
                    "contradictions_found": contradictions_count,
                    "similar_pitch_detected": similar_match is not None,
                    "llm_provider": agent.llm.get_provider_name(),
                    "client_ip_hash": client_ip[-4:] if client_ip else "unknown",
                }
            )

            evaluation_count += 1

    except WebSocketDisconnect:
        pass
    except Exception as e:
        metrics.record_error()
        logger.error(
            "Evaluation error",
            extra={"event": "evaluation_error", "error": str(e)},
            exc_info=True,
        )
        try:
            await websocket.send_json({"type": "error", "message": "Evaluation failed. Please try again."})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# =============================
# Startup
# =============================

if __name__ == "__main__":
    import os
    import uvicorn
    is_dev = os.getenv("ENV", "development") == "development"
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=is_dev, log_level="info")
