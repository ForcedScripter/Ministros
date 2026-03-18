# ==========================================================
# 🚀 CUSTOMER SERVICE AI — FASTAPI BRIDGE SERVER
# Combines repo features (collections API, Tavily, domain routing)
# with frontend bridge (voice-chat, upload, auth, session mgmt)
# ==========================================================

from dotenv import load_dotenv
load_dotenv()  # Load .env file for local development

import asyncio
import base64
import io
import os
import tempfile
import time
import uuid
from typing import Optional

import pdfplumber
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from auth import decode_token, login_user, signup_user
from collections_api import router as collections_router
from config import COLLECTIONS, resolve_collection_name
from rag_pipeline import run_rag
from sarvam_streaming_stt import transcribe_chunk
from tts import set_voice, speak
from vector_store import (
    create_collection,
    insert_document,
    list_collections,
)

app = FastAPI(title="Customer Service AI")
app.include_router(collections_router)

app.add_middleware(
    CORSMiddleware,
    # Frontend origins – adjust via FRONTEND_ORIGINS env if needed
    allow_origins=os.getenv(
        "FRONTEND_ORIGINS",
        "https://natural-vocl-ai.vercel.app,http://localhost:3000",
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# SESSION TRACKING
# ==========================================================

active_sessions = {}      # user_id -> session collection name (uploaded PDFs)
user_pretrained = {}      # user_id -> pretrained domain (e.g. "car_booking")
user_voices = {}          # user_id -> voice name (e.g. "Kavya", "Aditya")

# Pre-trained collections available in Qdrant (mirrors config.COLLECTIONS)
PRETRAINED_COLLECTIONS = COLLECTIONS  # {"ecommerce": "ecommerce", "car_booking": "car_booking"}


# ==========================================================
# AUTH + RATE LIMITING HELPERS
# ==========================================================


API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "60"))  # requests
API_RATE_WINDOW = int(os.getenv("API_RATE_WINDOW", "60"))  # seconds
_rate_hits: dict[str, list[float]] = {}


def _get_identifier(request: Request, username: Optional[str]) -> str:
    if username:
        return f"user:{username}"
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


async def rate_limiter(
    request: Request,
    current_user: Optional[str] = Depends(lambda: None),
):
    """
    Simple in-process sliding-window rate limiter.
    Uses username when authenticated, otherwise IP.
    """
    key = _get_identifier(request, current_user)
    now = time.time()
    window_start = now - API_RATE_WINDOW

    hits = _rate_hits.get(key, [])
    hits = [t for t in hits if t >= window_start]

    if len(hits) >= API_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
        )

    hits.append(now)
    _rate_hits[key] = hits


def optional_current_user(authorization: str = Header(default="", alias="Authorization")) -> Optional[str]:
    """
    Decode JWT token if present, otherwise return None.
    Does not raise, so it can be used in public endpoints and rate limiting.
    """
    if not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        return None
    return str(payload["sub"])


def get_session_collection(user_id: str) -> str | None:
    return active_sessions.get(user_id)


def ensure_session_collection(user_id: str) -> str:
    if user_id in active_sessions:
        return active_sessions[user_id]
    col_name = f"demo_{user_id[:8]}"
    create_collection(col_name)
    active_sessions[user_id] = col_name
    return col_name


# ==========================================================
# MODELS
# ==========================================================


class ChatRequest(BaseModel):
    query: str
    user_id: str | None = None
    customer_type: str | None = None   # "ecommerce" or "car_booking"


class VoiceOption(BaseModel):
    voice: str
    user_id: str = ""


class CollectionOption(BaseModel):
    collection: str
    user_id: str = ""


class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


# ==========================================================
# HEALTH CHECK
# ==========================================================


@app.get("/health")
async def health():
    try:
        collections = await run_in_threadpool(list_collections)
        return {"status": "ok", "qdrant_status": "connected", "collections": collections}
    except Exception as e:
        return {"status": "ok", "qdrant_status": f"error: {str(e)}", "collections": []}


# ==========================================================
# TEXT CHAT
# ==========================================================


@app.post("/chat")
async def chat(
    req: ChatRequest,
    current_user: Optional[str] = Depends(optional_current_user),
    _: None = Depends(rate_limiter),
):
    uid = req.user_id or current_user or str(uuid.uuid4())

    # domain: explicit customer_type > user's pretrained selection > default
    domain = req.customer_type or user_pretrained.get(uid)
    session_col = get_session_collection(uid)

    response = await run_in_threadpool(
        run_rag,
        uid,
        req.query,
        customer_type=domain,
        session_collection=session_col,
    )
    return {"response": response, "user_id": uid}


# ==========================================================
# VOICE CHAT — Browser audio → STT → RAG → TTS
# Voice name sent with each request for guaranteed correctness
# ==========================================================


@app.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    user_id: str = Form(default=""),
    voice: str = Form(default=""),          # sent with every request from frontend
    current_user: Optional[str] = Depends(optional_current_user),
    _: None = Depends(rate_limiter),
):
    uid = user_id or current_user or str(uuid.uuid4())

    suffix = ".webm" if (audio.filename and audio.filename.endswith(".webm")) else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await audio.read()
        f.write(content)
        temp_path = f.name

    transcript = ""
    try:
        transcript = await run_in_threadpool(transcribe_chunk, temp_path)
    except Exception as e:
        print(f"STT Error: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not transcript.strip():
        return {"transcript": "", "response": "", "audio": None, "user_id": uid}

    # Domain: use user's pretrained selection or default
    domain = user_pretrained.get(uid)
    session_col = get_session_collection(uid)

    response = await run_in_threadpool(
        run_rag,
        uid,
        transcript,
        customer_type=domain,
        session_collection=session_col,
    )

    # Voice: FormData field (sent every request) > user_voices dict > global default
    use_voice = voice or user_voices.get(uid) or None
    print(f"🎯 Voice for {uid[:8]}: form='{voice}', stored={user_voices.get(uid)}, using={use_voice}")

    # Set voice globally before calling speak (SDK uses global CURRENT_VOICE)
    if use_voice:
        set_voice(use_voice)

    audio_base64 = None
    try:
        audio_path = await run_in_threadpool(speak, response)
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as af:
                audio_base64 = base64.b64encode(af.read()).decode("utf-8")
            os.remove(audio_path)
    except Exception as e:
        print(f"TTS Error: {e}")

    return {
        "transcript": transcript,
        "response": response,
        "audio": audio_base64,
        "user_id": uid,
    }


# ==========================================================
# DOCUMENT UPLOAD — PDF → session Qdrant collection
# ==========================================================


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    user_id: str = Form(default=""),
    current_user: Optional[str] = Depends(optional_current_user),
    _: None = Depends(rate_limiter),
):
    uid = user_id or current_user or str(uuid.uuid4())
    session_collection = ensure_session_collection(uid)

    content = await file.read()
    text = ""

    filename = file.filename or "upload"
    if filename.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
    else:
        try:
            text = content.decode("utf-8").strip()
        except Exception:
            raise HTTPException(status_code=400, detail="Unable to decode file")

    if not text:
        raise HTTPException(status_code=400, detail="No text extracted from file")

    await run_in_threadpool(
        insert_document,
        text,
        {"source_file": filename, "user_id": uid},
        session_collection,
    )

    return {
        "status": "ok",
        "collection": session_collection,
        "chars": len(text),
        "user_id": uid,
    }


# ==========================================================
# END SESSION
# ==========================================================


@app.post("/end-session")
async def end_session(
    user_id: str = Form(default=""),
    current_user: Optional[str] = Depends(optional_current_user),
    _: None = Depends(rate_limiter),
):
    uid = user_id or current_user
    if uid and uid in active_sessions:
        col_name = active_sessions[uid]
        # Note: we don't delete Qdrant collection here to preserve perf,
        # just drop the reference. Session data expires naturally.
        del active_sessions[uid]
        return {"status": "session_ended", "collection": col_name}
    if user_id and user_id in active_sessions:
        col_name = active_sessions[user_id]
        # Note: we don't delete Qdrant collection here to preserve perf,
        # just drop the reference. Session data expires naturally.
        del active_sessions[user_id]
        return {"status": "session_ended", "collection": col_name}
    return {"status": "no_active_session"}


# ==========================================================
# SET VOICE
# ==========================================================


@app.post("/set-voice")
async def set_voice_endpoint(
    req: VoiceOption,
    current_user: Optional[str] = Depends(optional_current_user),
    _: None = Depends(rate_limiter),
):
    # Prefer per-user voice when logged in
    user_key = req.user_id or current_user or ""
    set_voice(req.voice)
    if user_key:
        user_voices[user_key] = req.voice
    print(f"✅ set-voice: user={user_key or 'global'}, voice={req.voice}")
    return {"status": "ok", "voice": req.voice, "user_id": user_key or None}


# ==========================================================
# SET PRE-TRAINED COLLECTION
# ==========================================================


@app.post("/set-collection")
async def set_collection_endpoint(
    req: CollectionOption,
    current_user: Optional[str] = Depends(optional_current_user),
    _: None = Depends(rate_limiter),
):
    if req.collection not in PRETRAINED_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown collection: {req.collection}")
    uid = req.user_id or current_user or "anon"
    user_pretrained[uid] = req.collection
    print(f"✅ set-collection: user={uid}, domain={req.collection}")
    return {"status": "ok", "collection": req.collection, "user_id": uid}


# ==========================================================
# AUTH — Signup / Login
# ==========================================================


@app.post("/signup")
async def signup(user: UserCreate, _: None = Depends(rate_limiter)):
    ok, msg = await run_in_threadpool(signup_user, user.username, user.password)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"message": msg}


@app.post("/login")
async def login(user: UserLogin, _: None = Depends(rate_limiter)):
    ok, result = await run_in_threadpool(login_user, user.username, user.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    return {"access_token": result}
