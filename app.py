# ==========================================================
# 🚀 CUSTOMER SERVICE AI — FASTAPI BRIDGE SERVER
# Combines repo features (collections API, Tavily, domain routing)
# with frontend bridge (voice-chat, upload, auth, session mgmt)
# ==========================================================

from dotenv import load_dotenv
load_dotenv()  # Load .env file for local development

import tempfile
import os
import base64
import uuid
import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber

from rag_pipeline import run_rag
from sarvam_streaming_stt import transcribe_chunk
from tts import speak, set_voice
from vector_store import (
    insert_document,
    create_collection,
    list_collections,
)
from config import COLLECTIONS, resolve_collection_name
from collections_api import router as collections_router
from auth import signup_user, login_user

app = FastAPI(title="Customer Service AI")
app.include_router(collections_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
def health():
    try:
        collections = list_collections()
        return {"status": "ok", "qdrant_status": "connected", "collections": collections}
    except Exception as e:
        return {"status": "ok", "qdrant_status": f"error: {str(e)}", "collections": []}


# ==========================================================
# TEXT CHAT
# ==========================================================


@app.post("/chat")
def chat(req: ChatRequest):
    uid = req.user_id or str(uuid.uuid4())

    # domain: explicit customer_type > user's pretrained selection > default
    domain = req.customer_type or user_pretrained.get(uid)
    session_col = get_session_collection(uid)

    response = run_rag(uid, req.query, customer_type=domain, session_collection=session_col)
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
):
    uid = user_id or str(uuid.uuid4())

    suffix = ".webm" if (audio.filename and audio.filename.endswith(".webm")) else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await audio.read()
        f.write(content)
        temp_path = f.name

    transcript = ""
    try:
        transcript = transcribe_chunk(temp_path)
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

    response = run_rag(uid, transcript, customer_type=domain, session_collection=session_col)

    # Voice: FormData field (sent every request) > user_voices dict > global default
    use_voice = voice or user_voices.get(uid) or None
    print(f"🎯 Voice for {uid[:8]}: form='{voice}', stored={user_voices.get(uid)}, using={use_voice}")

    # Set voice globally before calling speak (SDK uses global CURRENT_VOICE)
    if use_voice:
        set_voice(use_voice)

    audio_base64 = None
    try:
        audio_path = speak(response)
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
):
    uid = user_id or str(uuid.uuid4())
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

    insert_document(
        text=text,
        metadata={"source_file": filename, "user_id": uid},
        domain=session_collection,
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
def end_session(user_id: str = Form(default="")):
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
def set_voice_endpoint(req: VoiceOption):
    set_voice(req.voice)
    if req.user_id:
        user_voices[req.user_id] = req.voice
    print(f"✅ set-voice: user={req.user_id or 'global'}, voice={req.voice}")
    return {"status": "ok", "voice": req.voice}


# ==========================================================
# SET PRE-TRAINED COLLECTION
# ==========================================================


@app.post("/set-collection")
def set_collection_endpoint(req: CollectionOption):
    if req.collection not in PRETRAINED_COLLECTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown collection: {req.collection}")
    uid = req.user_id or "anon"
    user_pretrained[uid] = req.collection
    print(f"✅ set-collection: user={uid}, domain={req.collection}")
    return {"status": "ok", "collection": req.collection, "user_id": uid}


# ==========================================================
# AUTH — Signup / Login
# ==========================================================


@app.post("/signup")
def signup(user: UserCreate):
    ok, msg = signup_user(user.username, user.password)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"message": msg}


@app.post("/login")
def login(user: UserLogin):
    ok, result = login_user(user.username, user.password)
    if not ok:
        raise HTTPException(status_code=401, detail=result)
    return {"access_token": result}
