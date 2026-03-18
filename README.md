## Customer Service AI Backend (Ministros)

This is the FastAPI backend that powers the MINISTROS / NaturalVoclAI demo.  
It exposes text and voice chat endpoints backed by:

- OpenAI `gpt-4.1-nano` for language reasoning
- Qdrant for vector search and RAG
- Optional Supabase (or JSON fallback) for user auth
- Sarvam AI for text‑to‑speech and streaming STT

The repository is deployed on Railway and consumed by the `NaturalVoclAI` frontend.

### Tech stack

- **Python / FastAPI**
- **Qdrant** for vector search (`vector_store.py`)
- **OpenAI** for LLM calls (`llm.py`, `rag_pipeline.py`)
- **Sarvam** for STT + TTS (`sarvam_streaming_stt.py`, `tts.py`)
- **JWT auth** with Supabase or JSON fallback (`auth.py`)

### Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env` file (or configure Railway variables) with at least:

```bash
OPENAI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
SARVAM_API_KEY=...
TAVILY_API_KEY=...          # optional, for web search fallback

JWT_SECRET=change-me
SUPABASE_URL=...           # optional
SUPABASE_KEY=...           # optional

FRONTEND_ORIGINS=https://natural-vocl-ai.vercel.app,http://localhost:3000
API_RATE_LIMIT=60
API_RATE_WINDOW=60
```

### Running locally

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Key endpoints:

- `POST /chat` – text RAG chat
- `POST /voice-chat` – voice → STT → RAG → TTS
- `POST /upload` – upload PDFs / text into a per‑session Qdrant collection
- `POST /set-voice` – select assistant voice
- `POST /set-collection` – switch between pre‑trained domains
- `POST /signup`, `POST /login` – auth; returns a JWT used by the frontend

### Notes

- Rate limiting and CORS are controlled entirely by environment variables to make Railway / production safer.
- Conversation history is stored in Redis when `REDIS_URL` is provided, otherwise in memory.
- The backend is written to be async‑friendly: blocking I/O is wrapped via `run_in_threadpool` for better concurrency.

