## Backend async & security update

### Overview

- Converted key FastAPI endpoints to use `async def` and wrapped blocking I/O in `run_in_threadpool` to improve concurrency under load on Railway.
- Added a lightweight in-process rate limiter and tightened CORS configuration to improve API safety.
- Kept existing auth model (JWT + Supabase/JSON fallback) but wired auth awareness into more endpoints via an optional current-user helper.

### Async / performance-focused changes

- `app.py`
  - Imported `asyncio`, `time`, `Request`, `Depends`, `Header`, and `run_in_threadpool`.
  - Updated endpoints to be asynchronous:
    - `/health` now `async def health()` and calls `list_collections` via `run_in_threadpool`.
    - `/chat` now `async def chat(...)` and calls `run_rag` via `run_in_threadpool`.
    - `/voice-chat` remains `async`, but now:
      - Calls `transcribe_chunk`, `run_rag`, and `speak` via `run_in_threadpool`.
    - `/upload` is `async` and now calls `insert_document` via `run_in_threadpool`.
    - `/end-session`, `/set-voice`, `/set-collection`, `/signup`, `/login` are now `async` and call their blocking helpers through `run_in_threadpool` where appropriate.

### API security & rate limiting

- **CORS tightening**
  - Replaced `allow_origins=["*"]` with configurable allowed origins:
    - Defaults to `https://natural-vocl-ai.vercel.app` and `http://localhost:3000`.
    - Can be overridden via `FRONTEND_ORIGINS` env variable (comma-separated).

- **JWT-aware helpers**
  - Added `optional_current_user(...)` in `app.py`:
    - Reads `Authorization: Bearer <token>` header if present.
    - Uses `auth.decode_token` to decode JWT and returns the username (`sub`) or `None` if invalid/missing.
    - Used in several endpoints to associate actions with a user when a token is provided.

- **Rate limiting**
  - Added configuration:
    - `API_RATE_LIMIT` (default: 60 requests).
    - `API_RATE_WINDOW` (default: 60 seconds).
  - Implemented `rate_limiter` dependency:
    - Uses an in-memory sliding window per-identifier.
    - Identifier is `user:<username>` when authenticated, otherwise `ip:<client_ip>`.
    - Returns HTTP 429 when a client exceeds the configured rate.
  - Applied `rate_limiter` to:
    - `/health`
    - `/chat`
    - `/voice-chat`
    - `/upload`
    - `/end-session`
    - `/set-voice`
    - `/set-collection`
    - `/signup`
    - `/login`

### Behavior notes

- Existing JWT-based login/signup flows are unchanged, but authenticated clients now:
  - Are identified for rate limiting by username instead of IP.
  - Influence per-user session behavior in `/set-voice`, `/set-collection`, and `/end-session` when `user_id` is omitted.
- The rate limiter is in-process and per-instance:
  - Each Railway dyno/container enforces its own limits.
  - This is sufficient for basic abuse protection and latency smoothing; for strict global limits, a shared store (e.g., Redis) would be required.

