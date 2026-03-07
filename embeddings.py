import json
import os
import hashlib
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

CACHE_FILE = "embedding_cache.json"

# Load cache if exists (safe for ephemeral filesystems)
embedding_cache = {}
try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            embedding_cache = json.load(f)
except Exception:
    embedding_cache = {}


def _hash_text(text: str) -> str:
    """Create a stable hash for the text"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_text(text: str):
    key = _hash_text(text)

    # 1️⃣ Check cache first
    if key in embedding_cache:
        return embedding_cache[key]

    # 2️⃣ Call OpenAI only if not cached
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)

    embedding = response.data[0].embedding

    # 3️⃣ Store in cache
    embedding_cache[key] = embedding

    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(embedding_cache, f)
    except Exception:
        pass  # Ephemeral filesystem — cache write may fail on Render

    return embedding
