import uuid
import hashlib
import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from config import QDRANT_URL, QDRANT_API_KEY, resolve_collection_name
from embeddings import embed_text


# ==============================
# 🚀 QDRANT CLIENT INIT
# ==============================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False,
    timeout=30,
)


# ==============================
# 🧠 COLLECTION HELPERS
# ==============================


def _collection_exists(collection_name: str) -> bool:
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def create_collection(domain: str, recreate: bool = False):
    """
    Creates a collection for the given domain.
    OpenAI text-embedding-3-large → 3072 dimensions
    """
    collection_name = resolve_collection_name(domain)

    if recreate:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' recreated successfully.")
        return

    if _collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )

    print(f"Collection '{collection_name}' created successfully.")


def list_collections() -> list[str]:
    cols = client.get_collections()
    collections = getattr(cols, "collections", None) or []

    names = []
    for c in collections:
        name = getattr(c, "name", None)
        if name:
            names.append(name)

    return names


# ==============================
# 📥 INSERT DOCUMENT
# ==============================


def insert_document(text: str, metadata: dict, domain: str):
    """
    Inserts document into Qdrant with embedding + metadata.
    """

    vector = embed_text(text)
    collection_name = resolve_collection_name(domain)

    if not _collection_exists(collection_name):
        create_collection(domain)

    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()), vector=vector, payload={"text": text, **metadata}
            )
        ],
    )

    print(f"Document inserted successfully into '{collection_name}'.")


# ==============================
# ⚡ RETRIEVAL CACHE
# ==============================

RETRIEVAL_CACHE_FILE = "retrieval_cache.json"

retrieval_cache = {}
try:
    if os.path.exists(RETRIEVAL_CACHE_FILE):
        with open(RETRIEVAL_CACHE_FILE, "r") as f:
            retrieval_cache = json.load(f)
except Exception:
    retrieval_cache = {}


def _cache_key(query: str, domain: str):
    key_string = f"{query}:{domain}"
    return hashlib.sha256(key_string.encode()).hexdigest()


def _save_cache():
    try:
        with open(RETRIEVAL_CACHE_FILE, "w") as f:
            json.dump(retrieval_cache, f)
    except Exception:
        pass  # Ephemeral filesystem — cache write may fail on Render


# ==============================
# 🔍 SEARCH
# ==============================


def search(query: str, domain: str, limit: int = 5):
    """
    Searches Qdrant using query_points().
    Uses retrieval cache for faster responses.
    """

    cache_key = _cache_key(query, domain)

    # ======================
    # CACHE HIT
    # ======================

    if cache_key in retrieval_cache:
        print("⚡ Retrieval Cache HIT")
        return retrieval_cache[cache_key]

    print("🗄️ Retrieval Cache MISS → querying Qdrant")

    query_vector = embed_text(query)
    collection_name = resolve_collection_name(domain)

    if not _collection_exists(collection_name):
        return []

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
    )

    points = []

    if results and hasattr(results, "points"):
        for p in results.points:
            points.append({"id": str(p.id), "score": p.score, "payload": p.payload})

    # ======================
    # STORE CACHE
    # ======================

    retrieval_cache[cache_key] = points
    _save_cache()

    return points
