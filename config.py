import os

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# Logical domains to Qdrant collection names.
# Add new domains here and reuse them across the codebase.
COLLECTIONS = {
    "ecommerce": "ecommerce",
    "car_booking": "car_booking",
    "restaurant_support": "restaurant_support",
}

# Default customer domain if none is explicitly provided.
DEFAULT_CUSTOMER_DOMAIN = "car_booking"

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1-nano"


def normalize_domain(domain: str) -> str:
    """
    Normalize a user-provided domain/customer_type into a safe collection key.
    Keeps it simple: lowercase + spaces to underscores.
    """
    return domain.strip().lower().replace(" ", "_")


def resolve_collection_name(domain: str | None) -> str:
    """
    Map a logical domain (e.g. 'ecommerce', 'car_booking') to a Qdrant collection.
    Falls back to DEFAULT_CUSTOMER_DOMAIN when domain is missing or unknown.
    """
    if not domain:
        domain = DEFAULT_CUSTOMER_DOMAIN

    domain = normalize_domain(domain)

    # If the domain isn't in the static mapping, treat the domain itself as the
    # collection name so runtime-created collections work without code changes.
    return COLLECTIONS.get(domain, domain)
