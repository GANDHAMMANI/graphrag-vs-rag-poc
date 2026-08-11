"""
config.py — Single source of truth for all configuration.

Uses pydantic-settings so every required variable is validated at import time.
If GROQ_API_KEY, NEO4J_URI, NEO4J_USERNAME, or NEO4J_PASSWORD are missing the
process exits immediately with a clear error — no silent None values scattered
across modules.

Install: pip install pydantic-settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Required ────────────────────────────────────────────────────────────
    groq_api_key: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str

    # ── Optional with sensible defaults ────────────────────────────────────
    extraction_model: str = "llama-3.3-70b-versatile"
    answer_model: str = "llama-3.1-8b-instant"
    embed_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_size: int = 1200
    chunk_overlap: int = 150

    # Max simultaneous Groq calls during batch extraction.
    # Groq free tier: ~30 req/min → keep at 5.
    # Paid tier: raise to 10–20.
    max_concurrent_extractions: int = 5

    # Cap on graph facts sent to the answer LLM. Beyond ~30 unique facts the
    # model gains no additional accuracy but burns tokens and increases latency.
    max_context_facts: int = 20

    # Cap on citation cards returned to the UI (always <= max_context_facts).
    max_citations: int = 6

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Module-level singleton — imported by every other module.
# Raises ValidationError at startup if any required var is missing.
settings = Settings()
