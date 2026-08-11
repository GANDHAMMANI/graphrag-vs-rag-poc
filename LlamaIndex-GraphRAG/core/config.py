"""
config.py — Single source of truth for configuration.

Same pattern as BI-GraphRAG/core/config.py: pydantic-settings validates all
required env vars at import time, process exits with a clear error if any
are missing.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Required ────────────────────────────────────────────────────────────
    groq_api_key: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"

    # ── Optional with sensible defaults ────────────────────────────────────
    # 8b-instant doesn't reliably follow the "output only JSON" instruction for
    # entity/relationship extraction — it sometimes writes example code instead
    # of extracting directly. 70b-versatile follows structured-output instructions
    # far more reliably; quota resets daily so this is fine on a fresh day.
    extraction_model: str = "llama-3.3-70b-versatile"
    answer_model: str = "llama-3.3-70b-versatile"
    embed_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 1200
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
