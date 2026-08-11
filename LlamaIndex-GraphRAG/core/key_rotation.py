"""
key_rotation.py — Rotate across a pool of Groq API keys on rate-limit errors,
for LlamaIndex's Groq LLM class (used during ingestion — extraction and
community summarization can burn through a single key's daily quota on a
large dataset).

Loads keys from .groq_keys.txt in the project root (one per line).

Note: llama_index.llms.groq.Groq is built on LlamaIndex's OpenAI-compatible
base client, so rate-limit errors surface as openai.RateLimitError here —
NOT groq.RateLimitError (that's only for the separate `groq` SDK, used by
eval/key_rotation.py's ChatGroq-based rotator instead).
"""

import logging
import threading
from pathlib import Path
from typing import Any, Sequence

import openai
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.llms.groq import Groq

logger = logging.getLogger(__name__)

KEYS_FILE = Path(__file__).parent.parent / ".groq_keys.txt"


def load_keys() -> list[str]:
    if not KEYS_FILE.exists():
        raise FileNotFoundError(f"Key pool file not found: {KEYS_FILE}")
    keys = [line.strip() for line in KEYS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not keys:
        raise ValueError(f"No keys found in {KEYS_FILE}")
    return keys


class KeyRotator:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self._lock = threading.Lock()
        self._idx = 0

    def current(self) -> str:
        with self._lock:
            return self.keys[self._idx % len(self.keys)]

    def advance(self) -> str:
        with self._lock:
            self._idx += 1
            key = self.keys[self._idx % len(self.keys)]
        logger.info("Rotated to key #%d/%d", (self._idx % len(self.keys)) + 1, len(self.keys))
        return key


class RotatingGroq(Groq):
    """Groq LLM that rotates through a KeyRotator's pool on RateLimitError.
    Requires reuse_client=False so _get_client() rebuilds from self.api_key,
    and max_retries=0 so LlamaIndex's own internal retry/backoff never runs —
    otherwise it retries the SAME key for minutes before our rotation ever
    gets a chance (daily token-limit errors don't get better by waiting)."""

    _rotator: KeyRotator = None

    def __init__(self, rotator: KeyRotator, **kwargs):
        kwargs["reuse_client"] = False
        kwargs["max_retries"] = 0
        super().__init__(api_key=rotator.current(), **kwargs)
        object.__setattr__(self, "_rotator", rotator)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        rotator = self._rotator
        last_exc = None
        for _ in range(len(rotator.keys)):
            self.api_key = rotator.current()
            try:
                return super().chat(messages, **kwargs)
            except openai.RateLimitError as e:
                last_exc = e
                logger.info("Rate limit hit, rotating key immediately (no retry on same key)")
                rotator.advance()
        raise last_exc

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        rotator = self._rotator
        last_exc = None
        for _ in range(len(rotator.keys)):
            self.api_key = rotator.current()
            try:
                return await super().achat(messages, **kwargs)
            except openai.RateLimitError as e:
                last_exc = e
                logger.info("Rate limit hit, rotating key immediately (no retry on same key)")
                rotator.advance()
        raise last_exc