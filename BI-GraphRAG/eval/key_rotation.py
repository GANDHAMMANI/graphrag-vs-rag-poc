"""
key_rotation.py — Rotate across a pool of Groq API keys on rate-limit errors.

Loads keys from .groq_keys.txt (one per line, project root). RotatingChatGroq
wraps ChatGroq and, on a RateLimitError, advances to the next key and retries
the same request — up to len(keys) attempts — before giving up.

Used by eval/score.py so RAGAs judge calls survive individual keys hitting
their daily token quota.
"""

import itertools
import logging
import threading
from pathlib import Path

import groq
from langchain_groq import ChatGroq

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
    """Thread-safe round-robin pointer into a shared key pool."""

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


class RotatingChatGroq(ChatGroq):
    """
    ChatGroq that rotates through a KeyRotator's key pool on RateLimitError.

    On each generation call, tries the rotator's current key; if it hits a
    RateLimitError, advances the rotator and retries with the next key —
    up to one full lap of the pool — before finally raising.
    """

    _rotator: KeyRotator = None

    def __init__(self, rotator: KeyRotator, **kwargs):
        super().__init__(api_key=rotator.current(), **kwargs)
        object.__setattr__(self, "_rotator", rotator)

    def _set_key(self, key: str):
        self.client._client.api_key = key
        self.async_client._client.api_key = key

    def _generate(self, *args, **kwargs):
        rotator = self._rotator
        last_exc = None
        for _ in range(len(rotator.keys)):
            self._set_key(rotator.current())
            try:
                return super()._generate(*args, **kwargs)
            except groq.RateLimitError as e:
                last_exc = e
                rotator.advance()
        raise last_exc

    async def _agenerate(self, *args, **kwargs):
        rotator = self._rotator
        last_exc = None
        for _ in range(len(rotator.keys)):
            self._set_key(rotator.current())
            try:
                return await super()._agenerate(*args, **kwargs)
            except groq.RateLimitError as e:
                last_exc = e
                rotator.advance()
        raise last_exc
