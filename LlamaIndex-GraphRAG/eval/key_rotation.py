"""
key_rotation.py — Same Groq API key rotation used in BI-GraphRAG/eval, copied
here so RAGAs scoring survives individual keys hitting their daily quota.

Reads keys from .groq_keys.txt in this project's root (one key per line) —
copy the file over from BI-GraphRAG or point KEYS_FILE elsewhere.
"""

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