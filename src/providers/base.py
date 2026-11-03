"""Small helpers shared by the providers."""

from __future__ import annotations

import uuid

MAX_RETRY_WAIT = 60.0  # seconds


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def backoff(attempt: int, server_says: float | None) -> float:
    """How long to wait before retrying: what the server asked for, else 2s, 4s, 8s…"""
    delay = server_says + 1 if server_says else 2.0 * 2**attempt
    return min(delay, MAX_RETRY_WAIT)
