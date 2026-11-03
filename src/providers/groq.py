"""Groq (open-weight models such as gpt-oss), through the official groq SDK.

Groq uses the OpenAI-style chat format: tool calls carry an id, and each tool
result is a "tool" message pointing back at that id.
"""

from __future__ import annotations

import json
import os

import groq

from .base import backoff, new_call_id

DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqProvider:
    name = "groq"

    def __init__(self, model: str | None = None):
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set (put it in .env)")
        self.model = model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
        self.client = groq.Groq(api_key=key, timeout=60, max_retries=0)  # we retry ourselves

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model, messages=to_groq(system, history), tools=to_groq_tools(tools), temperature=0
        )
        return from_groq(response)

    @staticmethod
    def retry_delay(e: Exception, attempt: int) -> float | None:
        """Seconds to wait before retrying this error, or None if retrying won't help."""
        if isinstance(e, groq.RateLimitError):
            try:
                server_says = float(e.response.headers.get("retry-after") or 0)
            except ValueError:
                server_says = None
            return backoff(attempt, server_says)
        if isinstance(e, (groq.APIConnectionError, groq.InternalServerError)):
            return backoff(attempt, None)
        # The model sometimes writes a malformed tool call; asking again usually fixes it.
        if isinstance(e, groq.BadRequestError) and "tool_use_failed" in str(e):
            return backoff(attempt, None)
        return None

    @staticmethod
    def short_error(e: Exception) -> str:
        return str(e)[:200]


def to_groq_tools(specs: list[dict]) -> list[dict]:
    return [
        {"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
        for s in specs
    ]


def to_groq(system: str, history: list[dict]) -> list[dict]:
    """Our messages -> Groq chat messages."""
    messages = [{"role": "system", "content": system}]
    for m in history:
        if m["role"] == "user":
            messages.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("text") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                    for c in m["tool_calls"]
                ]
            messages.append(msg)
        else:
            messages.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": json.dumps(m["result"])})
    return messages


def from_groq(response) -> dict:
    """Groq response -> one assistant message in our format."""
    msg = response.choices[0].message
    calls = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}  # the tool will report the missing argument back to the model
        calls.append({"id": tc.id or new_call_id(), "name": tc.function.name, "args": args, "meta": {}})
    return {"role": "assistant", "text": (msg.content or "").strip(), "tool_calls": calls}
