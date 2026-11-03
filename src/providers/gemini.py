"""Google Gemini, through the official google-genai SDK."""

from __future__ import annotations

import base64
import json
import os

import httpx
from google import genai
from google.genai import errors, types

from .base import backoff, new_call_id

DEFAULT_MODEL = "gemini-3.5-flash-lite"


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> dict:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[to_gemini_tools(tools)],
            # We run the tools ourselves: never let the SDK call functions on its own.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self.client.models.generate_content(
            model=self.model, contents=to_gemini(history, self.model), config=config
        )
        return from_gemini(response)

    @staticmethod
    def retry_delay(e: Exception, attempt: int) -> float | None:
        """Seconds to wait before retrying this error, or None if retrying won't help."""
        if isinstance(e, httpx.TransportError):
            return backoff(attempt, None)
        if not isinstance(e, errors.APIError) or e.code not in (429, 500, 502, 503, 504):
            return None
        server_says = None
        for detail in (e.details or {}).get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):  # e.g. {"retryDelay": "18s"}
                server_says = float(detail.get("retryDelay", "0s").rstrip("s") or 0)
            for violation in detail.get("violations", []):
                if "PerDay" in violation.get("quotaId", ""):
                    return None  # the daily quota is used up: waiting a minute won't help
        return backoff(attempt, server_says)

    @staticmethod
    def short_error(e: Exception) -> str:
        if isinstance(e, errors.APIError):
            text = f"{e.code} {e.status}: {(e.message or '').splitlines()[0][:160]}"
            if e.code == 429:
                text += " (try /model groq, or another GEMINI_MODEL in .env)"
            return text
        return str(e)[:200]


def to_gemini_tools(specs: list[dict]) -> types.Tool:
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=s["name"], description=s["description"], parameters_json_schema=s["parameters"]
            )
            for s in specs
        ]
    )


def to_gemini(history: list[dict], model: str = DEFAULT_MODEL) -> list[types.Content]:
    """Our messages -> Gemini contents.

    Gemini has two roles: "user" and "model". Tool results go back as
    function_response parts in a "user" turn, matched to the call by name.
    Neighbouring messages with the same role are merged into one turn.
    """
    contents: list[types.Content] = []
    for m in history:
        if m["role"] == "user":
            role, parts = "user", [types.Part(text=m["text"])]
        elif m["role"] == "assistant":
            role, parts = "model", []
            if m.get("text"):
                parts.append(types.Part(text=m["text"]))
            for call in m.get("tool_calls") or []:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(name=call["name"], args=call["args"]),
                        thought_signature=_signature(call, model),
                    )
                )
            if not parts:
                parts = [types.Part(text="(no reply)")]
        else:  # tool result
            role = "user"
            parts = [types.Part.from_function_response(name=m["name"], response=m["result"])]

        if contents and contents[-1].role == role:
            contents[-1].parts.extend(parts)
        else:
            contents.append(types.Content(role=role, parts=parts))
    return contents


def _signature(call: dict, model: str) -> bytes | None:
    """Gemini's thinking models attach a 'thought signature' to function calls and
    want it back. We keep it in the call's meta. Calls made by another provider have
    none; Gemini 3 insists, so it gets the documented placeholder."""
    sig = (call.get("meta") or {}).get("gemini_thought_signature")
    if sig:
        return base64.b64decode(sig)
    if model.startswith("gemini-3"):
        return b"skip_thought_signature_validator"
    return None


def from_gemini(response: types.GenerateContentResponse) -> dict:
    """Gemini response -> one assistant message in our format."""
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None or not candidate.content.parts:
        reason = getattr(candidate, "finish_reason", None) or response.prompt_feedback
        return {"role": "assistant", "text": f"(The model returned no answer: {reason}.)", "tool_calls": []}

    text, calls = [], []
    for part in candidate.content.parts:
        if part.function_call:
            meta = {}
            if part.thought_signature:
                meta["gemini_thought_signature"] = base64.b64encode(part.thought_signature).decode()
            calls.append(
                {
                    "id": part.function_call.id or new_call_id(),
                    "name": part.function_call.name,
                    "args": json.loads(json.dumps(part.function_call.args or {})),
                    "meta": meta,
                }
            )
        elif part.text and not part.thought:
            text.append(part.text)
    return {"role": "assistant", "text": "".join(text).strip(), "tool_calls": calls}
