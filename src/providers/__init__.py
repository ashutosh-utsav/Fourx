"""Model providers. Each one has `name`, `model` and

    complete(system, history, tools) -> {"role": "assistant", "text": str, "tool_calls": [...]}

`history` is our own message format (see src/store.py and README):
    {"role": "user", "text": ...}
    {"role": "assistant", "text": ..., "tool_calls": [{"id", "name", "args", "meta"}]}
    {"role": "tool", "tool_call_id": ..., "name": ..., "result": {...}}
Each provider translates that into its own SDK's format on every call, so the
same history works with either one.
"""

PROVIDERS = ("gemini", "groq")


def get_provider(name: str):
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider()
    if name == "groq":
        from .groq import GroqProvider

        return GroqProvider()
    raise ValueError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")
