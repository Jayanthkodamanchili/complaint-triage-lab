# NEW CODE START
"""One place that knows how to reach every LLM provider (Week 2 Day 1 pattern)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None  # None = OpenAI's own default address
    key_env: str | None  # the key's name in .env; None = no key needed
    model: str


PROVIDERS: dict[str, Provider] = {
    "ollama": Provider("ollama", "http://localhost:11434/v1", None, "llama3.2"),
    "gemini": Provider(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GOOGLE_API_KEY",
        "gemini-flash-lite-latest",
    ),
    "openai": Provider("openai", None, "OPENAI_API_KEY", "gpt-6-luna"),
    "sol": Provider("openai", None, "OPENAI_API_KEY", "gpt-6.2-sol"),  # premium test
    "anthropic": Provider(
        "anthropic",
        "https://api.anthropic.com/v1/",
        "ANTHROPIC_API_KEY",
        "claude-haiku-5-5",
    ),
}


def get_client(provider: Provider) -> OpenAI:
    """Build an OpenAI-style client for any provider."""
    if provider.key_env is None:
        api_key = "ollama"  # Ollama ignores the key, but the client needs some text
    else:
        api_key = os.getenv(provider.key_env)
        if not api_key:
            raise RuntimeError(f"{provider.key_env} is missing from .env")

    headers: dict[str, str] = {}
    if provider.name == "anthropic":
        # Anthropic's own model list expects the key in this header
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

    return OpenAI(base_url=provider.base_url, api_key=api_key, default_headers=headers)


# NEW CODE END
