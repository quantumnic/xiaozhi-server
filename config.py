"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present
load_dotenv(Path(__file__).parent / ".env")


def _get(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise ValueError(f"Required environment variable {key} is not set")
    return val


# --- API Keys ---
OPENAI_API_KEY: str | None = _get("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None = _get("ANTHROPIC_API_KEY")

# --- LLM ---
LLM_PROVIDER: str = _get("LLM_PROVIDER", "openai")  # "openai" | "anthropic"
LLM_MODEL: str | None = _get("LLM_MODEL")  # None → auto-select based on provider

# --- STT ---
WHISPER_MODEL: str = _get("WHISPER_MODEL", "whisper-1")

# --- TTS ---
TTS_VOICE: str = _get("TTS_VOICE", "alloy")
TTS_MODEL: str = _get("TTS_MODEL", "tts-1")

# --- Server ---
SERVER_HOST: str = _get("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = int(_get("SERVER_PORT", "8765"))

# --- Auth ---
AUTH_TOKEN: str = _get("AUTH_TOKEN", "")

# --- Prompts ---
SYSTEM_PROMPT: str = _get(
    "SYSTEM_PROMPT",
    "Du bist ein hilfreicher Assistent. Antworte kurz und prägnant.",
)

# --- Conversation ---
MAX_HISTORY: int = int(_get("MAX_HISTORY", "10"))


def get_llm_model() -> str:
    """Return the configured LLM model or a sensible default."""
    if LLM_MODEL:
        return LLM_MODEL
    return "gpt-4o" if LLM_PROVIDER == "openai" else "claude-sonnet-4-20250514"


def validate() -> list[str]:
    """Validate configuration. Returns list of errors (empty = OK)."""
    errors: list[str] = []
    if LLM_PROVIDER not in ("openai", "anthropic"):
        errors.append(f"LLM_PROVIDER must be 'openai' or 'anthropic', got '{LLM_PROVIDER}'")
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
    # Whisper & TTS always need OpenAI
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required for Whisper STT and TTS")
    return errors
