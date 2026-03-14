"""Speech-to-text using OpenAI Whisper API."""

from __future__ import annotations

import io
import logging

import openai

import config

logger = logging.getLogger(__name__)


async def transcribe(wav_data: bytes, language: str | None = None) -> str:
    """Transcribe WAV audio data to text using OpenAI Whisper.

    Args:
        wav_data: Audio in WAV format (16-bit PCM).
        language: Optional ISO-639-1 language hint (e.g. "de", "en").

    Returns:
        Transcribed text string.
    """
    client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    audio_file = io.BytesIO(wav_data)
    audio_file.name = "audio.wav"

    kwargs: dict = {
        "model": config.WHISPER_MODEL,
        "file": audio_file,
        "response_format": "text",
    }
    if language:
        kwargs["language"] = language

    logger.info("Sending %d bytes to Whisper API (model=%s)", len(wav_data), config.WHISPER_MODEL)

    transcript = await client.audio.transcriptions.create(**kwargs)

    text = transcript.strip() if isinstance(transcript, str) else transcript.text.strip()
    logger.info("Whisper result: %s", text[:100])
    return text
