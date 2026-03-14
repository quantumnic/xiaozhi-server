"""Text-to-speech using OpenAI TTS API with sentence chunking."""

from __future__ import annotations

import logging
import re

import openai

import config
from xiaozhi.audio import OpusEncoder, split_pcm_frames

logger = logging.getLogger(__name__)

# Sentence boundary pattern
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at sentence boundaries."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


async def synthesize(text: str) -> bytes:
    """Synthesize text to raw PCM audio (24kHz, mono, 16-bit).

    Returns raw PCM bytes.
    """
    client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    logger.info("TTS request: voice=%s, model=%s, text=%s", config.TTS_VOICE, config.TTS_MODEL, text[:60])

    response = await client.audio.speech.create(
        model=config.TTS_MODEL,
        voice=config.TTS_VOICE,
        input=text,
        response_format="pcm",  # Raw 24kHz 16-bit mono PCM
    )

    # response.content is bytes (raw PCM at 24kHz)
    pcm_data = response.content
    logger.info("TTS response: %d bytes PCM", len(pcm_data))
    return pcm_data


async def synthesize_to_opus_frames(
    text: str,
    sample_rate: int = 24000,
    frame_duration_ms: int = 60,
) -> list[bytes]:
    """Synthesize text and return as Opus frames ready for WebSocket."""
    pcm_data = await synthesize(text)

    # Split PCM into frames
    pcm_frames = split_pcm_frames(pcm_data, frame_duration_ms, sample_rate)

    # Encode each frame to Opus
    encoder = OpusEncoder(sample_rate=sample_rate)
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    opus_frames = []
    for pcm_frame in pcm_frames:
        opus_frame = encoder.encode(pcm_frame, frame_size)
        opus_frames.append(opus_frame)

    logger.info("Encoded %d Opus frames from text: %s", len(opus_frames), text[:40])
    return opus_frames


class SentenceStreamer:
    """Accumulates streamed LLM text and yields complete sentences."""

    def __init__(self) -> None:
        self._buffer: str = ""

    def add(self, chunk: str) -> list[str]:
        """Add a text chunk. Returns list of complete sentences (may be empty)."""
        self._buffer += chunk
        sentences: list[str] = []
        while True:
            match = _SENTENCE_RE.search(self._buffer)
            if match:
                sentence = self._buffer[: match.start() + 1].strip()
                self._buffer = self._buffer[match.end() :]
                if sentence:
                    sentences.append(sentence)
            else:
                break
        return sentences

    def flush(self) -> str | None:
        """Flush remaining buffer as final sentence."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None
