"""Opus encode/decode and PCM utilities.

opuslib is optional — if unavailable, falls back to raw PCM with a warning.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# Try to import opuslib; graceful fallback if unavailable
_opus_available = False
try:
    import opuslib  # type: ignore[import-untyped]

    _opus_available = True
except Exception:  # ImportError, OSError, or opuslib "Could not find Opus library"
    logger.warning(
        "opuslib not available — Opus codec disabled. "
        "Audio will be treated as raw PCM. Install opuslib for proper Opus support."
    )
    opuslib = None  # type: ignore[assignment]

if TYPE_CHECKING:
    pass


def is_opus_available() -> bool:
    """Check whether Opus codec is available."""
    return _opus_available


class OpusDecoder:
    """Decode Opus frames to PCM (16-bit signed int)."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._decoder = None
        if _opus_available:
            self._decoder = opuslib.Decoder(sample_rate, channels)

    def decode(self, opus_data: bytes, frame_size: int = 960) -> bytes:
        """Decode a single Opus frame. Returns raw PCM bytes (16-bit LE)."""
        if self._decoder is not None:
            return self._decoder.decode(opus_data, frame_size)
        # Fallback: treat as raw PCM
        return opus_data


class OpusEncoder:
    """Encode PCM (16-bit signed int) to Opus frames."""

    def __init__(self, sample_rate: int = 24000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._encoder = None
        if _opus_available:
            self._encoder = opuslib.Encoder(sample_rate, channels, opuslib.APPLICATION_AUDIO)

    def encode(self, pcm_data: bytes, frame_size: int = 1440) -> bytes:
        """Encode a PCM frame to Opus. Returns Opus bytes."""
        if self._encoder is not None:
            return self._encoder.encode(pcm_data, frame_size)
        # Fallback: return raw PCM
        return pcm_data


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Convert raw PCM (16-bit LE) to WAV format in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def resample(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM data (16-bit LE) from one sample rate to another."""
    if from_rate == to_rate:
        return pcm_data
    from scipy.signal import resample as scipy_resample

    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float64)
    num_samples = int(len(samples) * to_rate / from_rate)
    resampled = scipy_resample(samples, num_samples)
    return resampled.astype(np.int16).tobytes()


def split_pcm_frames(pcm_data: bytes, frame_duration_ms: int, sample_rate: int) -> list[bytes]:
    """Split PCM data into fixed-size frames."""
    frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame
    bytes_per_frame = frame_size * 2  # 16-bit = 2 bytes per sample
    frames = []
    for i in range(0, len(pcm_data), bytes_per_frame):
        frame = pcm_data[i : i + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            # Pad last frame with silence
            frame += b"\x00" * (bytes_per_frame - len(frame))
        frames.append(frame)
    return frames


class AudioBuffer:
    """Accumulate Opus frames and decode to PCM."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, frame_duration: int = 60):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration = frame_duration
        self.frames: list[bytes] = []
        self.decoder = OpusDecoder(sample_rate, channels)
        # samples per frame = sample_rate * frame_duration_ms / 1000
        self._frame_size = int(sample_rate * frame_duration / 1000)

    def add_frame(self, data: bytes) -> None:
        """Add an Opus frame."""
        self.frames.append(data)

    def clear(self) -> None:
        """Clear accumulated frames."""
        self.frames.clear()

    def get_pcm(self) -> bytes:
        """Decode all accumulated frames to PCM."""
        pcm_chunks = []
        for frame in self.frames:
            pcm = self.decoder.decode(frame, self._frame_size)
            pcm_chunks.append(pcm)
        return b"".join(pcm_chunks)

    def get_wav(self) -> bytes:
        """Decode all frames and return as WAV."""
        pcm = self.get_pcm()
        return pcm_to_wav(pcm, self.sample_rate, self.channels)

    @property
    def frame_count(self) -> int:
        return len(self.frames)
