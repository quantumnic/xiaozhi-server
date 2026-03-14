"""Protocol message types for the xiaozhi WebSocket protocol."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    HELLO = "hello"
    LISTEN = "listen"
    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    ABORT = "abort"


class ListenState(str, Enum):
    START = "start"
    STOP = "stop"


class ListenMode(str, Enum):
    AUTO_STOP = "auto_stop"
    MANUAL = "manual"


class TTSState(str, Enum):
    SENTENCE_START = "sentence_start"
    SENTENCE_END = "sentence_end"
    STOP = "stop"


@dataclass
class AudioParams:
    format: str = "opus"
    sample_rate: int = 16000
    channels: int = 1
    frame_duration: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_duration": self.frame_duration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioParams:
        return cls(
            format=data.get("format", "opus"),
            sample_rate=data.get("sample_rate", 16000),
            channels=data.get("channels", 1),
            frame_duration=data.get("frame_duration", 60),
        )


# --- Incoming messages (ESP32 → Server) ---


@dataclass
class HelloMessage:
    version: int = 1
    transport: str = "websocket"
    audio_params: AudioParams = field(default_factory=AudioParams)
    features: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HelloMessage:
        return cls(
            version=data.get("version", 1),
            transport=data.get("transport", "websocket"),
            audio_params=AudioParams.from_dict(data.get("audio_params", {})),
            features=data.get("features", {}),
        )


@dataclass
class ListenMessage:
    state: ListenState
    mode: ListenMode = ListenMode.AUTO_STOP

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ListenMessage:
        return cls(
            state=ListenState(data["state"]),
            mode=ListenMode(data.get("mode", "auto_stop")),
        )


@dataclass
class AbortMessage:
    reason: str = "wake_word_detected"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AbortMessage:
        return cls(reason=data.get("reason", "unknown"))


# --- Outgoing messages (Server → ESP32) ---


def hello_response(session_id: str | None = None) -> str:
    """Build server hello response JSON."""
    sid = session_id or uuid.uuid4().hex[:16]
    msg = {
        "type": MessageType.HELLO.value,
        "transport": "websocket",
        "session_id": sid,
        "audio_params": AudioParams(
            format="opus",
            sample_rate=24000,
            channels=1,
            frame_duration=60,
        ).to_dict(),
    }
    return json.dumps(msg)


def stt_response(text: str) -> str:
    return json.dumps({"type": MessageType.STT.value, "text": text})


def llm_response(text: str, emotion: str = "neutral") -> str:
    return json.dumps({
        "type": MessageType.LLM.value,
        "text": text,
        "emotion": emotion,
    })


def tts_message(state: TTSState, text: str | None = None) -> str:
    msg: dict[str, Any] = {"type": MessageType.TTS.value, "state": state.value}
    if text is not None:
        msg["text"] = text
    return json.dumps(msg)


def parse_message(raw: str) -> dict[str, Any]:
    """Parse a JSON message and return as dict. Raises ValueError on invalid JSON."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON message: {e}") from e
    if "type" not in data:
        raise ValueError("Message missing 'type' field")
    return data
