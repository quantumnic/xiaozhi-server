"""Tests for protocol message types."""

import json
import pytest

from xiaozhi.protocol import (
    AudioParams,
    HelloMessage,
    ListenMessage,
    ListenState,
    ListenMode,
    AbortMessage,
    TTSState,
    hello_response,
    stt_response,
    llm_response,
    tts_message,
    parse_message,
)


class TestAudioParams:
    def test_defaults(self):
        p = AudioParams()
        assert p.format == "opus"
        assert p.sample_rate == 16000
        assert p.channels == 1
        assert p.frame_duration == 60

    def test_to_dict(self):
        p = AudioParams(format="opus", sample_rate=24000, channels=1, frame_duration=60)
        d = p.to_dict()
        assert d["sample_rate"] == 24000

    def test_from_dict(self):
        d = {"format": "opus", "sample_rate": 48000, "channels": 2, "frame_duration": 20}
        p = AudioParams.from_dict(d)
        assert p.sample_rate == 48000
        assert p.channels == 2

    def test_from_dict_defaults(self):
        p = AudioParams.from_dict({})
        assert p.format == "opus"
        assert p.sample_rate == 16000


class TestHelloMessage:
    def test_from_dict_full(self):
        data = {
            "type": "hello",
            "version": 1,
            "features": {"mcp": True},
            "transport": "websocket",
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60,
            },
        }
        hello = HelloMessage.from_dict(data)
        assert hello.version == 1
        assert hello.features == {"mcp": True}
        assert hello.audio_params.sample_rate == 16000

    def test_from_dict_minimal(self):
        hello = HelloMessage.from_dict({"type": "hello"})
        assert hello.version == 1
        assert hello.transport == "websocket"


class TestListenMessage:
    def test_start(self):
        msg = ListenMessage.from_dict({"state": "start", "mode": "auto_stop"})
        assert msg.state == ListenState.START
        assert msg.mode == ListenMode.AUTO_STOP

    def test_stop(self):
        msg = ListenMessage.from_dict({"state": "stop"})
        assert msg.state == ListenState.STOP

    def test_invalid_state(self):
        with pytest.raises(ValueError):
            ListenMessage.from_dict({"state": "invalid"})


class TestAbortMessage:
    def test_from_dict(self):
        msg = AbortMessage.from_dict({"reason": "wake_word_detected"})
        assert msg.reason == "wake_word_detected"

    def test_default_reason(self):
        msg = AbortMessage.from_dict({})
        assert msg.reason == "unknown"


class TestHelloResponse:
    def test_structure(self):
        resp = json.loads(hello_response("test-session"))
        assert resp["type"] == "hello"
        assert resp["transport"] == "websocket"
        assert resp["session_id"] == "test-session"
        assert resp["audio_params"]["sample_rate"] == 24000
        assert resp["audio_params"]["format"] == "opus"

    def test_auto_session_id(self):
        resp = json.loads(hello_response())
        assert len(resp["session_id"]) == 16


class TestSttResponse:
    def test_structure(self):
        resp = json.loads(stt_response("hello world"))
        assert resp["type"] == "stt"
        assert resp["text"] == "hello world"


class TestLlmResponse:
    def test_structure(self):
        resp = json.loads(llm_response("answer", "happy"))
        assert resp["type"] == "llm"
        assert resp["text"] == "answer"
        assert resp["emotion"] == "happy"

    def test_default_emotion(self):
        resp = json.loads(llm_response("test"))
        assert resp["emotion"] == "neutral"


class TestTtsMessage:
    def test_sentence_start(self):
        resp = json.loads(tts_message(TTSState.SENTENCE_START, "Hi there"))
        assert resp["type"] == "tts"
        assert resp["state"] == "sentence_start"
        assert resp["text"] == "Hi there"

    def test_stop_no_text(self):
        resp = json.loads(tts_message(TTSState.STOP))
        assert resp["state"] == "stop"
        assert "text" not in resp


class TestParseMessage:
    def test_valid(self):
        data = parse_message('{"type":"hello","version":1}')
        assert data["type"] == "hello"

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_message("not json")

    def test_missing_type(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            parse_message('{"version":1}')
