"""WebSocket session handler for xiaozhi protocol."""

from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum

from starlette.websockets import WebSocket, WebSocketDisconnect

import config
from xiaozhi.audio import AudioBuffer
from xiaozhi.llm import ConversationHistory, stream_response
from xiaozhi.protocol import (
    AbortMessage,
    HelloMessage,
    ListenMessage,
    ListenState,
    MessageType,
    TTSState,
    hello_response,
    llm_response,
    parse_message,
    stt_response,
    tts_message,
)
from xiaozhi.stt import transcribe
from xiaozhi.tts import SentenceStreamer, synthesize_to_opus_frames

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class XiaozhiSession:
    """Handles a single WebSocket connection from an ESP32 device."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.session_id = uuid.uuid4().hex[:16]
        self.state = SessionState.IDLE
        self.history = ConversationHistory(max_messages=config.MAX_HISTORY)
        self.audio_buffer = AudioBuffer(sample_rate=16000, channels=1, frame_duration=60)
        self._tts_task: asyncio.Task | None = None
        self._abort_event = asyncio.Event()
        self._device_id: str | None = None
        self._client_id: str | None = None

    async def handle(self) -> None:
        """Main session loop."""
        await self.websocket.accept()
        logger.info("WebSocket connected: session=%s", self.session_id)

        try:
            # Wait for hello
            first_msg = await self.websocket.receive_text()
            data = parse_message(first_msg)
            if data.get("type") != MessageType.HELLO.value:
                logger.warning("Expected hello, got: %s", data.get("type"))
                await self.websocket.close(1002, "Expected hello message")
                return

            hello = HelloMessage.from_dict(data)
            logger.info("Hello from device: version=%d, features=%s", hello.version, hello.features)

            # Update audio buffer with client's params
            self.audio_buffer = AudioBuffer(
                sample_rate=hello.audio_params.sample_rate,
                channels=hello.audio_params.channels,
                frame_duration=hello.audio_params.frame_duration,
            )

            # Send hello response
            await self.websocket.send_text(hello_response(self.session_id))

            # Main message loop
            while True:
                message = await self.websocket.receive()

                if "text" in message:
                    await self._handle_text(message["text"])
                elif "bytes" in message:
                    await self._handle_binary(message["bytes"])

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: session=%s", self.session_id)
        except Exception as e:
            logger.exception("Session error: %s", e)
        finally:
            await self._cancel_tts()
            logger.info("Session ended: %s", self.session_id)

    async def _handle_text(self, raw: str) -> None:
        """Handle a text (JSON) message."""
        try:
            data = parse_message(raw)
        except ValueError as e:
            logger.warning("Invalid message: %s", e)
            return

        msg_type = data.get("type")

        if msg_type == MessageType.LISTEN.value:
            listen = ListenMessage.from_dict(data)
            if listen.state == ListenState.START:
                await self._on_listen_start(listen)
            elif listen.state == ListenState.STOP:
                await self._on_listen_stop()

        elif msg_type == MessageType.ABORT.value:
            abort = AbortMessage.from_dict(data)
            await self._on_abort(abort)

        else:
            logger.debug("Unhandled message type: %s", msg_type)

    async def _handle_binary(self, data: bytes) -> None:
        """Handle a binary (audio) frame."""
        if self.state == SessionState.LISTENING:
            self.audio_buffer.add_frame(data)

    async def _on_listen_start(self, listen: ListenMessage) -> None:
        """Start collecting audio frames."""
        await self._cancel_tts()  # Stop any ongoing TTS
        self.audio_buffer.clear()
        self.state = SessionState.LISTENING
        self._abort_event.clear()
        logger.info("Listening started (mode=%s)", listen.mode.value)

    async def _on_listen_stop(self) -> None:
        """Stop collecting and process audio."""
        if self.state != SessionState.LISTENING:
            return

        self.state = SessionState.PROCESSING
        frame_count = self.audio_buffer.frame_count
        logger.info("Listening stopped, processing %d frames", frame_count)

        if frame_count == 0:
            self.state = SessionState.IDLE
            return

        # STT
        wav_data = self.audio_buffer.get_wav()
        try:
            text = await transcribe(wav_data)
        except Exception as e:
            logger.error("STT failed: %s", e)
            self.state = SessionState.IDLE
            return

        if not text:
            self.state = SessionState.IDLE
            return

        # Send STT result
        await self.websocket.send_text(stt_response(text))

        # Add to history
        self.history.add("user", text)

        # LLM + TTS pipeline
        self._tts_task = asyncio.create_task(self._llm_tts_pipeline(text))

    async def _llm_tts_pipeline(self, user_text: str) -> None:
        """Run LLM streaming → sentence chunking → TTS → send Opus frames."""
        self.state = SessionState.SPEAKING
        streamer = SentenceStreamer()
        full_response: list[str] = []

        try:
            async for chunk in stream_response(self.history, config.SYSTEM_PROMPT):
                if self._abort_event.is_set():
                    logger.info("LLM streaming aborted")
                    break

                full_response.append(chunk)
                sentences = streamer.add(chunk)

                for sentence in sentences:
                    if self._abort_event.is_set():
                        break
                    await self._send_tts_sentence(sentence)

            # Flush remaining text
            if not self._abort_event.is_set():
                remaining = streamer.flush()
                if remaining:
                    await self._send_tts_sentence(remaining)

            # Send full LLM response
            response_text = "".join(full_response)
            if response_text and not self._abort_event.is_set():
                self.history.add("assistant", response_text)
                await self.websocket.send_text(llm_response(response_text))

            # TTS stop
            if not self._abort_event.is_set():
                await self.websocket.send_text(tts_message(TTSState.STOP))

        except Exception as e:
            logger.exception("LLM/TTS pipeline error: %s", e)
        finally:
            self.state = SessionState.IDLE

    async def _send_tts_sentence(self, sentence: str) -> None:
        """Synthesize and send one sentence as Opus frames."""
        if self._abort_event.is_set():
            return

        # Send sentence_start
        await self.websocket.send_text(tts_message(TTSState.SENTENCE_START, sentence))

        try:
            opus_frames = await synthesize_to_opus_frames(sentence)

            for frame in opus_frames:
                if self._abort_event.is_set():
                    break
                await self.websocket.send_bytes(frame)

        except Exception as e:
            logger.error("TTS synthesis failed for sentence: %s — %s", sentence[:40], e)

        # Send sentence_end
        await self.websocket.send_text(tts_message(TTSState.SENTENCE_END))

    async def _on_abort(self, abort: AbortMessage) -> None:
        """Handle abort message from ESP32."""
        logger.info("Abort received: reason=%s", abort.reason)
        self._abort_event.set()
        await self._cancel_tts()
        self.state = SessionState.IDLE

    async def _cancel_tts(self) -> None:
        """Cancel any running TTS task."""
        if self._tts_task and not self._tts_task.done():
            self._abort_event.set()
            self._tts_task.cancel()
            try:
                await self._tts_task
            except asyncio.CancelledError:
                pass
            self._tts_task = None
