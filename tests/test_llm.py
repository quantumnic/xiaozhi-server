"""Tests for LLM abstraction."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from xiaozhi.llm import ConversationHistory, Message, stream_response, get_full_response


class TestMessage:
    def test_create(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"


class TestConversationHistory:
    def test_add(self):
        h = ConversationHistory()
        h.add("user", "hello")
        assert len(h.messages) == 1
        assert h.messages[0].role == "user"

    def test_rolling_window(self):
        h = ConversationHistory(max_messages=3)
        for i in range(5):
            h.add("user", f"msg {i}")
        assert len(h.messages) == 3
        assert h.messages[0].content == "msg 2"

    def test_clear(self):
        h = ConversationHistory()
        h.add("user", "hello")
        h.clear()
        assert len(h.messages) == 0

    def test_to_openai_format(self):
        h = ConversationHistory()
        h.add("user", "hi")
        h.add("assistant", "hello")
        fmt = h.to_openai_format()
        assert fmt == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_to_anthropic_format(self):
        h = ConversationHistory()
        h.add("user", "hi")
        fmt = h.to_anthropic_format()
        assert fmt == [{"role": "user", "content": "hi"}]


class TestStreamOpenAI:
    @pytest.mark.asyncio
    async def test_stream_openai(self):
        history = ConversationHistory()
        history.add("user", "hello")

        # Mock the streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hi"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " there"

        mock_chunk3 = MagicMock()
        mock_chunk3.choices = []  # empty choices (finish)

        async def mock_stream():
            for c in [mock_chunk1, mock_chunk2, mock_chunk3]:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch("xiaozhi.llm.openai.AsyncOpenAI", return_value=mock_client):
            with patch("xiaozhi.llm.config") as mock_config:
                mock_config.OPENAI_API_KEY = "test-key"
                mock_config.LLM_PROVIDER = "openai"
                mock_config.get_llm_model.return_value = "gpt-4o"
                mock_config.SYSTEM_PROMPT = "test"

                from xiaozhi.llm import stream_openai
                chunks = []
                async for chunk in stream_openai(history, "test prompt"):
                    chunks.append(chunk)

                assert chunks == ["Hi", " there"]


class TestStreamAnthropic:
    @pytest.mark.asyncio
    async def test_stream_anthropic(self):
        history = ConversationHistory()
        history.add("user", "hello")

        # Build async context manager mock
        mock_stream_ctx = AsyncMock()

        async def mock_text_stream():
            for t in ["Hello", " from", " Claude"]:
                yield t

        mock_stream_ctx.text_stream = mock_text_stream()

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("xiaozhi.llm.anthropic.AsyncAnthropic", return_value=mock_client):
            with patch("xiaozhi.llm.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.LLM_PROVIDER = "anthropic"
                mock_config.get_llm_model.return_value = "claude-sonnet-4-20250514"
                mock_config.SYSTEM_PROMPT = "test"

                from xiaozhi.llm import stream_anthropic
                chunks = []
                async for chunk in stream_anthropic(history, "test prompt"):
                    chunks.append(chunk)

                assert chunks == ["Hello", " from", " Claude"]


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_invalid_provider(self):
        history = ConversationHistory()
        history.add("user", "hello")

        with patch("xiaozhi.llm.config") as mock_config:
            mock_config.LLM_PROVIDER = "invalid"

            with pytest.raises(ValueError, match="Unknown LLM provider"):
                async for _ in stream_response(history):
                    pass


class TestGetFullResponse:
    @pytest.mark.asyncio
    async def test_full_response(self):
        history = ConversationHistory()
        history.add("user", "hello")

        async def mock_stream(*args, **kwargs):
            for chunk in ["Hello", " ", "world"]:
                yield chunk

        with patch("xiaozhi.llm.stream_response", side_effect=mock_stream):
            result = await get_full_response(history)
            assert result == "Hello world"
