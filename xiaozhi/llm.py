"""LLM abstraction supporting OpenAI and Anthropic streaming."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import anthropic
import openai

import config

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ConversationHistory:
    """Manages conversation history with a rolling window."""

    messages: list[Message] = field(default_factory=list)
    max_messages: int = 10

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        # Keep only the last N messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def to_openai_format(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_anthropic_format(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        self.messages.clear()


async def stream_openai(
    history: ConversationHistory,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Stream response from OpenAI GPT."""
    client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    model = config.get_llm_model()

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history.to_openai_format())

    logger.info("OpenAI request: model=%s, messages=%d", model, len(messages))

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def stream_anthropic(
    history: ConversationHistory,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Stream response from Anthropic Claude."""
    client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    model = config.get_llm_model()

    logger.info("Anthropic request: model=%s, messages=%d", model, len(history.messages))

    async with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=system_prompt or config.SYSTEM_PROMPT,
        messages=history.to_anthropic_format(),
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def stream_response(
    history: ConversationHistory,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Stream LLM response using the configured provider."""
    provider = config.LLM_PROVIDER
    if provider == "openai":
        async for chunk in stream_openai(history, system_prompt):
            yield chunk
    elif provider == "anthropic":
        async for chunk in stream_anthropic(history, system_prompt):
            yield chunk
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def get_full_response(
    history: ConversationHistory,
    system_prompt: str | None = None,
) -> str:
    """Get full LLM response (non-streaming, for simpler use cases)."""
    chunks: list[str] = []
    async for chunk in stream_response(history, system_prompt):
        chunks.append(chunk)
    return "".join(chunks)
