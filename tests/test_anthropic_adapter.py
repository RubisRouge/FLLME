from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from fllme.errors import SerializationError, StructuredOutputError
from fllme.models.input import (
    BasicOutputType,
    GenerationInput,
    LLMConfig,
    ThinkingLevel,
)
from fllme.models.message import Message, MessageSource, TextContent
from fllme.models.output import FinishReason, GenerationOutput
from fllme.providers.anthropic.vertex_v1 import AnthropicVertexV1
from fllme.providers.anthropic.vertex_v2 import AnthropicVertexV2


class _Contact(BaseModel):
    name: str
    city: str


def _sse(events: list[tuple[str, dict[str, Any]]]) -> AsyncIterator[str]:
    async def gen() -> AsyncIterator[str]:
        for event, data in events:
            yield f"event: {event}"
            yield f"data: {json.dumps(data)}"

    return gen()


def _text_stream(parts: list[str], stop_reason: str) -> AsyncIterator[str]:
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "message": {
                    "id": "msg_1",
                    "model": "claude",
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                }
            },
        ),
        (
            "content_block_start",
            {"content_block": {"type": "text", "text": ""}},
        ),
    ]
    events += [
        (
            "content_block_delta",
            {"delta": {"type": "text_delta", "text": part}},
        )
        for part in parts
    ]
    events += [
        ("content_block_stop", {}),
        (
            "message_delta",
            {"delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 9}},
        ),
        ("message_stop", {}),
    ]
    return _sse(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls", [AnthropicVertexV1, AnthropicVertexV2])
class TestStructuredOutput:
    async def test_parsed(self, adapter_cls: type[AnthropicVertexV1]) -> None:
        lines = _text_stream(['{"name": "Ana", ', '"city": "Paris"}'], "end_turn")
        results = [item async for item in adapter_cls().parse_stream(_Contact, lines)]

        out = results[-1]
        assert isinstance(out, GenerationOutput)
        assert out.finish_reason == FinishReason.STOP
        assert out.parsed == _Contact(name="Ana", city="Paris")

    async def test_invalid_json_raises(
        self, adapter_cls: type[AnthropicVertexV1]
    ) -> None:
        lines = _text_stream(["not json"], "end_turn")
        with pytest.raises(StructuredOutputError):
            async for _ in adapter_cls().parse_stream(_Contact, lines):
                pass

    async def test_max_tokens_raises(
        self, adapter_cls: type[AnthropicVertexV1]
    ) -> None:
        lines = _text_stream(['{"name": "An'], "max_tokens")
        with pytest.raises(StructuredOutputError):
            async for _ in adapter_cls().parse_stream(_Contact, lines):
                pass

    async def test_text_output_unchanged(
        self, adapter_cls: type[AnthropicVertexV1]
    ) -> None:
        lines = _text_stream(["hello"], "end_turn")
        results = [
            item
            async for item in adapter_cls().parse_stream(BasicOutputType.TEXT, lines)
        ]
        out = results[-1]
        assert isinstance(out, GenerationOutput)
        assert out.parsed is None


def _input(thinking: ThinkingLevel, max_tokens: int) -> GenerationInput:
    return GenerationInput(
        model="claude",
        conversation=[
            Message(source=MessageSource.USER, contents=[TextContent(text="Hi")])
        ],
        llm_config=LLMConfig(thinking=thinking, max_tokens=max_tokens),
    )


@pytest.mark.parametrize("adapter_cls", [AnthropicVertexV1, AnthropicVertexV2])
class TestThinkingBudget:
    def test_budget_below_max_tokens(
        self, adapter_cls: type[AnthropicVertexV1]
    ) -> None:
        payload = adapter_cls().serialize(_input(ThinkingLevel.HIGH, 8192))
        assert payload["thinking"]["type"] == "enabled"
        assert payload["thinking"]["budget_tokens"] < 8192
        assert payload["thinking"]["budget_tokens"] >= 1024

    def test_large_max_tokens_keeps_budget(
        self, adapter_cls: type[AnthropicVertexV1]
    ) -> None:
        payload = adapter_cls().serialize(_input(ThinkingLevel.HIGH, 64000))
        assert payload["thinking"]["budget_tokens"] == 32768

    def test_default_max_tokens_raises(
        self, adapter_cls: type[AnthropicVertexV1]
    ) -> None:
        for level in (ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH):
            with pytest.raises(SerializationError, match="max_tokens"):
                adapter_cls().serialize(_input(level, 1024))

    def test_smallest_valid_max_tokens(
        self, adapter_cls: type[AnthropicVertexV1]
    ) -> None:
        payload = adapter_cls().serialize(_input(ThinkingLevel.LOW, 2048))
        assert payload["thinking"]["budget_tokens"] == 1024

    def test_no_thinking_untouched(self, adapter_cls: type[AnthropicVertexV1]) -> None:
        payload = adapter_cls().serialize(_input(ThinkingLevel.NO, 1024))
        assert "thinking" not in payload
