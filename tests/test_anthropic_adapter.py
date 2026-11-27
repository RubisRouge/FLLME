from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from fllme.errors import StructuredOutputError
from fllme.models.input import BasicOutputType
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
