from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from func_llm.models.input import (
    GenerationInput,
    LLMConfig,
    ThinkingLevel,
    Tool,
    ToolsCallingMode,
    ToolsConfig,
)
from func_llm.models.message import (
    Base64Source,
    MediaContent,
    Message,
    MessageSource,
    TextContent,
    ToolCallContent,
    ToolResponseContent,
    UrlSource,
)
from func_llm.models.output import (
    FinishReason,
    GenerationOutput,
    TextDelta,
    ThinkingDelta,
)
from func_llm.providers.openai.azure_v1 import OpenAIAzureV1


def _simple_input(text: str = "Hello") -> GenerationInput:
    return GenerationInput(
        model="gpt-4o",
        conversation=[
            Message(
                source=MessageSource.USER,
                contents=[TextContent(text=text)],
            ),
        ],
    )


async def _lines_from(chunks: list[dict[str, Any]]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}"


class TestSerialize:
    def test_basic_text(self) -> None:
        adapter = OpenAIAzureV1()
        payload = adapter.serialize(_simple_input("Hi"))
        messages = payload["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hi"}

    def test_system_prompt(self) -> None:
        adapter = OpenAIAzureV1()
        inp = _simple_input()
        inp = inp.model_copy(update={"system_prompt": "Be helpful."})
        payload = adapter.serialize(inp)
        assert payload["messages"][0] == {
            "role": "system",
            "content": "Be helpful.",
        }
        assert payload["messages"][1]["role"] == "user"

    def test_no_system_prompt(self) -> None:
        adapter = OpenAIAzureV1()
        payload = adapter.serialize(_simple_input())
        assert payload["messages"][0]["role"] == "user"

    def test_generation_config(self) -> None:
        adapter = OpenAIAzureV1()
        inp = _simple_input()
        inp = inp.model_copy(
            update={
                "llm_config": LLMConfig(
                    temperature=0.5,
                    top_p=0.9,
                    max_tokens=512,
                    stop=["---"],
                    presence_penalty=0.1,
                    frequency_penalty=0.2,
                ),
            }
        )
        payload = adapter.serialize(inp)
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.9
        assert payload["max_completion_tokens"] == 512
        assert payload["stop"] == ["---"]
        assert payload["presence_penalty"] == 0.1
        assert payload["frequency_penalty"] == 0.2

    def test_thinking_config(self) -> None:
        adapter = OpenAIAzureV1()
        inp = _simple_input()
        inp = inp.model_copy(
            update={"llm_config": LLMConfig(thinking=ThinkingLevel.MEDIUM)},
        )
        payload = adapter.serialize(inp)
        assert payload["reasoning_effort"] == "medium"

    def test_no_thinking_config(self) -> None:
        adapter = OpenAIAzureV1()
        payload = adapter.serialize(_simple_input())
        assert "reasoning_effort" not in payload

    def test_multi_turn(self) -> None:
        adapter = OpenAIAzureV1()
        inp = GenerationInput(
            model="gpt-4o",
            conversation=[
                Message(
                    source=MessageSource.USER,
                    contents=[TextContent(text="Hi")],
                ),
                Message(
                    source=MessageSource.MODEL,
                    contents=[TextContent(text="Hello!")],
                ),
                Message(
                    source=MessageSource.USER,
                    contents=[TextContent(text="How?")],
                ),
            ],
        )
        payload = adapter.serialize(inp)
        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][1]["content"] == "Hello!"
        assert payload["messages"][2]["role"] == "user"

    def test_inline_image(self) -> None:
        adapter = OpenAIAzureV1()
        inp = GenerationInput(
            model="gpt-4o",
            conversation=[
                Message(
                    source=MessageSource.USER,
                    contents=[
                        TextContent(text="Describe this."),
                        MediaContent(
                            media_type="image/jpeg",
                            source=Base64Source(data="abc123"),
                        ),
                    ],
                ),
            ],
        )
        payload = adapter.serialize(inp)
        content = payload["messages"][0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Describe this."}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,abc123"},
        }

    def test_url_image(self) -> None:
        adapter = OpenAIAzureV1()
        inp = GenerationInput(
            model="gpt-4o",
            conversation=[
                Message(
                    source=MessageSource.USER,
                    contents=[
                        TextContent(text="Describe."),
                        MediaContent(
                            media_type="image/png",
                            source=UrlSource(url="https://example.com/img.png"),
                        ),
                    ],
                ),
            ],
        )
        payload = adapter.serialize(inp)
        content = payload["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        }

    def test_tool_definitions(self) -> None:
        adapter = OpenAIAzureV1()
        inp = _simple_input()
        inp = inp.model_copy(
            update={
                "tool_config": ToolsConfig(
                    tools=[
                        Tool(
                            name="get_weather",
                            description="Get weather.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                },
                                "required": ["city"],
                            },
                        ),
                    ],
                    parallel_calling=True,
                    mode=ToolsCallingMode.AUTO,
                ),
            }
        )
        payload = adapter.serialize(inp)
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["function"]["name"] == "get_weather"
        assert payload["tool_choice"] == "auto"
        assert payload["parallel_tool_calls"] is True

    def test_tool_response_serialization(self) -> None:
        adapter = OpenAIAzureV1()
        inp = GenerationInput(
            model="gpt-4o",
            conversation=[
                Message(
                    source=MessageSource.USER,
                    contents=[TextContent(text="Weather?")],
                ),
                Message(
                    source=MessageSource.MODEL,
                    contents=[
                        ToolCallContent(
                            id="call_1",
                            name="get_weather",
                            arguments={"city": "Paris"},
                        ),
                    ],
                ),
                Message(
                    source=MessageSource.TOOL,
                    contents=[
                        ToolResponseContent(
                            tool_call_id="call_1",
                            content='{"temp": 22}',
                        ),
                    ],
                ),
            ],
        )
        payload = adapter.serialize(inp)
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][1]["tool_calls"][0] == {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
        }
        assert payload["messages"][2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"temp": 22}',
        }

    def test_system_messages_in_conversation(self) -> None:
        adapter = OpenAIAzureV1()
        inp = GenerationInput(
            model="gpt-4o",
            conversation=[
                Message(
                    source=MessageSource.SYSTEM,
                    contents=[TextContent(text="System note")],
                ),
                Message(
                    source=MessageSource.USER,
                    contents=[TextContent(text="Hi")],
                ),
            ],
        )
        payload = adapter.serialize(inp)
        assert payload["messages"][0] == {
            "role": "system",
            "content": "System note",
        }
        assert payload["messages"][1]["role"] == "user"

    def test_stream_flags_always_present(self) -> None:
        adapter = OpenAIAzureV1()
        payload = adapter.serialize(_simple_input())
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}


class TestParseStream:
    @pytest.mark.asyncio
    async def test_text_streaming(self) -> None:
        adapter = OpenAIAzureV1()
        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world"},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "model": "gpt-4o",
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
        ]

        results: list[Any] = []
        async for item in adapter.parse_stream(_lines_from(chunks)):
            results.append(item)

        assert len(results) == 3
        assert isinstance(results[0], TextDelta)
        assert results[0].text == "Hello"
        assert isinstance(results[1], TextDelta)
        assert results[1].text == " world"
        assert isinstance(results[2], GenerationOutput)
        out: GenerationOutput = results[2]
        assert out.finish_reason == FinishReason.STOP
        assert out.usage.input_tokens == 5
        assert out.usage.output_tokens == 3
        assert out.id == "chatcmpl-123"
        assert out.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_thinking_stream(self) -> None:
        adapter = OpenAIAzureV1()
        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-456",
                "model": "o1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "thinking..."},
                    }
                ],
            },
            {
                "id": "chatcmpl-456",
                "model": "o1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "id": "chatcmpl-456",
                "model": "o1",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
            },
        ]

        results: list[Any] = []
        async for item in adapter.parse_stream(_lines_from(chunks)):
            results.append(item)

        assert isinstance(results[0], ThinkingDelta)
        assert results[0].thinking == "thinking..."
        assert isinstance(results[1], TextDelta)
        assert results[1].text == "Answer"
        out: GenerationOutput = results[2]
        assert out.usage.thinking_tokens == 20

    @pytest.mark.asyncio
    async def test_function_call_stream(self) -> None:
        adapter = OpenAIAzureV1()
        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-789",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"ci',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "id": "chatcmpl-789",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'ty": "Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {
                "id": "chatcmpl-789",
                "model": "gpt-4o",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ]

        results: list[Any] = []
        async for item in adapter.parse_stream(_lines_from(chunks)):
            results.append(item)

        assert len(results) == 1
        assert isinstance(results[0], GenerationOutput)
        out: GenerationOutput = results[0]
        assert out.finish_reason == FinishReason.TOOL_USE
        assert len(out.message.contents) == 1
        tc = out.message.contents[0]
        assert isinstance(tc, ToolCallContent)
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Paris"}
        assert tc.id == "call_abc"

    @pytest.mark.asyncio
    async def test_skips_non_sse_lines(self) -> None:
        adapter = OpenAIAzureV1()

        async def lines() -> AsyncIterator[str]:
            yield ""
            yield "event: message"
            yield f"data: {json.dumps({'id': 'c1', 'model': 'gpt-4o', 'choices': [{'index': 0, 'delta': {'content': 'ok'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})}"
            yield "data: [DONE]"
            yield ""

        results: list[Any] = []
        async for item in adapter.parse_stream(lines()):
            results.append(item)

        assert isinstance(results[0], TextDelta)
        assert isinstance(results[1], GenerationOutput)

    @pytest.mark.asyncio
    async def test_usage_with_cache(self) -> None:
        adapter = OpenAIAzureV1()
        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-cache",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "cached"},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "id": "chatcmpl-cache",
                "model": "gpt-4o",
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "prompt_tokens_details": {"cached_tokens": 50},
                },
            },
        ]

        results: list[Any] = []
        async for item in adapter.parse_stream(_lines_from(chunks)):
            results.append(item)

        out: GenerationOutput = results[-1]
        assert out.usage.input_tokens == 100
        assert out.usage.output_tokens == 10
        assert out.usage.cache is not None
        assert out.usage.cache.read_tokens == 50
