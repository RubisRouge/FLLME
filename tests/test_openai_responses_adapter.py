from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from fllme.errors import ProviderError, SerializationError, StructuredOutputError
from fllme.models.deployment import AdapterType
from fllme.models.input import (
    BasicOutputType,
    GenerationInput,
    LLMConfig,
    ThinkingLevel,
    Tool,
    ToolsCallingMode,
    ToolsConfig,
    WebSearchTool,
)
from fllme.models.message import (
    Base64Source,
    MediaContent,
    Message,
    MessageSource,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResponseContent,
    UrlSource,
)
from fllme.models.model import LLMModel, ModelCapabilities, Provider, ThinkingStyle
from fllme.models.output import (
    FinishReason,
    GenerationOutput,
    StreamDelta,
    TextDelta,
    ThinkingDelta,
)
from fllme.providers import get_adapter
from fllme.providers.openai.azure_responses_v1 import OpenAIAzureResponsesV1


class _Contact(BaseModel):
    name: str
    city: str


def _input(text: str = "Hello", **kwargs: Any) -> GenerationInput:
    return GenerationInput(
        model="gpt-5.6-luna",
        conversation=[
            Message(source=MessageSource.USER, contents=[TextContent(text=text)])
        ],
        **kwargs,
    )


def _tools(parallel: bool = False) -> ToolsConfig:
    return ToolsConfig(
        tools=[
            Tool(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            )
        ],
        parallel_calling=parallel,
        mode=ToolsCallingMode.AUTO,
    )


def _model(**caps: Any) -> LLMModel:
    return LLMModel(
        id="gpt-5.6-luna",
        name="Luna",
        provider=Provider.OPENAI,
        capabilities=ModelCapabilities(**caps),
    )


class TestSerialize:
    def test_registered(self) -> None:
        assert isinstance(
            get_adapter(AdapterType.OPENAI_AZURE_RESPONSES_V1),
            OpenAIAzureResponsesV1,
        )

    def test_basic_text(self) -> None:
        payload = OpenAIAzureResponsesV1().serialize(_input("Hi"))
        assert payload["model"] == "gpt-5.6-luna"
        assert payload["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
        ]
        assert payload["max_output_tokens"] == 1024
        assert payload["stream"] is True
        assert payload["store"] is False
        for chat_only in ("messages", "max_completion_tokens", "stream_options"):
            assert chat_only not in payload
        for absent in ("reasoning", "tools", "text", "instructions"):
            assert absent not in payload

    def test_system_prompt_and_system_messages(self) -> None:
        inp = _input(system_prompt="Be brief")
        inp.conversation.insert(
            0, Message(source=MessageSource.SYSTEM, contents=[TextContent(text="Rule")])
        )
        payload = OpenAIAzureResponsesV1().serialize(inp)
        assert payload["instructions"] == "Be brief"
        assert payload["input"][0] == {"role": "system", "content": "Rule"}

    def test_sampling_params(self) -> None:
        inp = _input(
            llm_config=LLMConfig(
                temperature=0.2,
                top_p=0.9,
                max_tokens=50,
                stop=["x"],
                presence_penalty=0.1,
                frequency_penalty=0.1,
            )
        )
        payload = OpenAIAzureResponsesV1().serialize(inp)
        assert payload["temperature"] == 0.2
        assert payload["top_p"] == 0.9
        assert payload["max_output_tokens"] == 50
        for unsupported in ("stop", "presence_penalty", "frequency_penalty"):
            assert unsupported not in payload

    def test_candidates_raises(self) -> None:
        with pytest.raises(SerializationError, match="candidates"):
            OpenAIAzureResponsesV1().serialize(
                _input(llm_config=LLMConfig(candidates=2))
            )

    def test_media(self) -> None:
        inp = GenerationInput(
            model="m",
            conversation=[
                Message(
                    source=MessageSource.USER,
                    contents=[
                        TextContent(text="look"),
                        MediaContent(
                            source=Base64Source(data="AAA"), media_type="image/png"
                        ),
                        MediaContent(
                            source=UrlSource(url="https://x/y.png"),
                            media_type="image/png",
                        ),
                    ],
                )
            ],
        )
        content = OpenAIAzureResponsesV1().serialize(inp)["input"][0]["content"]
        assert content == [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAA"},
            {"type": "input_image", "image_url": "https://x/y.png"},
        ]

    @pytest.mark.parametrize("level", [ThinkingLevel.LOW, ThinkingLevel.HIGH])
    def test_thinking(self, level: ThinkingLevel) -> None:
        payload = OpenAIAzureResponsesV1().serialize(
            _input(llm_config=LLMConfig(thinking=level))
        )
        assert payload["reasoning"] == {"effort": level.value, "summary": "auto"}

    def test_thinking_skipped_for_models_without_reasoning(self) -> None:
        payload = OpenAIAzureResponsesV1().serialize(
            _input(llm_config=LLMConfig(thinking=ThinkingLevel.HIGH)),
            _model(thinking_style=ThinkingStyle.NONE),
        )
        assert "reasoning" not in payload

    def test_tools_are_flat_and_non_strict(self) -> None:
        payload = OpenAIAzureResponsesV1().serialize(_input(tool_config=_tools(True)))
        assert payload["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
                "strict": False,
            }
        ]
        assert payload["tool_choice"] == "auto"
        assert payload["parallel_tool_calls"] is True

    def test_tools_with_thinking_together(self) -> None:
        # The case chat/completions rejects for GPT-5.6.
        payload = OpenAIAzureResponsesV1().serialize(
            _input(
                tool_config=_tools(),
                llm_config=LLMConfig(thinking=ThinkingLevel.LOW),
            )
        )
        assert payload["reasoning"]["effort"] == "low"
        assert payload["tools"]

    def test_parallel_tool_calls_capability(self) -> None:
        payload = OpenAIAzureResponsesV1().serialize(
            _input(tool_config=_tools()), _model(supports_parallel_tool_calls=False)
        )
        assert "parallel_tool_calls" not in payload

    def test_web_search_raises(self) -> None:
        inp = _input(
            tool_config=ToolsConfig(
                tools=[WebSearchTool()],
                parallel_calling=False,
                mode=ToolsCallingMode.AUTO,
            )
        )
        with pytest.raises(SerializationError):
            OpenAIAzureResponsesV1().serialize(inp)

    def test_tool_call_round_trip_items(self) -> None:
        inp = GenerationInput(
            model="m",
            conversation=[
                Message(source=MessageSource.USER, contents=[TextContent(text="w?")]),
                Message(
                    source=MessageSource.MODEL,
                    contents=[
                        ThinkingContent(thinking="hmm"),
                        TextContent(text="Checking"),
                        ToolCallContent(
                            id="call_1", name="get_weather", arguments={"city": "Paris"}
                        ),
                    ],
                ),
                Message(
                    source=MessageSource.TOOL,
                    contents=[
                        ToolResponseContent(tool_call_id="call_1", content="sunny")
                    ],
                ),
            ],
        )
        items = OpenAIAzureResponsesV1().serialize(inp)["input"]
        assert items[1] == {"role": "assistant", "content": "Checking"}
        assert items[2] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        }
        assert items[3] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "sunny",
        }
        assert len(items) == 4

    def test_structured_output_schema(self) -> None:
        payload = OpenAIAzureResponsesV1().serialize(_input(output_type=_Contact))
        fmt = payload["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "_Contact"
        assert fmt["strict"] is True
        assert fmt["schema"]["additionalProperties"] is False
        assert fmt["schema"]["required"] == ["name", "city"]

    def test_text_output_has_no_format(self) -> None:
        payload = OpenAIAzureResponsesV1().serialize(
            _input(output_type=BasicOutputType.TEXT)
        )
        assert "text" not in payload


def _sse(events: list[dict[str, Any]]) -> AsyncIterator[str]:
    async def gen() -> AsyncIterator[str]:
        for event in events:
            yield f"event: {event['type']}"
            yield f"data: {json.dumps(event)}"
            yield ""

    return gen()


def _created() -> dict[str, Any]:
    return {
        "type": "response.created",
        "response": {"id": "resp_1", "model": "gpt-5.6-luna", "status": "in_progress"},
    }


def _completed(**response: Any) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "model": "gpt-5.6-luna",
            "status": "completed",
            **response,
        },
    }


def _text_delta(text: str) -> dict[str, Any]:
    return {"type": "response.output_text.delta", "item_id": "msg_1", "delta": text}


async def _run(
    events: list[dict[str, Any]], output_type: Any = BasicOutputType.TEXT
) -> list[StreamDelta | GenerationOutput]:
    return [
        item
        async for item in OpenAIAzureResponsesV1().parse_stream(
            output_type, _sse(events)
        )
    ]


def _final(items: list[StreamDelta | GenerationOutput]) -> GenerationOutput:
    out = items[-1]
    assert isinstance(out, GenerationOutput)
    return out


class TestParseStream:
    @pytest.mark.asyncio
    async def test_text_and_usage(self) -> None:
        items = await _run(
            [
                _created(),
                _text_delta("Hel"),
                _text_delta("lo"),
                _completed(
                    usage={
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 4},
                        "output_tokens": 7,
                        "output_tokens_details": {"reasoning_tokens": 3},
                        "total_tokens": 17,
                    }
                ),
            ]
        )
        deltas = [i for i in items if isinstance(i, StreamDelta)]
        assert [d.delta.text for d in deltas if isinstance(d.delta, TextDelta)] == [
            "Hel",
            "lo",
        ]
        out = _final(items)
        assert out.id == "resp_1"
        assert out.model == "gpt-5.6-luna"
        assert out.finish_reason == FinishReason.STOP
        assert out.message.contents == [TextContent(text="Hello")]
        assert out.usage.input_tokens == 10
        assert out.usage.output_tokens == 7
        assert out.usage.thinking_tokens == 3
        assert out.usage.cache.read_tokens == 4
        assert out.usage.total_tokens == 17

    @pytest.mark.asyncio
    async def test_reasoning_summary(self) -> None:
        items = await _run(
            [
                _created(),
                {"type": "response.reasoning_summary_part.added", "summary_index": 0},
                {"type": "response.reasoning_summary_text.delta", "delta": "Think"},
                {"type": "response.reasoning_summary_text.delta", "delta": "ing"},
                {"type": "response.reasoning_summary_part.added", "summary_index": 1},
                {"type": "response.reasoning_summary_text.delta", "delta": "More"},
                _text_delta("Answer"),
                _completed(),
            ]
        )
        thinking = [
            i.delta.thinking
            for i in items
            if isinstance(i, StreamDelta) and isinstance(i.delta, ThinkingDelta)
        ]
        assert thinking == ["Think", "ing", "\n\n", "More"]
        assert _final(items).message.contents == [
            ThinkingContent(thinking="Thinking\n\nMore"),
            TextContent(text="Answer"),
        ]

    @pytest.mark.asyncio
    async def test_function_call(self) -> None:
        items = await _run(
            [
                _created(),
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": "",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"city": ',
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '"Paris"}',
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                },
                _completed(),
            ]
        )
        out = _final(items)
        assert out.finish_reason == FinishReason.TOOL_USE
        assert out.message.contents == [
            ToolCallContent(
                id="call_1", name="get_weather", arguments={"city": "Paris"}
            )
        ]

    @pytest.mark.asyncio
    async def test_two_parallel_function_calls(self) -> None:
        events: list[dict[str, Any]] = [_created()]
        for idx, city in enumerate(["Paris", "Rome"]):
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": {
                        "type": "function_call",
                        "id": f"fc_{idx}",
                        "call_id": f"call_{idx}",
                        "name": "get_weather",
                        "arguments": "",
                    },
                }
            )
        for idx, city in enumerate(["Paris", "Rome"]):
            events.append(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": f"fc_{idx}",
                    "output_index": idx,
                    "delta": json.dumps({"city": city}),
                }
            )
        events.append(_completed())
        out = _final(await _run(events))
        assert [
            c.arguments for c in out.message.contents if isinstance(c, ToolCallContent)
        ] == [
            {"city": "Paris"},
            {"city": "Rome"},
        ]

    @pytest.mark.asyncio
    async def test_incomplete_max_tokens(self) -> None:
        items = await _run(
            [
                _created(),
                _text_delta("cut"),
                {
                    "type": "response.incomplete",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-5.6-luna",
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                },
            ]
        )
        assert _final(items).finish_reason == FinishReason.MAX_TOKENS

    @pytest.mark.asyncio
    async def test_incomplete_content_filter(self) -> None:
        items = await _run(
            [
                _created(),
                {
                    "type": "response.incomplete",
                    "response": {"incomplete_details": {"reason": "content_filter"}},
                },
            ]
        )
        assert _final(items).finish_reason == FinishReason.CONTENT_FILTER

    @pytest.mark.asyncio
    async def test_structured_output_parsed(self) -> None:
        items = await _run(
            [
                _created(),
                _text_delta('{"name": "Ana", '),
                _text_delta('"city": "Paris"}'),
                _completed(),
            ],
            _Contact,
        )
        assert _final(items).parsed == _Contact(name="Ana", city="Paris")

    @pytest.mark.asyncio
    async def test_structured_output_truncated_raises(self) -> None:
        with pytest.raises(StructuredOutputError):
            await _run(
                [
                    _created(),
                    _text_delta('{"name": "An'),
                    {
                        "type": "response.incomplete",
                        "response": {
                            "incomplete_details": {"reason": "max_output_tokens"}
                        },
                    },
                ],
                _Contact,
            )

    @pytest.mark.asyncio
    async def test_failed_response_raises_provider_error(self) -> None:
        with pytest.raises(ProviderError) as exc:
            await _run(
                [
                    _created(),
                    {
                        "type": "response.failed",
                        "response": {
                            "error": {"code": "server_error", "message": "boom"}
                        },
                    },
                ]
            )
        assert "boom" in exc.value.body

    @pytest.mark.asyncio
    async def test_error_event_raises_provider_error(self) -> None:
        with pytest.raises(ProviderError) as exc:
            await _run([{"type": "error", "code": "x", "message": "bad things"}])
        assert "bad things" in exc.value.body

    @pytest.mark.asyncio
    async def test_skips_non_data_lines_and_unknown_events(self) -> None:
        async def lines() -> AsyncIterator[str]:
            yield ": keepalive"
            yield "event: response.created"
            yield "data: " + json.dumps(_created())
            yield "data: " + json.dumps({"type": "response.some_future_event"})
            yield "data: [DONE]"

        items = [
            i
            async for i in OpenAIAzureResponsesV1().parse_stream(
                BasicOutputType.TEXT, lines()
            )
        ]
        assert _final(items).id == "resp_1"
