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
from func_llm.providers.gemini.vertex_v1 import GeminiVertexV1


def _simple_input(text: str = "Hello") -> GenerationInput:
    return GenerationInput(
        model="gemini-2.5-flash",
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
        adapter = GeminiVertexV1()
        payload = adapter.serialize(_simple_input("Hi"))
        assert payload["contents"] == [
            {"role": "user", "parts": [{"text": "Hi"}]},
        ]

    def test_system_prompt(self) -> None:
        adapter = GeminiVertexV1()
        inp = _simple_input()
        inp = inp.model_copy(update={"system_prompt": "Be helpful."})
        payload = adapter.serialize(inp)
        assert payload["systemInstruction"] == {
            "parts": [{"text": "Be helpful."}],
        }

    def test_no_system_prompt(self) -> None:
        adapter = GeminiVertexV1()
        payload = adapter.serialize(_simple_input())
        assert "systemInstruction" not in payload

    def test_generation_config(self) -> None:
        adapter = GeminiVertexV1()
        inp = _simple_input()
        inp = inp.model_copy(
            update={
                "llm_config": LLMConfig(
                    temperature=0.5,
                    top_p=0.9,
                    top_k=40,
                    max_tokens=512,
                    stop=["---"],
                ),
            }
        )
        payload = adapter.serialize(inp)
        gc = payload["generationConfig"]
        assert gc["temperature"] == 0.5
        assert gc["topP"] == 0.9
        assert gc["topK"] == 40
        assert gc["maxOutputTokens"] == 512
        assert gc["stopSequences"] == ["---"]

    def test_thinking_config(self) -> None:
        adapter = GeminiVertexV1()
        inp = _simple_input()
        inp = inp.model_copy(
            update={"llm_config": LLMConfig(thinking=ThinkingLevel.MEDIUM)},
        )
        payload = adapter.serialize(inp)
        assert payload["generationConfig"]["thinkingConfig"] == {
            "thinkingBudget": 8192,
        }

    def test_no_thinking_config(self) -> None:
        adapter = GeminiVertexV1()
        payload = adapter.serialize(_simple_input())
        assert "thinkingConfig" not in payload.get("generationConfig", {})

    def test_multi_turn(self) -> None:
        adapter = GeminiVertexV1()
        inp = GenerationInput(
            model="gemini-2.5-flash",
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
        assert len(payload["contents"]) == 3
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][1]["role"] == "model"
        assert payload["contents"][2]["role"] == "user"

    def test_inline_image(self) -> None:
        adapter = GeminiVertexV1()
        inp = GenerationInput(
            model="gemini-2.5-flash",
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
        parts = payload["contents"][0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "Describe this."}
        assert parts[1] == {
            "inlineData": {"mimeType": "image/jpeg", "data": "abc123"},
        }

    def test_gcs_file(self) -> None:
        adapter = GeminiVertexV1()
        inp = GenerationInput(
            model="gemini-2.5-flash",
            conversation=[
                Message(
                    source=MessageSource.USER,
                    contents=[
                        TextContent(text="Summarize."),
                        MediaContent(
                            media_type="application/pdf",
                            source=UrlSource(url="gs://bucket/doc.pdf"),
                        ),
                    ],
                ),
            ],
        )
        payload = adapter.serialize(inp)
        parts = payload["contents"][0]["parts"]
        assert parts[1] == {
            "fileData": {
                "mimeType": "application/pdf",
                "fileUri": "gs://bucket/doc.pdf",
            },
        }

    def test_tool_definitions(self) -> None:
        adapter = GeminiVertexV1()
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
        decls = payload["tools"][0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert payload["toolConfig"] == {
            "functionCallingConfig": {"mode": "AUTO"},
        }

    def test_tool_response_serialization(self) -> None:
        adapter = GeminiVertexV1()
        inp = GenerationInput(
            model="gemini-2.5-flash",
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
        assert payload["contents"][1]["role"] == "model"
        assert payload["contents"][1]["parts"][0] == {
            "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
        }
        assert payload["contents"][2]["role"] == "user"
        assert payload["contents"][2]["parts"][0] == {
            "functionResponse": {
                "name": "get_weather",
                "response": {"temp": 22},
            },
        }

    def test_tool_response_plain_text(self) -> None:
        adapter = GeminiVertexV1()
        inp = GenerationInput(
            model="gemini-2.5-flash",
            conversation=[
                Message(
                    source=MessageSource.MODEL,
                    contents=[
                        ToolCallContent(
                            id="c1",
                            name="greet",
                            arguments={},
                        ),
                    ],
                ),
                Message(
                    source=MessageSource.TOOL,
                    contents=[
                        ToolResponseContent(
                            tool_call_id="c1",
                            content="hello world",
                        ),
                    ],
                ),
            ],
        )
        payload = adapter.serialize(inp)
        resp_part = payload["contents"][1]["parts"][0]
        assert resp_part["functionResponse"]["response"] == {
            "result": "hello world",
        }

    def test_system_messages_excluded(self) -> None:
        adapter = GeminiVertexV1()
        inp = GenerationInput(
            model="gemini-2.5-flash",
            conversation=[
                Message(
                    source=MessageSource.SYSTEM,
                    contents=[TextContent(text="ignored")],
                ),
                Message(
                    source=MessageSource.USER,
                    contents=[TextContent(text="Hi")],
                ),
            ],
        )
        payload = adapter.serialize(inp)
        assert len(payload["contents"]) == 1
        assert payload["contents"][0]["role"] == "user"


class TestParseStream:
    @pytest.mark.asyncio
    async def test_text_streaming(self) -> None:
        adapter = GeminiVertexV1()
        chunks: list[dict[str, Any]] = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello"}],
                            "role": "model",
                        },
                    }
                ],
                "modelVersion": "gemini-2.5-flash",
            },
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": " world"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                        "safetyRatings": [],
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 8,
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

    @pytest.mark.asyncio
    async def test_thinking_stream(self) -> None:
        adapter = GeminiVertexV1()
        chunks: list[dict[str, Any]] = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "reasoning...", "thought": True}],
                            "role": "model",
                        },
                    }
                ],
            },
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Answer"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 20,
                    "totalTokenCount": 35,
                },
            },
        ]

        results: list[Any] = []
        async for item in adapter.parse_stream(_lines_from(chunks)):
            results.append(item)

        assert isinstance(results[0], ThinkingDelta)
        assert results[0].thinking == "reasoning..."
        assert isinstance(results[1], TextDelta)
        assert results[1].text == "Answer"
        out: GenerationOutput = results[2]
        assert out.usage.thinking_tokens == 20

    @pytest.mark.asyncio
    async def test_function_call_stream(self) -> None:
        adapter = GeminiVertexV1()
        chunks: list[dict[str, Any]] = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"city": "Paris"},
                                    },
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
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

    @pytest.mark.asyncio
    async def test_safety_blocked(self) -> None:
        adapter = GeminiVertexV1()
        chunks: list[dict[str, Any]] = [
            {
                "candidates": [
                    {
                        "content": {"parts": [], "role": "model"},
                        "finishReason": "SAFETY",
                        "safetyRatings": [
                            {
                                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "probability": "HIGH",
                                "blocked": True,
                            }
                        ],
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 0,
                    "totalTokenCount": 5,
                },
            },
        ]

        results: list[Any] = []
        async for item in adapter.parse_stream(_lines_from(chunks)):
            results.append(item)

        out: GenerationOutput = results[0]
        assert out.finish_reason == FinishReason.CONTENT_FILTER
        assert out.safety is not None
        assert out.safety.blocked is True

    @pytest.mark.asyncio
    async def test_skips_non_sse_lines(self) -> None:
        adapter = GeminiVertexV1()

        async def lines() -> AsyncIterator[str]:
            yield ""
            yield "event: message"
            yield f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': 'ok'}], 'role': 'model'}, 'finishReason': 'STOP'}], 'usageMetadata': {'promptTokenCount': 1, 'candidatesTokenCount': 1, 'totalTokenCount': 2}})}"
            yield "data: [DONE]"
            yield ""

        results: list[Any] = []
        async for item in adapter.parse_stream(lines()):
            results.append(item)

        assert isinstance(results[0], TextDelta)
        assert isinstance(results[1], GenerationOutput)
