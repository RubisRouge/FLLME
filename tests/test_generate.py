import importlib
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from fllme.errors import ProviderError
from fllme.generate import configure, generate, get_service
from fllme.models.deployment import AdapterType, Deployment
from fllme.models.input import GenerationInput
from fllme.models.message import (
    Message,
    MessageSource,
    TextContent,
    ThinkingContent,
    WebSearchQueryContent,
)
from fllme.models.model import LLMModel, Provider
from fllme.models.output import (
    CitationType,
    FinishReason,
    GenerationOutput,
    Response,
    StreamDelta,
    TextDelta,
    ThinkingDelta,
)
from fllme.service import DeploymentService


def _gen_mod() -> Any:
    return importlib.import_module("fllme.generate")


class TestConfigureAndGetService:
    def test_get_service_unconfigured(self) -> None:
        mod = _gen_mod()
        original = mod.DEFAULT_SERVICE
        mod.DEFAULT_SERVICE = None
        try:
            with pytest.raises(RuntimeError, match="No service configured"):
                get_service()
        finally:
            mod.DEFAULT_SERVICE = original

    @pytest.mark.asyncio
    async def test_configure(self) -> None:
        mod = _gen_mod()
        original = mod.DEFAULT_SERVICE
        try:
            svc = await DeploymentService.from_sqlite(":memory:")
            configure(svc)
            assert get_service() is svc
        finally:
            mod.DEFAULT_SERVICE = original


_GEMINI_CHUNKS: list[dict[str, Any]] = [
    {
        "modelVersion": "gemini-2.5-flash",
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Looking up France.", "thought": True}],
                    "role": "model",
                }
            }
        ],
    },
    {"candidates": [{"content": {"parts": [{"text": "Paris is "}], "role": "model"}}]},
    {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "the capital of France."}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "groundingMetadata": {
                    "webSearchQueries": ["capital of France"],
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.com/paris",
                                "title": "Paris - Wikipedia",
                            }
                        }
                    ],
                    "groundingSupports": [
                        {
                            "segment": {
                                "startIndex": 0,
                                "endIndex": 31,
                                "text": "Paris is the capital of France.",
                            },
                            "groundingChunkIndices": [0],
                        }
                    ],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 4,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": 3,
            "totalTokenCount": 14,
        },
    },
]

_GEMINI_SSE = "".join(f"data: {json.dumps(c)}\n\n" for c in _GEMINI_CHUNKS).encode()


def _input(*, stream: bool) -> GenerationInput:
    return GenerationInput(
        model="gemini",
        stream=stream,
        conversation=[
            Message(
                source=MessageSource.USER,
                contents=[TextContent(text="What is the capital of France?")],
            )
        ],
    )


@pytest_asyncio.fixture
async def service() -> AsyncGenerator[DeploymentService]:
    svc = await DeploymentService.from_sqlite(":memory:")
    await svc.add_model(LLMModel(id="gemini", name="gemini", provider=Provider.GEMINI))
    await svc.add_deployment(
        Deployment(
            id="gemini-vertex",
            url="https://example.test/gemini:streamGenerateContent?alt=sse",
            model_id="gemini",
            adapter=AdapterType.GEMINI_VERTEX_V1,
            auth_id="google_adc",
        )
    )
    with patch.object(svc, "get_auth_headers", AsyncMock(return_value={})):
        yield svc


def _mock_client(
    status_code: int = 200, body: bytes = _GEMINI_SSE
) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), requests


class TestGenerateStreamArgument:
    @pytest.mark.asyncio
    async def test_stream_false_and_true(self, service: DeploymentService) -> None:
        client, requests = _mock_client()
        with patch("fllme.generate.get_client", return_value=client):
            collected = await generate(_input(stream=False), service=service)
            assert len(requests) == 1

            stream = await generate(_input(stream=True), service=service)
            assert len(requests) == 1
            assert not isinstance(stream, GenerationOutput)
            responses = [r async for r in stream]
            assert len(requests) == 2

        assert isinstance(collected, GenerationOutput)
        assert collected.model == "gemini-2.5-flash"
        assert collected.finish_reason == FinishReason.STOP
        assert collected.message.source == MessageSource.MODEL
        assert collected.message.contents == [
            ThinkingContent(thinking="Looking up France."),
            TextContent(text="Paris is the capital of France."),
            WebSearchQueryContent(query="capital of France"),
        ]
        assert collected.usage.input_tokens == 4
        assert collected.usage.output_tokens == 7
        assert collected.usage.thinking_tokens == 3
        assert collected.usage.total_tokens == 14
        assert len(collected.citations) == 1
        assert collected.citations[0].type == CitationType.SEARCH
        assert collected.citations[0].url == "https://example.com/paris"
        assert collected.citations[0].title == "Paris - Wikipedia"
        assert collected.citations[0].span is not None
        assert collected.citations[0].span.text == "Paris is the capital of France."

        assert all(isinstance(r, Response) for r in responses)
        assert [r.content for r in responses[:-1]] == [
            StreamDelta(delta=ThinkingDelta(thinking="Looking up France.")),
            StreamDelta(delta=TextDelta(text="Paris is ")),
            StreamDelta(delta=TextDelta(text="the capital of France.")),
        ]
        streamed = responses[-1].content
        assert isinstance(streamed, GenerationOutput)
        assert streamed.model_dump(exclude={"id"}) == collected.model_dump(
            exclude={"id"}
        )

    @pytest.mark.asyncio
    async def test_provider_error(self, service: DeploymentService) -> None:
        client, _ = _mock_client(status_code=429, body=b"quota exceeded")
        with patch("fllme.generate.get_client", return_value=client):
            with pytest.raises(ProviderError, match="quota exceeded") as collect_err:
                await generate(_input(stream=False), service=service)

            stream = await generate(_input(stream=True), service=service)
            assert not isinstance(stream, GenerationOutput)
            with pytest.raises(ProviderError, match="quota exceeded") as stream_err:
                [r async for r in stream]

        assert collect_err.value.status_code == 429
        assert stream_err.value.status_code == 429
