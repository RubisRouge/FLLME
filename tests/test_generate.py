import importlib
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from fllme.errors import StructuredOutputError
from fllme.generate import configure, get_service
from fllme.models.deployment import AdapterType, Deployment
from fllme.models.input import BasicOutputType, GenerationInput
from fllme.models.message import Message, MessageSource, TextContent
from fllme.models.model import LLMModel, ModelCapabilities, Provider, ThinkingStyle
from fllme.models.output import GenerationOutput
from fllme.models.output import FinishReason
from fllme.providers import ADAPTERS
from fllme.providers.gemini.vertex_v1 import GeminiVertexV1
from fllme.service import DeploymentService


def _gen_mod() -> Any:
    return importlib.import_module("fllme.generate")


class _Contact(BaseModel):
    name: str
    age: int


def _sse(chunks: list[dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)


def _client(body: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, text=body)),
    )


def _gemini_chunk(text: str, finish: str | None = None) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "content": {"parts": [{"text": text}], "role": "model"},
    }
    chunk: dict[str, Any] = {
        "candidates": [candidate],
        "modelVersion": "gemini-2.5-flash",
    }
    if finish is not None:
        candidate["finishReason"] = finish
        candidate["safetyRatings"] = []
        chunk["usageMetadata"] = {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8,
        }
    return chunk


class TestStructuredOutput:
    @pytest.mark.asyncio
    async def test_parses_json_split_across_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _sse(
            [
                _gemini_chunk('{"name": "Ana", '),
                _gemini_chunk('"age": 30}', finish="STOP"),
            ]
        )
        monkeypatch.setattr(
            _gen_mod(), "get_client", lambda timeout=120.0: _client(body)
        )
        out = await _gen_mod()._collect("https://x", {}, _Contact, {}, GeminiVertexV1())
        assert out.parsed == _Contact(name="Ana", age=30)

    @pytest.mark.asyncio
    async def test_max_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = _sse(
            [_gemini_chunk('{"name": "An', finish="MAX_TOKENS")],
        )
        monkeypatch.setattr(
            _gen_mod(), "get_client", lambda timeout=120.0: _client(body)
        )
        with pytest.raises(StructuredOutputError) as exc:
            await _gen_mod()._collect("https://x", {}, _Contact, {}, GeminiVertexV1())
        assert exc.value.output.finish_reason == FinishReason.MAX_TOKENS

    @pytest.mark.asyncio
    async def test_schema_mismatch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _sse([_gemini_chunk('{"name": "Ana"}', finish="STOP")])
        monkeypatch.setattr(
            _gen_mod(), "get_client", lambda timeout=120.0: _client(body)
        )
        with pytest.raises(StructuredOutputError):
            await _gen_mod()._collect("https://x", {}, _Contact, {}, GeminiVertexV1())

    @pytest.mark.asyncio
    async def test_text_output_not_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = _sse([_gemini_chunk("hello", finish="STOP")])
        monkeypatch.setattr(
            _gen_mod(), "get_client", lambda timeout=120.0: _client(body)
        )
        out = await _gen_mod()._collect(
            "https://x", {}, BasicOutputType.TEXT, {}, GeminiVertexV1()
        )
        assert out.parsed is None


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


class _RecordingAdapter(GeminiVertexV1):
    def __init__(self) -> None:
        self.seen: LLMModel | None = None

    def serialize(
        self, gen_input: GenerationInput, model: LLMModel | None = None
    ) -> dict[str, Any]:
        self.seen = model
        return super().serialize(gen_input, model)


class TestModelCapabilitiesReachAdapter:
    @pytest.mark.asyncio
    async def test_adapter_receives_resolved_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc = await DeploymentService.from_sqlite(":memory:")
        model = LLMModel(
            id="gem",
            name="Gem",
            provider=Provider.GEMINI,
            capabilities=ModelCapabilities(thinking_style=ThinkingStyle.LEVEL),
        )
        await svc.add_model(model)
        await svc.add_deployment(
            Deployment(
                id="d1",
                url="https://x",
                model_id="gem",
                adapter=AdapterType.GEMINI_VERTEX_V1,
                auth_id="api_key",
            )
        )

        async def _no_headers(_: Deployment) -> dict[str, str]:
            return {}

        monkeypatch.setattr(svc, "get_auth_headers", _no_headers)
        body = _sse([_gemini_chunk("hi", finish="STOP")])
        monkeypatch.setattr(
            _gen_mod(), "get_client", lambda timeout=120.0: _client(body)
        )
        adapter = _RecordingAdapter()
        monkeypatch.setitem(ADAPTERS, AdapterType.GEMINI_VERTEX_V1, adapter)

        out = await _gen_mod().generate(
            GenerationInput(
                model="gem",
                conversation=[
                    Message(
                        source=MessageSource.USER,
                        contents=[TextContent(text="Hi")],
                    )
                ],
            ),
            service=svc,
        )

        assert isinstance(out, GenerationOutput)
        assert adapter.seen == model
        assert adapter.seen.capabilities.thinking_style == ThinkingStyle.LEVEL
