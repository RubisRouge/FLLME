import importlib
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from fllme.errors import StructuredOutputError
from fllme.generate import configure, get_service
from fllme.models.input import BasicOutputType
from fllme.models.output import FinishReason
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
