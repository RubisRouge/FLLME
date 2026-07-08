from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from .errors import ProviderError
from .http import get_client
from .media import MediaResolver, resolve_references, store_output_media
from .models.input import GenerationInput
from .models.output import GenerationOutput, StreamDelta
from .providers import get_adapter
from .providers.base import Adapter
from .service import DeploymentService

DEFAULT_SERVICE: DeploymentService | None = None


def configure(service: DeploymentService) -> None:
    global DEFAULT_SERVICE  # noqa: PLW0603
    DEFAULT_SERVICE = service


def get_service() -> DeploymentService:
    if DEFAULT_SERVICE is None:
        msg = "No service configured. Call func_llm.configure(service) first."
        raise RuntimeError(msg)
    return DEFAULT_SERVICE


async def generate(
    gen_input: GenerationInput,
    *,
    deployment_id: str | None = None,
    service: DeploymentService | None = None,
    media_resolver: MediaResolver | None = None,
) -> GenerationOutput | AsyncIterator[StreamDelta | GenerationOutput]:
    svc = service or get_service()

    if media_resolver is not None:
        resolved = await resolve_references(
            gen_input.conversation, media_resolver
        )
        gen_input = gen_input.model_copy(update={"conversation": resolved})

    deployment = await svc.resolve_deployment(gen_input.model, deployment_id)
    adapter = get_adapter(deployment.adapter)
    payload: dict[str, Any] = adapter.serialize(gen_input)
    headers = await svc.get_auth_headers(deployment)

    if gen_input.stream:
        return _stream(
            deployment.url, headers, payload, adapter, media_resolver
        )

    return await _collect(
        deployment.url, headers, payload, adapter, media_resolver
    )


async def _collect(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    adapter: Adapter,
    media_resolver: MediaResolver | None = None,
) -> GenerationOutput:
    result: GenerationOutput | None = None
    async for item in _stream(url, headers, payload, adapter, media_resolver):
        if isinstance(item, GenerationOutput):
            result = item
    if result is None:
        msg = "Stream ended without a final GenerationOutput"
        raise ProviderError(0, msg)
    return result


async def _stream(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    adapter: Adapter,
    media_resolver: MediaResolver | None = None,
) -> AsyncIterator[StreamDelta | GenerationOutput]:
    client = get_client()
    async with client.stream("POST", url, headers=headers, json=payload) as response:
        if response.status_code != 200:
            body = await response.aread()
            raise ProviderError(response.status_code, body.decode())
        async for item in adapter.parse_stream(response.aiter_lines()):
            if isinstance(item, GenerationOutput) and media_resolver is not None:
                item = await store_output_media(item, media_resolver)
            yield item
