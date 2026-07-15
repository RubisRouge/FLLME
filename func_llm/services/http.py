from typing import AsyncIterator

from httpx import AsyncClient
from httpx import Timeout

from .types import DeploymentService
from ..auth.header import HTTPHeader
from ..auth.types import AuthResolver
from ..errors import ProviderError
from ..errors import EMPTY_OUTPUT_STREAM_ERROR
from ..media import MediaResolver
from ..media import resolve_references
from ..media import store_output_media
from ..models.deployment import Deployment
from ..models.input import GenerationInput
from ..models.output import GenerationOutput
from ..models.output import StreamDelta
from ..providers import get_adapter
from ..store import AuthRepository
from ..store import DeploymentRepository
from ..store import ModelRepository

DEFAULT_HTTP_TIMEOUT = Timeout(120.0)


class HTTPDeploymentService(DeploymentService):
    def __init__(
        self,
        auth: AuthRepository,
        deployments: DeploymentRepository,
        models: ModelRepository,
        *,
        transport: AsyncClient | None = None,
    ) -> None:
        super().__init__(auth, deployments, models)
        if transport is None:
            transport = AsyncClient(
                timeout=DEFAULT_HTTP_TIMEOUT,
            )
        self._transport = transport

    async def close(self) -> None:
        await self._transport.aclose()

    async def stream(
        self,
        deployment: Deployment,
        input: GenerationInput,
        *,
        authorization: HTTPHeader | None = None,
        media_resolver: MediaResolver | None = None,
    ) -> AsyncIterator[StreamDelta | GenerationOutput]:
        adapter = get_adapter(deployment.adapter)
        # TODO: add support for additional headers if needed.
        headers = authorization
        payload = adapter.serialize(input)
        # TODO: add timeout handle and retry policy.
        async with self._transport.stream(
            "POST",
            deployment.url,
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise ProviderError(response.status_code, body.decode())
            async for item in adapter.parse_stream(response.aiter_lines()):
                if isinstance(item, GenerationOutput) and media_resolver is not None:
                    item = await store_output_media(item, media_resolver)
                yield item

    async def generate(
        self,
        input: GenerationInput,
        *,
        auth_resolver: AuthResolver[HTTPHeader] | None = None,
        deployment_id: str | None = None,
        media_resolver: MediaResolver | None = None,
    ) -> GenerationOutput | AsyncIterator[StreamDelta | GenerationOutput]:
        # TODO: ensure auth_resolver is supported instance of HTTP Header.
        deployment = await self.resolve_deployment(input.model, deployment_id)
        if media_resolver is not None:
            references = await resolve_references(
                input.conversation,
                media_resolver,
            )
            input = input.model_copy(
                update={"conversation": references},
            )
        authorization: HTTPHeader | None = None
        if auth_resolver is not None:
            principle = await self._auth.get(deployment.auth_id)
            authorization = await auth_resolver.resolve(principle)
        stream = self.stream(
            deployment,
            input,
            authorization=authorization,
            media_resolver=media_resolver,
        )
        if input.stream:
            return stream
        async for item in stream:
            if isinstance(item, GenerationOutput):
                return item
        raise EMPTY_OUTPUT_STREAM_ERROR
