from typing import Awaitable
from typing import Callable
from .types import AuthPrinciple
from .types import AuthResolver

DEFAULT_HEADER_NAME = "Authorization"

type HTTPHeader = dict[str, str]
type HTTPHeaderProvider = Callable[[], Awaitable[str]]



class HTTPHeaderAuthResolver(AuthResolver[HTTPHeader]):
    def __init__(
        self,
        header_provider: HTTPHeaderProvider | None = None,
    ) -> None:
        self._header_provider = header_provider

    async def resolve(
        self,
        principle: AuthPrinciple,
    ) -> HTTPHeader:
        # TODO: get config key 
        key = principle.config.get(..., DEFAULT_HEADER_NAME)
        value = await self._header_provider()
        return {key: value}
    