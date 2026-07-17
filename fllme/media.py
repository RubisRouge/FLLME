from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models.input.main import GenerationInput
from .models.output.main import GenerationOutput


@runtime_checkable
class MediaResolver(Protocol):
    async def resolve_media(self, generation_input: GenerationInput) -> None:
        """Work directly on the input.conversation message's content to transform them into Base64 or UrlSource"""
        ...

    async def store_media(self, output: GenerationOutput) -> None:
        """Work directly on the output.message's content to transform them into ReferenceSource"""
        ...
