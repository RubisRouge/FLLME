from __future__ import annotations

import copy
from typing import Protocol, runtime_checkable

from .errors import MediaResolutionError
from .models.message import (
    Base64Source,
    MediaContent,
    Message,
    ReferenceSource,
    UrlSource,
)
from .models.output.main import GenerationOutput


@runtime_checkable
class MediaResolver(Protocol):
    async def resolve(
        self, references: list[ReferenceSource]
    ) -> list[Base64Source | UrlSource]:
        """Outbound: convert user-domain IDs to provider-sendable sources."""
        ...

    async def store(
        self, media: list[Base64Source | UrlSource]
    ) -> list[ReferenceSource]:
        """Inbound: upload AI-generated media, return user-domain IDs."""
        ...


async def resolve_references(
    messages: list[Message], resolver: MediaResolver
) -> list[Message]:
    refs: list[ReferenceSource] = []
    positions: list[tuple[int, int]] = []

    for msg_idx, msg in enumerate(messages):
        for blk_idx, block in enumerate(msg.contents):
            if isinstance(block, MediaContent) and isinstance(
                block.source, ReferenceSource
            ):
                refs.append(block.source)
                positions.append((msg_idx, blk_idx))

    if not refs:
        return messages

    try:
        resolved = await resolver.resolve(refs)
    except Exception as exc:
        raise MediaResolutionError(
            failed_ids=[r.id for r in refs],
        ) from exc

    out = copy.deepcopy(messages)
    for (msg_idx, blk_idx), new_source in zip(positions, resolved):
        block = out[msg_idx].contents[blk_idx]
        assert isinstance(block, MediaContent)
        block.source = new_source

    return out


async def store_media(message: Message, resolver: MediaResolver) -> Message:
    sources: list[Base64Source | UrlSource] = []
    positions: list[int] = []

    for blk_idx, block in enumerate(message.contents):
        if isinstance(block, MediaContent) and isinstance(
            block.source, Base64Source | UrlSource
        ):
            sources.append(block.source)
            positions.append(blk_idx)

    if not sources:
        return message

    try:
        stored = await resolver.store(sources)
    except Exception as exc:
        raise MediaResolutionError() from exc

    out = message.model_copy(deep=True)
    for blk_idx, new_ref in zip(positions, stored):
        block = out.contents[blk_idx]
        assert isinstance(block, MediaContent)
        block.source = new_ref

    return out


async def store_output_media(
    output: GenerationOutput, resolver: MediaResolver
) -> GenerationOutput:
    new_message = await store_media(output.message, resolver)
    if new_message is output.message:
        return output
    return output.model_copy(update={"message": new_message})
