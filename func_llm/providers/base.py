from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from ..models.input import GenerationInput
from ..models.message import Content, TextContent, ThinkingContent
from ..models.output import GenerationOutput, StreamDelta


class Adapter(Protocol):
    def serialize(self, gen_input: GenerationInput) -> dict[str, Any]: ...

    def parse_stream(
        self,
        lines: AsyncIterator[str],
    ) -> AsyncIterator[StreamDelta | GenerationOutput]: ...


def accumulate_content(accumulated: list[Content], item: Content) -> None:
    """Merge adjacent same-type text/thinking deltas into a single block."""
    if isinstance(item, TextContent) and accumulated and isinstance(accumulated[-1], TextContent):
        accumulated[-1] = TextContent(text=accumulated[-1].text + item.text)
    elif isinstance(item, ThinkingContent) and accumulated and isinstance(accumulated[-1], ThinkingContent):
        last = accumulated[-1]
        accumulated[-1] = ThinkingContent(thinking=last.thinking + item.thinking, signature=last.signature)
    else:
        accumulated.append(item)
