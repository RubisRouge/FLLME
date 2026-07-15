from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ...models.input import GenerationInput, ThinkingLevel, ToolsCallingMode
from ...models.output import (
    GenerationOutput,
    StreamDelta,
)
from .vertex_v1 import (
    AnthropicVertexV1,
    _serialize_content_blocks,
    _serialize_tool_results,
)


class AnthropicVertexV2(AnthropicVertexV1):
    """Anthropic adapter that omits sampling params (temperature, top_p, top_k).

    Use for models that reject these params (e.g. Claude Opus 4.7+).
    """

    def serialize(self, gen_input: GenerationInput) -> dict[str, Any]:
        payload = super().serialize(gen_input)
        payload.pop("temperature", None)
        payload.pop("top_p", None)
        payload.pop("top_k", None)
        return payload
