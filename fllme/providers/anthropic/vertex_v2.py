from __future__ import annotations

from typing import Any

from ...models.input import GenerationInput
from .vertex_v1 import (
    AnthropicVertexV1,
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
