from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ...models.input import GenerationInput
from ...models.output import GenerationOutput, StreamDelta


class GeminiVertexV1:
    def serialize(self, gen_input: GenerationInput) -> dict[str, Any]:
        raise NotImplementedError

    async def parse_stream(
        self,
        lines: AsyncIterator[str],
    ) -> AsyncIterator[StreamDelta | GenerationOutput]:
        raise NotImplementedError
        yield
