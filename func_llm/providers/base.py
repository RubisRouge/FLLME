from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from ..models.input import GenerationInput
from ..models.output import GenerationOutput, StreamDelta


class Adapter(Protocol):
    def serialize(self, gen_input: GenerationInput) -> dict[str, Any]: ...

    def parse_stream(
        self,
        lines: AsyncIterator[str],
    ) -> AsyncIterator[StreamDelta | GenerationOutput]: ...
