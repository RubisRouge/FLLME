from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, TypeGuard

from pydantic import BaseModel, ValidationError

from ..errors import StructuredOutputError
from ..models.input import GenerationInput, OutputType
from ..models.message import Content, TextContent, ThinkingContent
from ..models.output import FinishReason, GenerationOutput, StreamDelta


class Adapter(Protocol):
    def serialize(self, gen_input: GenerationInput) -> dict[str, Any]: ...

    def parse_stream(
        self,
        output_type: OutputType,
        lines: AsyncIterator[str],
    ) -> AsyncIterator[StreamDelta | GenerationOutput]: ...


def accumulate_content(accumulated: list[Content], item: Content) -> None:
    """Merge adjacent same-type text/thinking deltas into a single block."""
    if (
        isinstance(item, TextContent)
        and accumulated
        and isinstance(accumulated[-1], TextContent)
    ):
        accumulated[-1] = TextContent(text=accumulated[-1].text + item.text)
    elif (
        isinstance(item, ThinkingContent)
        and accumulated
        and isinstance(accumulated[-1], ThinkingContent)
    ):
        last = accumulated[-1]
        accumulated[-1] = ThinkingContent(
            thinking=last.thinking + item.thinking,
            signature=item.signature or last.signature,
        )
    else:
        accumulated.append(item)


def is_structured(output_type: OutputType) -> TypeGuard[type[BaseModel]]:
    return isinstance(output_type, type) and issubclass(output_type, BaseModel)


def parse_structured(output: GenerationOutput, output_type: type[BaseModel]) -> None:
    if output.finish_reason != FinishReason.STOP:
        msg = f"Structured output not parsed: finish_reason={output.finish_reason}"
        raise StructuredOutputError(output, msg)
    text = "".join(
        c.text for c in output.message.contents if isinstance(c, TextContent)
    )
    try:
        output.parsed = output_type.model_validate_json(text)
    except ValidationError as e:
        msg = f"Response does not match {output_type.__name__}: {e}"
        raise StructuredOutputError(output, msg) from e
