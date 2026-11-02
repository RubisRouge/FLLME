from enum import StrEnum
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import Field

from .citation import Citation
from .safety import SafetyResult
from .stream import StreamDelta
from .usage import Usage
from ..message import Message


class FinishReason(StrEnum):
    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class GenerationOutput(BaseModel):
    type: Literal["GenerationOutput"] = "GenerationOutput"

    id: str
    model: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    citations: list[Citation] = []
    safety: SafetyResult | None = None
    created: int | None = None
    parsed: Any = None


class Response(BaseModel):
    content: Annotated[
        Union[GenerationOutput, StreamDelta], Field(discriminator="type")
    ]
