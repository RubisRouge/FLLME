from pydantic import BaseModel
from enum import StrEnum

from .usage import Usage
from .citation import Citation
from .safety import SafetyResult
from ..message import Message


class FinishReason(StrEnum):
    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class GenerationOutput(BaseModel):
    id : str
    model : str
    message : Message
    finish_reason : FinishReason
    usage : Usage
    citations : list[Citation] = []
    safety : SafetyResult | None = None
    created : int | None = None
