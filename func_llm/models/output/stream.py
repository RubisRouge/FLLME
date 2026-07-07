from pydantic import BaseModel
from enum import StrEnum


class StreamEventType(StrEnum):
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"


class TextDelta(BaseModel):
    type: StreamEventType = StreamEventType.TEXT_DELTA
    text: str


class ThinkingDelta(BaseModel):
    type: StreamEventType = StreamEventType.THINKING_DELTA
    thinking: str


StreamDelta = TextDelta | ThinkingDelta
