from pydantic import BaseModel
from enum import StrEnum
from typing import Any


class ContentType(StrEnum):
    TEXT = "text"
    MEDIA = "media"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    THINKING = "thinking"
    ERROR = "error"


class SourceType(StrEnum):
    BYTES = "base64"
    URL = "url"
    REFERENCE = "reference"


class Base64Source(BaseModel):
    type: SourceType = SourceType.BYTES
    data: str


class UrlSource(BaseModel):
    type: SourceType = SourceType.URL
    url: str


class ReferenceSource(BaseModel):
    type: SourceType = SourceType.REFERENCE
    id: str


MediaSource = Base64Source | UrlSource | ReferenceSource


class TextContent(BaseModel):
    type: ContentType = ContentType.TEXT
    text: str


class MediaContent(BaseModel):
    type: ContentType = ContentType.MEDIA
    media_type: str
    source: MediaSource
    title: str | None = None


class ToolCallContent(BaseModel):
    type: ContentType = ContentType.TOOL_CALL
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResponseContent(BaseModel):
    type: ContentType = ContentType.TOOL_RESPONSE
    tool_call_id: str
    content: str
    is_error: bool = False


class ThinkingContent(BaseModel):
    type: ContentType = ContentType.THINKING
    thinking: str
    signature: str | None = None


class ErrorContent(BaseModel):
    type: ContentType = ContentType.ERROR
    message: str


Content = (
    TextContent
    | MediaContent
    | ToolCallContent
    | ToolResponseContent
    | ThinkingContent
    | ErrorContent
)


class MessageSource(StrEnum):
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    source: MessageSource
    contents: list[Content]
