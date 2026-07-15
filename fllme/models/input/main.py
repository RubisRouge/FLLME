from pydantic import BaseModel
from enum import StrEnum

from .image import ImageConfig
from .tools import ToolsConfig
from ..message import Message


class ThinkingLevel(StrEnum):
    NO = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LLMConfig(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int = 1024
    stop: list[str] = []
    candidates: int = 1
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    thinking: ThinkingLevel = ThinkingLevel.NO


class BasicOutputType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    HYBRID = "hybrid"


OutputType = BasicOutputType | type[BaseModel]


class GenerationInput(BaseModel):
    model: str
    conversation: list[Message]
    llm_config: LLMConfig = LLMConfig()
    system_prompt: str | None = None
    stream: bool = False
    tool_config: ToolsConfig | None = None
    image_config: ImageConfig | None = None
    output_type: BasicOutputType = BasicOutputType.TEXT
