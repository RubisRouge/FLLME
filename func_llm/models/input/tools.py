from pydantic import BaseModel
from enum import StrEnum
from typing import Any


class ToolsCallingMode(StrEnum):
    AUTO = "auto"
    ANY = "any"
    NONE = "none"


class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolsConfig(BaseModel):
    tools: list[Tool]
    parallel_calling: bool
    mode: ToolsCallingMode
