from enum import StrEnum

from pydantic import BaseModel


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    OPENAI = "openai"


class LLMModel(BaseModel):
    id: str
    name: str
    provider: Provider
