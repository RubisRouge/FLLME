from enum import StrEnum

from pydantic import BaseModel


class AdapterType(StrEnum):
    ANTHROPIC_VERTEX_V1 = "anthropic_vertex_v1"
    ANTHROPIC_VERTEX_V2 = "anthropic_vertex_v2"
    GEMINI_VERTEX_V1 = "gemini_vertex_v1"
    MISTRAL_VERTEX_V1 = "mistral_vertex_v1"
    OPENAI_AZURE_V1 = "openai_azure_v1"
    OPENAI_AZURE_V2 = "openai_azure_v2"
    OPENAI_AZURE_V3 = "openai_azure_v3"


class Deployment(BaseModel):
    id: str
    url: str
    model_id: str
    adapter: AdapterType
    auth_id: str
