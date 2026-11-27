from enum import StrEnum

from pydantic import BaseModel, field_validator


class AdapterType(StrEnum):
    ANTHROPIC_VERTEX_V1 = "anthropic_vertex_v1"
    ANTHROPIC_VERTEX_V2 = "anthropic_vertex_v2"
    GEMINI_VERTEX_V1 = "gemini_vertex_v1"
    MISTRAL_VERTEX_V1 = "mistral_vertex_v1"
    OPENAI_AZURE_V1 = "openai_azure_v1"
    OPENAI_AZURE_V2 = "openai_azure_v2"
    OPENAI_AZURE_V3 = "openai_azure_v3"
    OPENAI_AZURE_RESPONSES_V1 = "openai_azure_responses_v1"


class Deployment(BaseModel):
    id: str
    url: str
    model_id: str
    # A built-in AdapterType, or the id of an adapter added with register_adapter().
    adapter: AdapterType | str
    auth_id: str

    @field_validator("adapter", mode="before")
    @classmethod
    def _known_adapter_as_enum(cls, value: object) -> object:
        # Keep built-in ids typed as AdapterType; leave custom ids as plain str.
        if isinstance(value, str):
            try:
                return AdapterType(value)
            except ValueError:
                return value
        return value
