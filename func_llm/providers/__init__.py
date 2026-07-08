from ..models.deployment import AdapterType
from .anthropic.vertex_v1 import AnthropicVertexV1
from .base import Adapter
from .gemini.vertex_v1 import GeminiVertexV1
from .mistral.vertex_v1 import MistralVertexV1
from .openai.azure_v1 import OpenAIAzureV1

ADAPTERS: dict[AdapterType, Adapter] = {
    AdapterType.ANTHROPIC_VERTEX_V1: AnthropicVertexV1(),
    AdapterType.GEMINI_VERTEX_V1: GeminiVertexV1(),
    AdapterType.MISTRAL_VERTEX_V1: MistralVertexV1(),
    AdapterType.OPENAI_AZURE_V1: OpenAIAzureV1(),
}


def get_adapter(adapter_type: AdapterType) -> Adapter:
    adapter = ADAPTERS.get(adapter_type)
    if adapter is None:
        msg = f"Unknown adapter type: {adapter_type!r}"
        raise ValueError(msg)
    return adapter


__all__ = [
    "ADAPTERS",
    "Adapter",
    "AdapterType",
    "get_adapter",
]
