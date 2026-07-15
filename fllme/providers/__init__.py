from ..models.deployment import AdapterType
from .anthropic.vertex_v1 import AnthropicVertexV1
from .anthropic.vertex_v2 import AnthropicVertexV2
from .base import Adapter
from .gemini.vertex_v1 import GeminiVertexV1
from .mistral.vertex_v1 import MistralVertexV1
from .openai.azure_v1 import OpenAIAzureV1
from .openai.azure_v2 import OpenAIAzureV2
from .openai.azure_v3 import OpenAIAzureV3

ADAPTERS: dict[AdapterType, Adapter] = {
    AdapterType.ANTHROPIC_VERTEX_V1: AnthropicVertexV1(),
    AdapterType.ANTHROPIC_VERTEX_V2: AnthropicVertexV2(),
    AdapterType.GEMINI_VERTEX_V1: GeminiVertexV1(),
    AdapterType.MISTRAL_VERTEX_V1: MistralVertexV1(),
    AdapterType.OPENAI_AZURE_V1: OpenAIAzureV1(),
    AdapterType.OPENAI_AZURE_V2: OpenAIAzureV2(),
    AdapterType.OPENAI_AZURE_V3: OpenAIAzureV3(),
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
