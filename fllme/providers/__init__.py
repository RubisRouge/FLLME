from ..errors import AdapterNotFoundError
from ..models.deployment import AdapterType
from .anthropic.vertex_v1 import AnthropicVertexV1
from .anthropic.vertex_v2 import AnthropicVertexV2
from .base import Adapter
from .gemini.vertex_v1 import GeminiVertexV1
from .mistral.vertex_v1 import MistralVertexV1
from .openai.azure_responses_v1 import OpenAIAzureResponsesV1
from .openai.azure_v1 import OpenAIAzureV1
from .openai.azure_v2 import OpenAIAzureV2
from .openai.azure_v3 import OpenAIAzureV3

ADAPTERS: dict[str, Adapter] = {
    AdapterType.ANTHROPIC_VERTEX_V1: AnthropicVertexV1(),
    AdapterType.ANTHROPIC_VERTEX_V2: AnthropicVertexV2(),
    AdapterType.GEMINI_VERTEX_V1: GeminiVertexV1(),
    AdapterType.MISTRAL_VERTEX_V1: MistralVertexV1(),
    AdapterType.OPENAI_AZURE_V1: OpenAIAzureV1(),
    AdapterType.OPENAI_AZURE_V2: OpenAIAzureV2(),
    AdapterType.OPENAI_AZURE_V3: OpenAIAzureV3(),
    AdapterType.OPENAI_AZURE_RESPONSES_V1: OpenAIAzureResponsesV1(),
}


def register_adapter(adapter_id: str, adapter: Adapter) -> None:
    """Register an adapter under ``adapter_id`` so deployments can reference it.

    Registering a built-in id (an ``AdapterType`` value) replaces that adapter
    for every deployment using it; use a new id to affect only your own.
    """
    ADAPTERS[adapter_id] = adapter


def get_adapter(adapter_id: AdapterType | str) -> Adapter:
    adapter = ADAPTERS.get(adapter_id)
    if adapter is None:
        known = ", ".join(sorted(ADAPTERS))
        msg = f"Unknown adapter: {str(adapter_id)!r} (registered: {known})"
        raise AdapterNotFoundError(msg)
    return adapter


__all__ = [
    "ADAPTERS",
    "Adapter",
    "AdapterType",
    "get_adapter",
    "register_adapter",
]
