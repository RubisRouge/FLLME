from __future__ import annotations

from typing import Any

from ...models.input import GenerationInput
from .azure_v2 import OpenAIAzureV2


class OpenAIAzureV3(OpenAIAzureV2):
    """Azure OpenAI adapter for the v1 unified endpoint.

    Uses ``/openai/v1/chat/completions`` where the deployment name is passed
    as the ``model`` field in the request body instead of the URL path.
    Inherits V2 behaviour (no temperature / top_p).
    """

    def serialize(self, gen_input: GenerationInput) -> dict[str, Any]:
        payload = super().serialize(gen_input)
        payload["model"] = gen_input.model
        return payload
