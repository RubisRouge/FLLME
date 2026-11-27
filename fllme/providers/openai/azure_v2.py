from __future__ import annotations

from typing import Any

from ...models.input import GenerationInput
from ...models.model import LLMModel
from .azure_v1 import OpenAIAzureV1


class OpenAIAzureV2(OpenAIAzureV1):
    """OpenAI Azure adapter that omits sampling params (temperature, top_p).

    Use for models that reject these params (e.g. GPT-5+).
    """

    def serialize(
        self,
        gen_input: GenerationInput,
        model: LLMModel | None = None,
    ) -> dict[str, Any]:
        payload = super().serialize(gen_input, model)
        payload.pop("temperature", None)
        payload.pop("top_p", None)
        return payload
