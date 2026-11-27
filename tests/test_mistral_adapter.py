from __future__ import annotations

import pytest

from fllme.models.input import GenerationInput, LLMConfig, ThinkingLevel
from fllme.models.message import Message, MessageSource, TextContent
from fllme.models.model import LLMModel, ModelCapabilities, Provider, ThinkingStyle
from fllme.providers.mistral.vertex_v1 import MistralVertexV1


def _input(thinking: ThinkingLevel) -> GenerationInput:
    return GenerationInput(
        model="mistral",
        conversation=[
            Message(source=MessageSource.USER, contents=[TextContent(text="Hi")])
        ],
        llm_config=LLMConfig(thinking=thinking),
    )


def _model(
    style: ThinkingStyle, efforts: list[ThinkingLevel] | None = None
) -> LLMModel:
    return LLMModel(
        id="mistral",
        name="Mistral",
        provider=Provider.MISTRAL,
        capabilities=ModelCapabilities(thinking_style=style, thinking_efforts=efforts),
    )


class TestReasoningEffort:
    @pytest.mark.parametrize(
        "level", [ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH]
    )
    def test_no_reasoning_model_omits_key(self, level: ThinkingLevel) -> None:
        payload = MistralVertexV1().serialize(_input(level), _model(ThinkingStyle.NONE))
        assert "reasoning_effort" not in payload

    @pytest.mark.parametrize(
        "level", [ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH]
    )
    def test_effort_model_only_gets_high(self, level: ThinkingLevel) -> None:
        payload = MistralVertexV1().serialize(
            _input(level), _model(ThinkingStyle.EFFORT)
        )
        assert payload["reasoning_effort"] == "high"

    def test_without_capabilities_only_gets_high(self) -> None:
        payload = MistralVertexV1().serialize(_input(ThinkingLevel.LOW))
        assert payload["reasoning_effort"] == "high"

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (ThinkingLevel.LOW, "low"),
            (ThinkingLevel.MEDIUM, "high"),
            (ThinkingLevel.HIGH, "high"),
        ],
    )
    def test_closest_accepted_effort(self, level: ThinkingLevel, expected: str) -> None:
        model = _model(ThinkingStyle.EFFORT, [ThinkingLevel.LOW, ThinkingLevel.HIGH])
        payload = MistralVertexV1().serialize(_input(level), model)
        assert payload["reasoning_effort"] == expected

    def test_no_thinking_omits_key(self) -> None:
        payload = MistralVertexV1().serialize(
            _input(ThinkingLevel.NO), _model(ThinkingStyle.EFFORT)
        )
        assert "reasoning_effort" not in payload
