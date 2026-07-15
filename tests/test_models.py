import pytest
from pydantic import ValidationError

from fllm.models.auth import BUILTIN_PRINCIPLES, AuthPrinciple
from fllm.models.deployment import AdapterType, Deployment
from fllm.models.model import LLMModel, Provider


class TestLLMModel:
    def test_create(self) -> None:
        model = LLMModel(
            id="claude-sonnet-4",
            name="Claude Sonnet 4",
            provider=Provider.ANTHROPIC,
        )
        assert model.id == "claude-sonnet-4"
        assert model.name == "Claude Sonnet 4"
        assert model.provider == Provider.ANTHROPIC

    def test_missing_field(self) -> None:
        with pytest.raises(ValidationError):
            LLMModel(id="test")  # type: ignore[call-arg]

    def test_serialization(self) -> None:
        model = LLMModel(id="test", name="Test", provider=Provider.GEMINI)
        data = model.model_dump()
        assert data == {"id": "test", "name": "Test", "provider": "gemini"}
        assert LLMModel.model_validate(data) == model


class TestAdapterType:
    def test_values(self) -> None:
        assert AdapterType.ANTHROPIC_VERTEX_V2.value == "anthropic_vertex_v2"
        assert AdapterType.ANTHROPIC_VERTEX_V1.value == "anthropic_vertex_v1"
        assert AdapterType.GEMINI_VERTEX_V1.value == "gemini_vertex_v1"
        assert AdapterType.MISTRAL_VERTEX_V1.value == "mistral_vertex_v1"
        assert AdapterType.OPENAI_AZURE_V1.value == "openai_azure_v1"
        assert AdapterType.OPENAI_AZURE_V2.value == "openai_azure_v2"

    def test_from_string(self) -> None:
        assert AdapterType("anthropic_vertex_v1") == AdapterType.ANTHROPIC_VERTEX_V1


class TestDeployment:
    def test_create(self) -> None:
        dep = Deployment(
            id="claude-vertex-euw1",
            url="https://europe-west1-aiplatform.googleapis.com/v1/...",
            model_id="claude-sonnet-4",
            adapter=AdapterType.ANTHROPIC_VERTEX_V1,
            auth_id="google_adc",
        )
        assert dep.id == "claude-vertex-euw1"
        assert dep.adapter == AdapterType.ANTHROPIC_VERTEX_V1
        assert dep.auth_id == "google_adc"

    def test_serialization_roundtrip(self) -> None:
        dep = Deployment(
            id="test",
            url="https://example.com",
            model_id="m1",
            adapter=AdapterType.OPENAI_AZURE_V1,
            auth_id="api_key",
        )
        data = dep.model_dump()
        restored = Deployment.model_validate(data)
        assert restored == dep


class TestAuthPrinciple:
    def test_defaults(self) -> None:
        principle = AuthPrinciple(
            id="test",
            name="Test",
            resolver_id="test_resolver",
        )
        assert principle.required_env_vars == []
        assert principle.config == {}

    def test_with_env_vars(self) -> None:
        principle = AuthPrinciple(
            id="azure_key",
            name="Azure API Key",
            resolver_id="api_key",
            required_env_vars=["AZURE_OPENAI_KEY"],
            config={"header_name": "api-key"},
        )
        assert principle.required_env_vars == ["AZURE_OPENAI_KEY"]
        assert principle.config["header_name"] == "api-key"

    def test_builtin_principles_exist(self) -> None:
        assert len(BUILTIN_PRINCIPLES) == 2
        ids = {p.id for p in BUILTIN_PRINCIPLES}
        assert "google_adc" in ids
        assert "api_key" in ids
