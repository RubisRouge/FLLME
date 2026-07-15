from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio

from fllm.errors import DeploymentNotFoundError
from fllm.models.auth import AuthPrinciple
from fllm.models.deployment import AdapterType, Deployment
from fllm.models.model import LLMModel, Provider
from fllm.service import DeploymentService
from fllm.store.deployments.sqlite import SQLiteStore


@pytest_asyncio.fixture
async def service() -> AsyncGenerator[DeploymentService]:
    store = await SQLiteStore.create(":memory:")
    svc = DeploymentService(
        models=store.models,
        deployments=store.deployments,
        auth=store.auth,
    )
    yield svc
    await store.close()


@pytest.fixture
def sample_model() -> LLMModel:
    return LLMModel(
        id="claude-sonnet-4",
        name="Claude Sonnet 4",
        provider=Provider.ANTHROPIC,
    )


@pytest.fixture
def sample_deployment() -> Deployment:
    return Deployment(
        id="claude-vertex-euw1",
        url="https://europe-west1-aiplatform.googleapis.com/v1/test",
        model_id="claude-sonnet-4",
        adapter=AdapterType.ANTHROPIC_VERTEX_V1,
        auth_id="google_adc",
    )


class TestDeploymentService:
    @pytest.mark.asyncio
    async def test_add_and_resolve_deployment(
        self,
        service: DeploymentService,
        sample_model: LLMModel,
        sample_deployment: Deployment,
    ) -> None:
        await service.add_model(sample_model)
        await service.add_deployment(sample_deployment)
        result = await service.resolve_deployment("claude-sonnet-4")
        assert result == sample_deployment

    @pytest.mark.asyncio
    async def test_resolve_by_deployment_id(
        self,
        service: DeploymentService,
        sample_model: LLMModel,
        sample_deployment: Deployment,
    ) -> None:
        await service.add_model(sample_model)
        await service.add_deployment(sample_deployment)
        result = await service.resolve_deployment(
            "claude-sonnet-4",
            deployment_id="claude-vertex-euw1",
        )
        assert result.id == "claude-vertex-euw1"

    @pytest.mark.asyncio
    async def test_resolve_no_deployments(
        self,
        service: DeploymentService,
        sample_model: LLMModel,
    ) -> None:
        await service.add_model(sample_model)
        with pytest.raises(DeploymentNotFoundError, match="No deployments found"):
            await service.resolve_deployment("claude-sonnet-4")

    @pytest.mark.asyncio
    async def test_add_deployment_missing_model(
        self,
        service: DeploymentService,
        sample_deployment: Deployment,
    ) -> None:
        with pytest.raises(DeploymentNotFoundError, match="does not exist"):
            await service.add_deployment(sample_deployment)

    @pytest.mark.asyncio
    async def test_add_deployment_missing_auth(
        self,
        service: DeploymentService,
        sample_model: LLMModel,
    ) -> None:
        await service.add_model(sample_model)
        dep = Deployment(
            id="test",
            url="https://example.com",
            model_id="claude-sonnet-4",
            adapter=AdapterType.ANTHROPIC_VERTEX_V1,
            auth_id="nonexistent_auth",
        )
        with pytest.raises(DeploymentNotFoundError, match="does not exist"):
            await service.add_deployment(dep)

    @pytest.mark.asyncio
    async def test_check_deployment_ready_missing_env(
        self,
        service: DeploymentService,
        sample_model: LLMModel,
    ) -> None:
        await service.add_model(sample_model)

        auth = AuthPrinciple(
            id="azure_key",
            name="Azure Key",
            resolver_id="api_key",
            required_env_vars=["AZURE_KEY"],
            config={"header_name": "api-key"},
        )
        await service.add_auth_principle(auth)

        dep = Deployment(
            id="test",
            url="https://example.com",
            model_id="claude-sonnet-4",
            adapter=AdapterType.OPENAI_AZURE_V1,
            auth_id="azure_key",
        )

        with patch.dict("os.environ", {}, clear=True):
            issues = await service.check_deployment_ready(dep)
        assert any("AZURE_KEY" in issue for issue in issues)

    @pytest.mark.asyncio
    async def test_check_deployment_ready_all_good(
        self,
        service: DeploymentService,
        sample_model: LLMModel,
        sample_deployment: Deployment,
    ) -> None:
        await service.add_model(sample_model)
        issues = await service.check_deployment_ready(sample_deployment)
        assert issues == []

    @pytest.mark.asyncio
    async def test_from_sqlite(self) -> None:
        svc = await DeploymentService.from_sqlite(":memory:")
        models = await svc.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models(
        self,
        service: DeploymentService,
        sample_model: LLMModel,
    ) -> None:
        await service.add_model(sample_model)
        models = await service.list_models()
        assert len(models) == 1
        assert models[0] == sample_model

    @pytest.mark.asyncio
    async def test_remove_model(
        self,
        service: DeploymentService,
        sample_model: LLMModel,
    ) -> None:
        await service.add_model(sample_model)
        await service.remove_model("claude-sonnet-4")
        models = await service.list_models()
        assert models == []
