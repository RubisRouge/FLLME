from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from func_llm.errors import AuthError, DeploymentNotFoundError, ModelNotFoundError
from func_llm.models.auth import AuthPrinciple
from func_llm.models.deployment import AdapterType, Deployment
from func_llm.models.model import LLMModel
from func_llm.store.deployments import SQLiteStore


@pytest_asyncio.fixture
async def store() -> AsyncGenerator[SQLiteStore]:
    s = await SQLiteStore.create(":memory:")
    yield s
    await s.close()


@pytest.fixture
def sample_model() -> LLMModel:
    return LLMModel(id="claude-sonnet-4", name="Claude Sonnet 4")


@pytest.fixture
def sample_deployment() -> Deployment:
    return Deployment(
        id="claude-vertex-euw1",
        url="https://europe-west1-aiplatform.googleapis.com/v1/test",
        model_id="claude-sonnet-4",
        adapter=AdapterType.ANTHROPIC_VERTEX_V1,
        auth_id="google_adc",
    )


class TestModelRepository:
    @pytest.mark.asyncio
    async def test_add_and_get(
        self, store: SQLiteStore, sample_model: LLMModel
    ) -> None:
        await store.models.add(sample_model)
        result = await store.models.get("claude-sonnet-4")
        assert result == sample_model

    @pytest.mark.asyncio
    async def test_get_not_found(self, store: SQLiteStore) -> None:
        with pytest.raises(ModelNotFoundError):
            await store.models.get("nonexistent")

    @pytest.mark.asyncio
    async def test_list(self, store: SQLiteStore, sample_model: LLMModel) -> None:
        await store.models.add(sample_model)
        await store.models.add(LLMModel(id="gemini-2", name="Gemini 2"))
        models = await store.models.list()
        assert len(models) == 2

    @pytest.mark.asyncio
    async def test_exists(self, store: SQLiteStore, sample_model: LLMModel) -> None:
        assert not await store.models.exists("claude-sonnet-4")
        await store.models.add(sample_model)
        assert await store.models.exists("claude-sonnet-4")

    @pytest.mark.asyncio
    async def test_remove(self, store: SQLiteStore, sample_model: LLMModel) -> None:
        await store.models.add(sample_model)
        await store.models.remove("claude-sonnet-4")
        assert not await store.models.exists("claude-sonnet-4")

    @pytest.mark.asyncio
    async def test_upsert(self, store: SQLiteStore, sample_model: LLMModel) -> None:
        await store.models.add(sample_model)
        updated = LLMModel(id="claude-sonnet-4", name="Claude Sonnet 4 Updated")
        await store.models.add(updated)
        result = await store.models.get("claude-sonnet-4")
        assert result.name == "Claude Sonnet 4 Updated"


class TestDeploymentRepository:
    @pytest.mark.asyncio
    async def test_add_and_get(
        self,
        store: SQLiteStore,
        sample_model: LLMModel,
        sample_deployment: Deployment,
    ) -> None:
        await store.models.add(sample_model)
        await store.deployments.add(sample_deployment)
        result = await store.deployments.get("claude-vertex-euw1")
        assert result == sample_deployment

    @pytest.mark.asyncio
    async def test_get_not_found(self, store: SQLiteStore) -> None:
        with pytest.raises(DeploymentNotFoundError):
            await store.deployments.get("nonexistent")

    @pytest.mark.asyncio
    async def test_get_for_model(
        self,
        store: SQLiteStore,
        sample_model: LLMModel,
        sample_deployment: Deployment,
    ) -> None:
        await store.models.add(sample_model)
        await store.deployments.add(sample_deployment)

        dep2 = Deployment(
            id="claude-vertex-usw1",
            url="https://us-west1-aiplatform.googleapis.com/v1/test",
            model_id="claude-sonnet-4",
            adapter=AdapterType.ANTHROPIC_VERTEX_V1,
            auth_id="google_adc",
        )
        await store.deployments.add(dep2)

        deployments = await store.deployments.get_for_model("claude-sonnet-4")
        assert len(deployments) == 2

    @pytest.mark.asyncio
    async def test_get_for_model_empty(self, store: SQLiteStore) -> None:
        deployments = await store.deployments.get_for_model("nonexistent")
        assert deployments == []

    @pytest.mark.asyncio
    async def test_list(
        self,
        store: SQLiteStore,
        sample_model: LLMModel,
        sample_deployment: Deployment,
    ) -> None:
        await store.models.add(sample_model)
        await store.deployments.add(sample_deployment)
        deployments = await store.deployments.list()
        assert len(deployments) == 1

    @pytest.mark.asyncio
    async def test_remove(
        self,
        store: SQLiteStore,
        sample_model: LLMModel,
        sample_deployment: Deployment,
    ) -> None:
        await store.models.add(sample_model)
        await store.deployments.add(sample_deployment)
        await store.deployments.remove("claude-vertex-euw1")
        assert not await store.deployments.exists("claude-vertex-euw1")

    @pytest.mark.asyncio
    async def test_cascade_on_model_remove(
        self,
        store: SQLiteStore,
        sample_model: LLMModel,
        sample_deployment: Deployment,
    ) -> None:
        await store.models.add(sample_model)
        await store.deployments.add(sample_deployment)
        await store.models.remove("claude-sonnet-4")
        assert not await store.deployments.exists("claude-vertex-euw1")


class TestAuthRepository:
    @pytest.mark.asyncio
    async def test_builtin_principles_seeded(self, store: SQLiteStore) -> None:
        assert await store.auth.exists("google_adc")
        assert await store.auth.exists("api_key")

    @pytest.mark.asyncio
    async def test_get_builtin(self, store: SQLiteStore) -> None:
        principle = await store.auth.get("google_adc")
        assert principle.resolver_id == "google_adc"
        assert principle.required_env_vars == []

    @pytest.mark.asyncio
    async def test_add_custom(self, store: SQLiteStore) -> None:
        custom = AuthPrinciple(
            id="my_oauth",
            name="My OAuth",
            resolver_id="oauth2",
            required_env_vars=["OAUTH_CLIENT_ID", "OAUTH_SECRET"],
            config={"token_url": "https://auth.example.com/token"},
        )
        await store.auth.add(custom)
        result = await store.auth.get("my_oauth")
        assert result == custom

    @pytest.mark.asyncio
    async def test_get_not_found(self, store: SQLiteStore) -> None:
        with pytest.raises(AuthError):
            await store.auth.get("nonexistent")

    @pytest.mark.asyncio
    async def test_list(self, store: SQLiteStore) -> None:
        principles = await store.auth.list()
        assert len(principles) >= 2

    @pytest.mark.asyncio
    async def test_remove(self, store: SQLiteStore) -> None:
        custom = AuthPrinciple(
            id="temp",
            name="Temp",
            resolver_id="test",
        )
        await store.auth.add(custom)
        await store.auth.remove("temp")
        assert not await store.auth.exists("temp")
