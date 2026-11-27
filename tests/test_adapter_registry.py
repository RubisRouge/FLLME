from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import fllme
from fllme.errors import AdapterNotFoundError
from fllme.models.deployment import AdapterType, Deployment
from fllme.models.model import LLMModel, Provider
from fllme.providers import ADAPTERS, get_adapter, register_adapter
from fllme.providers.anthropic.vertex_v2 import AnthropicVertexV2
from fllme.service import DeploymentService
from fllme.store.deployments import SQLiteStore


class _CustomAdapter(AnthropicVertexV2):
    pass


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    snapshot = dict(ADAPTERS)
    yield
    ADAPTERS.clear()
    ADAPTERS.update(snapshot)


def _deployment(adapter: AdapterType | str) -> Deployment:
    return Deployment(
        id="d1",
        url="https://example.com",
        model_id="m1",
        adapter=adapter,
        auth_id="api_key",
    )


async def _service_with_model(db_path: str | Path = ":memory:") -> DeploymentService:
    svc = await DeploymentService.from_sqlite(db_path)
    await svc.add_model(LLMModel(id="m1", name="M1", provider=Provider.ANTHROPIC))
    return svc


class TestRegisterAdapter:
    def test_public_export(self) -> None:
        assert fllme.register_adapter is register_adapter

    @pytest.mark.asyncio
    async def test_custom_adapter_is_resolved_end_to_end(self) -> None:
        custom = _CustomAdapter()
        register_adapter("anthropic_vertex_custom", custom)
        svc = await _service_with_model()
        await svc.add_deployment(_deployment("anthropic_vertex_custom"))

        deployment = await svc.resolve_deployment("m1")

        assert deployment.adapter == "anthropic_vertex_custom"
        assert get_adapter(deployment.adapter) is custom

    @pytest.mark.asyncio
    async def test_custom_id_survives_sqlite_roundtrip(self, tmp_path: Path) -> None:
        register_adapter("anthropic_vertex_custom", _CustomAdapter())
        db_path = tmp_path / "d.db"
        svc = await _service_with_model(db_path)
        await svc.add_deployment(_deployment("anthropic_vertex_custom"))

        store = await SQLiteStore.create(db_path)
        try:
            got = await store.deployments.get("d1")
            assert got.adapter == "anthropic_vertex_custom"
            assert not isinstance(got.adapter, AdapterType)
        finally:
            await store.close()

    def test_unknown_adapter_raises_explicit_error(self) -> None:
        with pytest.raises(AdapterNotFoundError, match="no_such_adapter"):
            get_adapter("no_such_adapter")

    def test_unknown_adapter_error_is_still_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("no_such_adapter")

    @pytest.mark.asyncio
    async def test_unknown_id_fails_at_lookup_not_at_storage(self) -> None:
        # The adapter may be registered after the deployment is stored.
        svc = await _service_with_model()
        await svc.add_deployment(_deployment("registered_later"))
        deployment = await svc.resolve_deployment("m1")
        with pytest.raises(AdapterNotFoundError):
            get_adapter(deployment.adapter)
        register_adapter("registered_later", _CustomAdapter())
        assert isinstance(get_adapter(deployment.adapter), _CustomAdapter)

    def test_registering_builtin_id_replaces_it(self) -> None:
        custom = _CustomAdapter()
        register_adapter(AdapterType.ANTHROPIC_VERTEX_V2, custom)
        assert get_adapter(AdapterType.ANTHROPIC_VERTEX_V2) is custom


class TestBuiltinAdaptersUnchanged:
    @pytest.mark.parametrize("adapter_type", list(AdapterType))
    def test_every_builtin_is_registered(self, adapter_type: AdapterType) -> None:
        assert get_adapter(adapter_type) is ADAPTERS[adapter_type]
        assert get_adapter(adapter_type.value) is ADAPTERS[adapter_type]

    @pytest.mark.parametrize("adapter_type", list(AdapterType))
    def test_builtin_string_is_typed_as_enum(self, adapter_type: AdapterType) -> None:
        dep = _deployment(adapter_type.value)
        assert isinstance(dep.adapter, AdapterType)
        assert dep.adapter is adapter_type

    @pytest.mark.asyncio
    async def test_builtin_sqlite_roundtrip(self) -> None:
        svc = await _service_with_model()
        await svc.add_deployment(_deployment(AdapterType.OPENAI_AZURE_V3))
        got = await svc.get_deployment("d1")
        assert got.adapter is AdapterType.OPENAI_AZURE_V3
