from pathlib import Path

from httpx import AsyncClient

from .http import HTTPDeploymentService
from .types import DeploymentService
from ..store import SQLiteStore


async def create_default_deployment_service(
    *,
    sqllite_path: str | Path = ":memory:",
) -> DeploymentService:
    store = await SQLiteStore.create(sqllite_path)
    transport = AsyncClient()
    return HTTPDeploymentService(
        store.auth,
        store.deployments,
        store.models,
        transport,
    )
