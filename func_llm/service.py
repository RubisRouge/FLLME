from __future__ import annotations

import logging
from pathlib import Path

from .auth import get_resolver
from .errors import DeploymentNotFoundError
from .models.auth import AuthPrinciple
from .models.deployment import Deployment
from .models.model import LLMModel
from .store import AuthRepository, DeploymentRepository, ModelRepository, SQLiteStore

logger = logging.getLogger(__name__)


class DeploymentService:
    def __init__(
        self,
        models: ModelRepository,
        deployments: DeploymentRepository,
        auth: AuthRepository,
    ) -> None:
        self._models = models
        self._deployments = deployments
        self._auth = auth

    @classmethod
    async def from_sqlite(
        cls,
        db_path: str | Path = ":memory:",
    ) -> DeploymentService:
        store = await SQLiteStore.create(db_path)
        return cls(
            models=store.models,
            deployments=store.deployments,
            auth=store.auth,
        )

    async def add_model(self, model: LLMModel) -> None:
        await self._models.add(model)

    async def get_model(self, model_id: str) -> LLMModel:
        return await self._models.get(model_id)

    async def list_models(self) -> list[LLMModel]:
        return await self._models.list()

    async def remove_model(self, model_id: str) -> None:
        await self._models.remove(model_id)

    async def add_auth_principle(self, principle: AuthPrinciple) -> None:
        await self._auth.add(principle)

    async def add_deployment(self, deployment: Deployment) -> None:
        if not await self._models.exists(deployment.model_id):
            msg = f"Model {deployment.model_id!r} does not exist"
            raise DeploymentNotFoundError(msg)

        if not await self._auth.exists(deployment.auth_id):
            msg = f"Auth principle {deployment.auth_id!r} does not exist"
            raise DeploymentNotFoundError(msg)

        issues = await self.check_deployment_ready(deployment)
        if issues:
            for issue in issues:
                logger.warning("Deployment %r: %s", deployment.id, issue)

        await self._deployments.add(deployment)

    async def get_deployment(self, deployment_id: str) -> Deployment:
        return await self._deployments.get(deployment_id)

    async def resolve_deployment(
        self,
        model_id: str,
        deployment_id: str | None = None,
    ) -> Deployment:
        if deployment_id is not None:
            return await self._deployments.get(deployment_id)

        deployments = await self._deployments.get_for_model(model_id)
        if not deployments:
            msg = f"No deployments found for model {model_id!r}"
            raise DeploymentNotFoundError(msg)
        return deployments[0]

    async def get_auth_headers(self, deployment: Deployment) -> dict[str, str]:
        principle = await self._auth.get(deployment.auth_id)
        resolver = get_resolver(principle.resolver_id)
        return await resolver.get_headers(principle)

    async def check_deployment_ready(self, deployment: Deployment) -> list[str]:
        issues: list[str] = []

        if not await self._auth.exists(deployment.auth_id):
            issues.append(f"Auth principle {deployment.auth_id!r} not found")
            return issues

        principle = await self._auth.get(deployment.auth_id)
        resolver = get_resolver(principle.resolver_id)
        missing = resolver.check_env(principle)
        for var in missing:
            issues.append(f"Missing environment variable: {var}")

        return issues
