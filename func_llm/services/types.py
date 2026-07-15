from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import AsyncIterator

from ..auth.types import AuthPrinciple
from ..auth.types import AuthResolver
from ..errors import DeploymentNotFoundError
from ..media import MediaResolver
from ..models.input import GenerationInput
from ..models.output import GenerationOutput
from ..models.output import StreamDelta
from ..models.deployment import Deployment
from ..store import AuthRepository
from ..store import DeploymentRepository
from ..store import ModelRepository

logger = logging.getLogger(__name__)


class DeploymentService(ABC):
    def __init__(
        self,
        auth: AuthRepository,
        deployments: DeploymentRepository,
        models: ModelRepository,
    ) -> None:
        self._auth = auth
        self._deployments = deployments
        self._models = models

    async def close(self) -> None:
        pass

    @abstractmethod
    async def generate(
        self,
        input: GenerationInput,
        *,
        auth_resolver: AuthResolver[...] | None = None,
        deployment_id: str | None = None,
        media_resolver: MediaResolver | None = None,
    ) -> GenerationOutput | AsyncIterator[StreamDelta | GenerationOutput]:
        pass

    @property
    def models(self) -> ModelRepository:
        return self._models

    async def add_auth_principle(self, principle: AuthPrinciple) -> None:
        await self._auth.add(principle)

    async def get_deployment(self, deployment_id: str) -> Deployment:
        return await self._deployments.get(deployment_id)

    async def get_deployment_for_model(self, model_id: str) -> list[Deployment]:
        return await self._deployments.get_for_model(model_id)

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

    async def resolve_deployment(
        self,
        model_id: str,
        deployment_id: str | None = None,
    ) -> Deployment:
        if deployment_id is not None:
            return await self._deployments.get(deployment_id)
        deployments = await self._deployments.get_for_model(model_id)
        if not deployments:
            raise DeploymentNotFoundError(
                f"No deployments found for model `{model_id!r}`"
            )
        return deployments[0]
