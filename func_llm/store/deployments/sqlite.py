from __future__ import annotations

import json
import logging
from pathlib import Path

import aiosqlite

from ...errors import AuthError, DeploymentNotFoundError, ModelNotFoundError
from ...auth.builtins import BUILTIN_PRINCIPLES
from ...auth.types import AuthPrinciple
from ...models.deployment import AdapterType, Deployment
from ...models.model import LLMModel, Provider

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id       TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    provider TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_principles (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    resolver_id      TEXT NOT NULL,
    required_env_vars TEXT NOT NULL DEFAULT '[]',
    config           TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS deployments (
    id       TEXT PRIMARY KEY,
    url      TEXT NOT NULL,
    model_id TEXT NOT NULL REFERENCES models(id),
    adapter  TEXT NOT NULL,
    auth_id  TEXT NOT NULL REFERENCES auth_principles(id)
);
"""


class _SQLiteModelRepository:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def add(self, model: LLMModel) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO models (id, name, provider) VALUES (?, ?, ?)",
            (model.id, model.name, model.provider.value),
        )
        await self._db.commit()

    async def get(self, model_id: str) -> LLMModel:
        cursor = await self._db.execute(
            "SELECT id, name, provider FROM models WHERE id = ?",
            (model_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            msg = f"Model {model_id!r} not found"
            raise ModelNotFoundError(msg)
        return LLMModel(id=row[0], name=row[1], provider=Provider(row[2]))

    async def get_for_provider(self, provider: Provider) -> list[LLMModel]:
        cursor = await self._db.execute(
            "SELECT id, name, provider FROM models WHERE provider = ?",
            (provider.value,),
        )
        rows = await cursor.fetchall()
        return [LLMModel(id=r[0], name=r[1], provider=Provider(r[2])) for r in rows]

    async def list(self) -> list[LLMModel]:
        cursor = await self._db.execute("SELECT id, name, provider FROM models")
        rows = await cursor.fetchall()
        return [LLMModel(id=r[0], name=r[1], provider=Provider(r[2])) for r in rows]

    async def remove(self, model_id: str) -> None:
        await self._db.execute(
            "DELETE FROM deployments WHERE model_id = ?",
            (model_id,),
        )
        await self._db.execute(
            "DELETE FROM models WHERE id = ?",
            (model_id,),
        )
        await self._db.commit()

    async def exists(self, model_id: str) -> bool:
        cursor = await self._db.execute(
            "SELECT 1 FROM models WHERE id = ?",
            (model_id,),
        )
        return await cursor.fetchone() is not None


class _SQLiteDeploymentRepository:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def add(self, deployment: Deployment) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO deployments (id, url, model_id, adapter, auth_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                deployment.id,
                deployment.url,
                deployment.model_id,
                deployment.adapter.value,
                deployment.auth_id,
            ),
        )
        await self._db.commit()

    async def get(self, deployment_id: str) -> Deployment:
        cursor = await self._db.execute(
            "SELECT id, url, model_id, adapter, auth_id FROM deployments WHERE id = ?",
            (deployment_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            msg = f"Deployment {deployment_id!r} not found"
            raise DeploymentNotFoundError(msg)
        return Deployment(
            id=row[0],
            url=row[1],
            model_id=row[2],
            adapter=AdapterType(row[3]),
            auth_id=row[4],
        )

    async def get_for_model(self, model_id: str) -> list[Deployment]:
        cursor = await self._db.execute(
            "SELECT id, url, model_id, adapter, auth_id FROM deployments WHERE model_id = ?",
            (model_id,),
        )
        rows = await cursor.fetchall()
        return [
            Deployment(
                id=r[0],
                url=r[1],
                model_id=r[2],
                adapter=AdapterType(r[3]),
                auth_id=r[4],
            )
            for r in rows
        ]

    async def list(self) -> list[Deployment]:
        cursor = await self._db.execute(
            "SELECT id, url, model_id, adapter, auth_id FROM deployments",
        )
        rows = await cursor.fetchall()
        return [
            Deployment(
                id=r[0],
                url=r[1],
                model_id=r[2],
                adapter=AdapterType(r[3]),
                auth_id=r[4],
            )
            for r in rows
        ]

    async def remove(self, deployment_id: str) -> None:
        await self._db.execute(
            "DELETE FROM deployments WHERE id = ?",
            (deployment_id,),
        )
        await self._db.commit()

    async def exists(self, deployment_id: str) -> bool:
        cursor = await self._db.execute(
            "SELECT 1 FROM deployments WHERE id = ?",
            (deployment_id,),
        )
        return await cursor.fetchone() is not None


class _SQLiteAuthRepository:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def add(self, principle: AuthPrinciple) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO auth_principles "
            "(id, name, resolver_id, required_env_vars, config) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                principle.id,
                principle.name,
                principle.resolver_id,
                json.dumps(principle.required_env_vars),
                json.dumps(principle.config),
            ),
        )
        await self._db.commit()

    async def get(self, auth_id: str) -> AuthPrinciple:
        cursor = await self._db.execute(
            "SELECT id, name, resolver_id, required_env_vars, config "
            "FROM auth_principles WHERE id = ?",
            (auth_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            msg = f"Auth principle {auth_id!r} not found"
            raise AuthError(msg)
        return AuthPrinciple(
            id=row[0],
            name=row[1],
            resolver_id=row[2],
            required_env_vars=json.loads(row[3]),
            config=json.loads(row[4]),
        )

    async def list(self) -> list[AuthPrinciple]:
        cursor = await self._db.execute(
            "SELECT id, name, resolver_id, required_env_vars, config "
            "FROM auth_principles",
        )
        rows = await cursor.fetchall()
        return [
            AuthPrinciple(
                id=r[0],
                name=r[1],
                resolver_id=r[2],
                required_env_vars=json.loads(r[3]),
                config=json.loads(r[4]),
            )
            for r in rows
        ]

    async def remove(self, auth_id: str) -> None:
        await self._db.execute(
            "DELETE FROM auth_principles WHERE id = ?",
            (auth_id,),
        )
        await self._db.commit()

    async def exists(self, auth_id: str) -> bool:
        cursor = await self._db.execute(
            "SELECT 1 FROM auth_principles WHERE id = ?",
            (auth_id,),
        )
        return await cursor.fetchone() is not None


class SQLiteStore:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self.models = _SQLiteModelRepository(db)
        self.deployments = _SQLiteDeploymentRepository(db)
        self.auth = _SQLiteAuthRepository(db)

    @classmethod
    async def create(cls, db_path: str | Path = ":memory:") -> SQLiteStore:
        db = await aiosqlite.connect(str(db_path))
        await db.executescript(SCHEMA)
        store = cls(db)
        await store._seed_builtins()
        return store

    async def _seed_builtins(self) -> None:
        for principle in BUILTIN_PRINCIPLES:
            if not await self.auth.exists(principle.id):
                await self.auth.add(principle)

    async def close(self) -> None:
        await self._db.close()
