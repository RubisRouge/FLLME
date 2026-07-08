from __future__ import annotations

from typing import Protocol

from ..models.auth import AuthPrinciple


class AuthResolver(Protocol):
    async def get_headers(self, principle: AuthPrinciple) -> dict[str, str]: ...

    def check_env(self, principle: AuthPrinciple) -> list[str]: ...
