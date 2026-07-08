from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..errors import AuthError

if TYPE_CHECKING:
    from ..models.auth import AuthPrinciple


class ApiKeyResolver:
    async def get_headers(self, principle: AuthPrinciple) -> dict[str, str]:
        if not principle.required_env_vars:
            msg = "API key auth requires at least one env var in required_env_vars"
            raise AuthError(msg)

        env_var = principle.required_env_vars[0]
        value = os.environ.get(env_var)
        if value is None:
            msg = f"Missing required environment variable: {env_var}"
            raise AuthError(msg)

        header_name = principle.config.get("header_name", "api-key")
        return {header_name: value}

    def check_env(self, principle: AuthPrinciple) -> list[str]:
        missing: list[str] = []
        for var in principle.required_env_vars:
            if var not in os.environ:
                missing.append(var)
        return missing
