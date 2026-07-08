from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..errors import AuthError

if TYPE_CHECKING:
    from ..models.auth import AuthPrinciple


class GoogleADCResolver:
    async def get_headers(self, principle: AuthPrinciple) -> dict[str, str]:
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as exc:
            msg = (
                "google-auth is required for Google ADC authentication. "
                "Install it with: pip install google-auth"
            )
            raise AuthError(msg) from exc

        try:
            credentials, _ = await asyncio.to_thread(google.auth.default)
            auth_req = google.auth.transport.requests.Request()
            await asyncio.to_thread(credentials.refresh, auth_req)
        except Exception as exc:
            msg = f"Failed to obtain Google ADC credentials: {exc}"
            raise AuthError(msg) from exc

        token = str(credentials.token)
        return {"Authorization": f"Bearer {token}"}

    def check_env(self, principle: AuthPrinciple) -> list[str]:
        return []
