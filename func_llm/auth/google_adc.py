from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..errors import AuthError

if TYPE_CHECKING:
    from ..models.auth import AuthPrinciple


class GoogleADCResolver:
    def __init__(self) -> None:
        self._credentials: Any | None = None
        self._lock = asyncio.Lock()

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

        async with self._lock:
            if self._credentials is None:
                try:
                    self._credentials, _ = await asyncio.to_thread(
                        google.auth.default,
                    )
                except Exception as exc:
                    msg = f"Failed to obtain Google ADC credentials: {exc}"
                    raise AuthError(msg) from exc

            creds: Any = self._credentials
            if not creds.valid:
                try:
                    auth_req = google.auth.transport.requests.Request()
                    await asyncio.to_thread(creds.refresh, auth_req)
                except Exception as exc:
                    msg = f"Failed to refresh Google ADC credentials: {exc}"
                    raise AuthError(msg) from exc

            token: str = creds.token
        return {"Authorization": f"Bearer {token}"}
        return {"Authorization": f"Bearer {token}"}

    def check_env(self, principle: AuthPrinciple) -> list[str]:
        return []
