from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import AuthError
from .deprecated.api_key import ApiKeyResolver
from .deprecated.google_adc import GoogleADCResolver

if TYPE_CHECKING:
    pass

RESOLVERS: dict[str, AuthResolver] = {}


def _register_builtins() -> None:
    register_resolver("google_adc", GoogleADCResolver())
    register_resolver("api_key", ApiKeyResolver())


def register_resolver(resolver_id: str, resolver: AuthResolver) -> None:
    RESOLVERS[resolver_id] = resolver


def get_resolver(resolver_id: str) -> AuthResolver:
    resolver = RESOLVERS.get(resolver_id)
    if resolver is None:
        msg = f"Unknown auth resolver: {resolver_id!r}"
        raise AuthError(msg)
    return resolver


_register_builtins()

__all__ = [
    "RESOLVERS",
    "ApiKeyResolver",
    "AuthResolver",
    "GoogleADCResolver",
    "get_resolver",
    "register_resolver",
]
