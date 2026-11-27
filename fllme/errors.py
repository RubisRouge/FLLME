from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models.output import GenerationOutput


class FuncLLMError(Exception):
    pass


class ModelNotFoundError(FuncLLMError):
    pass


class DeploymentNotFoundError(FuncLLMError):
    pass


class AdapterNotFoundError(FuncLLMError, ValueError):
    pass


class AuthError(FuncLLMError):
    pass


class ProviderErrorKind(StrEnum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    POLICY = "policy"
    UNKNOWN = "unknown"


# Vertex / Google APIs report a canonical status next to the HTTP code.
_GOOGLE_STATUS_KIND: dict[str, ProviderErrorKind] = {
    "INVALID_ARGUMENT": ProviderErrorKind.INVALID_REQUEST,
    "FAILED_PRECONDITION": ProviderErrorKind.INVALID_REQUEST,
    "OUT_OF_RANGE": ProviderErrorKind.INVALID_REQUEST,
    "UNAUTHENTICATED": ProviderErrorKind.AUTHENTICATION,
    "PERMISSION_DENIED": ProviderErrorKind.PERMISSION,
    "NOT_FOUND": ProviderErrorKind.NOT_FOUND,
    "RESOURCE_EXHAUSTED": ProviderErrorKind.RATE_LIMITED,
    "UNAVAILABLE": ProviderErrorKind.UNAVAILABLE,
    "DEADLINE_EXCEEDED": ProviderErrorKind.UNAVAILABLE,
    "INTERNAL": ProviderErrorKind.UNAVAILABLE,
}

# Markers of a refusal by policy (organisation policy, content filter, ...)
# rather than a malformed request or missing permission.
_POLICY_MARKERS = (
    "constraints/",
    "organization policy",
    "org policy",
    "content_policy_violation",
    "responsibleaipolicyviolation",
    "content_filter",
)


def _error_object(parsed: Any) -> dict[str, Any]:
    """The dict holding the provider error, whatever the envelope."""
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return {}
    inner = parsed.get("error")
    if isinstance(inner, dict):
        return inner
    return parsed


def _extract_message(error: dict[str, Any]) -> str | None:
    # Anthropic / OpenAI / Azure / Vertex: error.message. Mistral: message.
    message = error.get("message")
    if isinstance(message, str) and message:
        return message
    # Mistral validation errors (422): {"detail": [{"loc": [...], "msg": "..."}]}
    detail = error.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if not isinstance(item, dict) or not isinstance(item.get("msg"), str):
                continue
            loc = item.get("loc")
            where = ".".join(str(p) for p in loc if p != "body") if loc else ""
            parts.append(f"{where}: {item['msg']}" if where else item["msg"])
        if parts:
            return "; ".join(parts)
    return None


def _kind_from_status(status_code: int) -> ProviderErrorKind:
    if status_code in (400, 409, 413, 422):
        return ProviderErrorKind.INVALID_REQUEST
    if status_code == 401:
        return ProviderErrorKind.AUTHENTICATION
    if status_code == 403:
        return ProviderErrorKind.PERMISSION
    if status_code == 404:
        return ProviderErrorKind.NOT_FOUND
    if status_code == 429:
        return ProviderErrorKind.RATE_LIMITED
    if status_code >= 500:
        return ProviderErrorKind.UNAVAILABLE
    return ProviderErrorKind.UNKNOWN


def classify_provider_error(
    status_code: int, body: str
) -> tuple[ProviderErrorKind, bool, str | None]:
    """Return ``(kind, retryable, provider_message)`` for a provider response.

    The HTTP status gives the generic classification; the body, when it is
    JSON, refines it (Google status, policy refusals) and yields the message.
    """
    try:
        error = _error_object(json.loads(body))
    except ValueError:
        error = {}

    kind = _kind_from_status(status_code)
    google_status = error.get("status")
    if kind != ProviderErrorKind.UNKNOWN and isinstance(google_status, str):
        kind = _GOOGLE_STATUS_KIND.get(google_status, kind)

    provider_message = _extract_message(error)

    if kind in (ProviderErrorKind.INVALID_REQUEST, ProviderErrorKind.PERMISSION):
        haystack = f"{provider_message or ''} {body}".lower()
        if any(marker in haystack for marker in _POLICY_MARKERS):
            kind = ProviderErrorKind.POLICY

    retryable = status_code == 429 or status_code >= 500
    return kind, retryable, provider_message


class ProviderError(FuncLLMError):
    """Error response from a provider.

    ``status_code`` and ``body`` are the raw response. ``kind``, ``retryable``
    and ``provider_message`` are derived from them unless given explicitly.
    """

    def __init__(
        self,
        status_code: int,
        body: str,
        *,
        kind: ProviderErrorKind | None = None,
        retryable: bool | None = None,
        provider_message: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        derived_kind, derived_retryable, derived_message = classify_provider_error(
            status_code, body
        )
        self.kind = kind if kind is not None else derived_kind
        self.retryable = retryable if retryable is not None else derived_retryable
        self.provider_message = (
            provider_message if provider_message is not None else derived_message
        )
        super().__init__(f"Provider returned {status_code}: {body}")


class SerializationError(FuncLLMError):
    pass


class MediaResolutionError(FuncLLMError):
    def __init__(self, failed_ids: list[str] | None = None, msg: str = "") -> None:
        self.failed_ids = failed_ids
        if not msg:
            msg = "Media resolution failed"
            if failed_ids:
                msg += f" for IDs: {', '.join(failed_ids)}"
        super().__init__(msg)


class StructuredOutputError(FuncLLMError):
    def __init__(self, output: GenerationOutput, msg: str) -> None:
        self.output = output
        super().__init__(msg)
